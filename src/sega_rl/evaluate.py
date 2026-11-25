from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from sega_rl.config import Settings
from sega_rl.envs.factory import make_single
from sega_rl.monitoring import mlflow_tracking
from sega_rl.monitoring.video import write_mp4

logger = logging.getLogger(__name__)

@dataclass
class EvalResult:
    win_rate: float
    mean_damage_dealt: float
    mean_damage_taken: float
    mean_return: float
    mean_progress: float = math.nan
    rounds_done: int = 0  # sample size behind win_rate (sf)
    train_win_rate: float = math.nan  # sonic multi: on levels it trains on (win_rate = held-out levels)
    train_progress: float = math.nan


def run_eval(model, cfg: Settings, episodes: int, video_path: Path | None = None,
             max_steps: int | None = None, heartbeat: Callable[[], None] | None = None,
             deterministic: bool = True) -> EvalResult:
    """Episodes in one env, taking the likeliest action or sampling like training does (the training process hosts no emulator, so this is allowed)."""
    if cfg.game == "sonic" and cfg.sonic_levels in ("multi", "all") and cfg.state is None:
        return run_levels_eval(model, cfg, video_path, max_steps, heartbeat, deterministic=deterministic)
    if cfg.game == "sf" and cfg.sf_states_dir is not None and cfg.state is None:
        return run_fights_eval(model, cfg, video_path, max_steps, heartbeat, deterministic=deterministic)
    max_steps = max_steps or cfg.eval_max_steps
    venv = VecFrameStack(DummyVecEnv([make_single(cfg, rank=10_000)]), cfg.frame_stack, channels_order="last")
    raw_env = venv.venv.envs[0]
    rounds_won = rounds_done = levels_done = 0
    returns, dealt, taken, progress = [], [], [], []
    try:
        for ep in range(episodes):
            obs = venv.reset()
            frames = [raw_env.render()] if ep == 0 and video_path else None
            ep_ret = ep_dealt = ep_taken = 0.0
            last_info: dict = {}
            for _ in range(max_steps):
                if heartbeat is not None:
                    heartbeat()
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, done, infos = venv.step(action)
                last_info = infos[0]
                ep_ret += float(reward[0])
                ep_dealt += last_info.get("damage_dealt", 0)
                ep_taken += last_info.get("damage_taken", 0)
                if last_info.get("round_won") is not None:
                    rounds_done += 1
                    rounds_won += int(last_info["round_won"])
                if frames is not None:
                    frames.append(raw_env.render())
                if done[0]:
                    break
            levels_done += int(bool(last_info.get("level_complete")))
            returns.append(ep_ret)
            dealt.append(ep_dealt)
            taken.append(ep_taken)
            if "progress" in last_info:
                progress.append(last_info["progress"])
            if frames is not None:
                write_mp4(frames, video_path)
    finally:
        venv.close()

    if cfg.game == "sf":
        win_rate = rounds_won / rounds_done if rounds_done else 0.0
    elif cfg.game == "sonic":
        win_rate = levels_done / episodes
    else:
        win_rate = math.nan
    return EvalResult(win_rate=win_rate, mean_damage_dealt=float(np.mean(dealt)),
                      mean_damage_taken=float(np.mean(taken)), mean_return=float(np.mean(returns)),
                      mean_progress=float(np.mean(progress)) if progress else math.nan,
                      rounds_done=rounds_done)


def run_levels_eval(model, cfg: Settings, video_path: Path | None = None, max_steps: int | None = None,
                    heartbeat: Callable[[], None] | None = None,
                    eval_fn: Callable[..., EvalResult] | None = None, deterministic: bool = True) -> EvalResult:
    """Sonic. "multi": the 4 held-out levels (the score, with a video each) and 4 training levels. "all": every level."""
    from sega_rl.envs.sonic import ALL_LEVELS, HELD_OUT, TRAIN_EVAL

    every = cfg.sonic_levels == "all"
    levels = ALL_LEVELS if every else (*HELD_OUT, *TRAIN_EVAL)
    filmed = TRAIN_EVAL if every else HELD_OUT
    results = {}
    for level in levels:
        video = video_path.with_name(f"{video_path.stem}_{level}.mp4") if video_path and level in filmed else None
        results[level] = (eval_fn or run_eval)(model, cfg.model_copy(update={"state": level}),
                                               cfg.sonic_level_episodes, video_path=video, max_steps=max_steps,
                                               heartbeat=heartbeat, deterministic=deterministic)
        logger.info("level %s: %s", level, results[level])
    mlflow_tracking.set_tag("eval_levels", json.dumps({k: [round(r.win_rate, 2), round(r.mean_progress, 2)]
                                                       for k, r in results.items()}))

    def mean(levels, field) -> float:
        return float(np.mean([getattr(results[k], field) for k in levels]))

    if every:
        return EvalResult(win_rate=mean(levels, "win_rate"), mean_damage_dealt=0.0, mean_damage_taken=0.0,
                          mean_return=mean(levels, "mean_return"), mean_progress=mean(levels, "mean_progress"))
    return EvalResult(win_rate=mean(HELD_OUT, "win_rate"), mean_damage_dealt=0.0, mean_damage_taken=0.0,
                      mean_return=mean(HELD_OUT, "mean_return"), mean_progress=mean(HELD_OUT, "mean_progress"),
                      train_win_rate=mean(TRAIN_EVAL, "win_rate"), train_progress=mean(TRAIN_EVAL, "mean_progress"))


def run_fights_eval(model, cfg: Settings, video_path: Path | None = None, max_steps: int | None = None,
                    heartbeat: Callable[[], None] | None = None, eval_fn: Callable[..., EvalResult] | None = None,
                    deterministic: bool = True) -> EvalResult:
    """Street Fighter: rounds won per opponent. Score = opponents never trained on; train_* = the others."""
    from sega_rl.envs.street_fighter import opponents

    has_heldout = (cfg.sf_states_dir / "heldout").is_dir()
    split = {"heldout": opponents(cfg, "heldout") if has_heldout else [], "train": opponents(cfg, "train")}
    per: dict[str, list[EvalResult]] = {"heldout": [], "train": []}
    for name, states in split.items():
        for state in states:
            who = Path(state).stem
            filmed = name == "heldout" or not has_heldout
            video = video_path.with_name(f"{video_path.stem}_{who}.mp4") if video_path and filmed else None
            r = (eval_fn or run_eval)(model, cfg.model_copy(update={"state": state}), cfg.sf_fight_episodes,
                                      video_path=video, max_steps=max_steps, heartbeat=heartbeat,
                                      deterministic=deterministic)
            logger.info("opponent %s (%s): %s", who, name, r)
            per[name].append(r)

    def mean(results, field) -> float:
        return float(np.mean([getattr(r, field) for r in results]))

    if not has_heldout:
        return EvalResult(win_rate=mean(per["train"], "win_rate"), mean_damage_dealt=mean(per["train"], "mean_damage_dealt"),
                          mean_damage_taken=mean(per["train"], "mean_damage_taken"),
                          mean_return=mean(per["train"], "mean_return"), rounds_done=sum(r.rounds_done for r in per["train"]))
    return EvalResult(win_rate=mean(per["heldout"], "win_rate"), mean_damage_dealt=mean(per["heldout"], "mean_damage_dealt"),
                      mean_damage_taken=mean(per["heldout"], "mean_damage_taken"),
                      mean_return=mean(per["heldout"], "mean_return"), rounds_done=sum(r.rounds_done for r in per["heldout"]),
                      train_win_rate=mean(per["train"], "win_rate"))


class StandStill:
    """Presses nothing. If this scores, the reward or the metric is wrong."""

    def __init__(self, action_space):
        self.shape = getattr(action_space, "shape", ()) or ()

    def predict(self, obs, deterministic=True):
        return np.zeros((len(obs), *self.shape), dtype=np.int64), None


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs, deterministic=True):
        return np.array([self.action_space.sample() for _ in range(len(obs))]), None


def run_baselines(cfg: Settings, run_dir: Path, action_space, eval_fn: Callable[..., EvalResult] | None = run_eval,
                  heartbeat: Callable[[], None] | None = None) -> dict[str, float]:
    """Measure stand-still and random play once per run, so every win rate has something to beat."""
    path = Path(run_dir) / "baselines.json"
    if path.exists():
        return {}
    out: dict[str, float] = {}
    for name, policy in (("stand_still", StandStill(action_space)), ("random", RandomPolicy(action_space))):
        result = eval_fn(policy, cfg, cfg.n_eval_episodes, heartbeat=heartbeat)
        logger.info("baseline %s: %s", name, result)
        out.update({f"baseline/{name}_{k}": v for k, v in asdict(result).items() if not math.isnan(v)})
    path.write_text(json.dumps(out, indent=2, default=float))
    return out


def score(game: str, r: EvalResult) -> tuple[float, ...]:
    """Ordering key for best-model selection (higher is better)."""
    if game == "sf" and not math.isnan(r.train_win_rate):
        return (r.train_win_rate, r.mean_return)  # never pick by the held-out fights
    if game == "sf":
        return (r.win_rate, r.mean_damage_dealt - r.mean_damage_taken)
    if game == "sonic" and not math.isnan(r.train_win_rate):
        return (r.train_win_rate, r.train_progress)  # picking the best model by the held-out levels would be peeking
    if game == "sonic":
        return (r.win_rate, r.mean_progress)
    return (r.mean_return,)


class EvalCallback(BaseCallback):
    """Every cfg.eval_freq steps: deterministic eval + one video; keeps best_model.zip."""

    def __init__(self, cfg: Settings, run_dir: Path, eval_fn: Callable[..., EvalResult] = run_eval,
                 heartbeat: Callable[[], None] | None = None):
        super().__init__()
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.eval_fn = eval_fn
        self.heartbeat = heartbeat
        self.next_eval = cfg.eval_freq
        self.best: tuple[float, ...] | None = self._load_best()

    def _load_best(self) -> tuple[float, ...] | None:
        """On resume, keep the best-so-far bar so a worse model can't overwrite best_model.zip."""
        path = self.run_dir / "best.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        known = {f.name for f in fields(EvalResult)}
        return score(self.cfg.game, EvalResult(**{k: v for k, v in data.items() if k in known}))

    def _on_step(self) -> bool:
        if self.model.num_timesteps < self.next_eval:
            return True
        steps = self.model.num_timesteps
        self.next_eval = (steps // self.cfg.eval_freq + 1) * self.cfg.eval_freq
        video = self.run_dir / "videos" / f"eval_{steps:010d}.mp4"
        result = self.eval_fn(self.model, self.cfg, self.cfg.n_eval_episodes, video_path=video,
                              heartbeat=self.heartbeat)
        for path in sorted(video.parent.glob(f"{video.stem}*.mp4")):
            mlflow_tracking.log_artifact(path, "videos")
        for k, v in asdict(result).items():
            if not math.isnan(v):
                self.logger.record(f"eval/{k}", v)
        if self.cfg.eval_sampled:
            sampled = self.eval_fn(self.model, self.cfg, self.cfg.n_eval_episodes, heartbeat=self.heartbeat,
                                   deterministic=False)
            for k, v in asdict(sampled).items():
                if not math.isnan(v):
                    self.logger.record(f"eval_sampled/{k}", v)
            logger.info("eval sampled @%d: %s", steps, sampled)
        key = score(self.cfg.game, result)
        logger.info("eval @%d: %s", steps, result)
        if self.best is None or key > self.best:
            self.best = key
            self.model.save(self.run_dir / "best_model.zip")
            (self.run_dir / "best.json").write_text(json.dumps({"step": steps, **asdict(result)}, default=float))
            mlflow_tracking.log_artifact(self.run_dir / "best.json")
            mlflow_tracking.log_artifact(self.run_dir / "best_model.zip")
        return True

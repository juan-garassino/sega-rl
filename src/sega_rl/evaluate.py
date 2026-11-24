from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
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


def run_eval(model, cfg: Settings, episodes: int, video_path: Path | None = None,
             max_steps: int | None = None) -> EvalResult:
    """Deterministic episodes in one env (the training process hosts no emulator, so this is allowed)."""
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
                action, _ = model.predict(obs, deterministic=True)
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
                      mean_progress=float(np.mean(progress)) if progress else math.nan)


def score(game: str, r: EvalResult) -> tuple[float, ...]:
    """Ordering key for best-model selection (higher is better)."""
    if game == "sf":
        return (r.win_rate, r.mean_damage_dealt - r.mean_damage_taken)
    if game == "sonic":
        return (r.win_rate, r.mean_progress)
    return (r.mean_return,)


class EvalCallback(BaseCallback):
    """Every cfg.eval_freq steps: deterministic eval + one video; keeps best_model.zip."""

    def __init__(self, cfg: Settings, run_dir: Path, eval_fn: Callable[..., EvalResult] = run_eval):
        super().__init__()
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.eval_fn = eval_fn
        self.next_eval = cfg.eval_freq
        self.best: tuple[float, ...] | None = None

    def _on_step(self) -> bool:
        if self.model.num_timesteps < self.next_eval:
            return True
        steps = self.model.num_timesteps
        self.next_eval = (steps // self.cfg.eval_freq + 1) * self.cfg.eval_freq
        video = self.run_dir / "videos" / f"eval_{steps:010d}.mp4"
        result = self.eval_fn(self.model, self.cfg, self.cfg.eval_episodes, video_path=video)
        if video.exists():
            mlflow_tracking.log_artifact(video, "videos")
        for k, v in asdict(result).items():
            if not math.isnan(v):
                self.logger.record(f"eval/{k}", v)
        key = score(self.cfg.game, result)
        logger.info("eval @%d: %s", steps, result)
        if self.best is None or key > self.best:
            self.best = key
            self.model.save(self.run_dir / "best_model.zip")
            (self.run_dir / "best.json").write_text(json.dumps({"step": steps, **asdict(result)}, default=float))
            mlflow_tracking.log_artifact(self.run_dir / "best.json")
        return True

"""One short pod run that answers two questions before paying for training (writes bench.json + moves.mp4):

1. Is training limited by the emulators or by the network update, and how many torch threads are best?
2. Does every Street Fighter move really work with these inputs? (the Hadouken must hit from across the screen)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from sega_rl.config import Settings
from sega_rl.envs.common import RetroEnv
from sega_rl.monitoring.video import write_mp4

logger = logging.getLogger(__name__)
THREADS = (4, 8)
N_ENVS = (16, 28)
REPEATS = 8
INTRO_STEPS = 60  # the game ignores input until after FIGHT!
NEAR_STEPS = 60  # RIGHT held this long pushes Guile to the corner


class SplitTimer(BaseCallback):
    """Seconds spent collecting steps vs updating the network, over the rollouts after the first."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        super().__init__()
        self.clock = clock
        self.collect, self.update = [], []
        self.started: float | None = None
        self.ended: float | None = None

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        now = self.clock()
        if self.ended is not None:
            self.update.append(now - self.ended)
        self.started = now

    def _on_rollout_end(self) -> None:
        self.ended = self.clock()
        self.collect.append(self.ended - self.started)


def time_training(cfg: Settings, rollouts: int = 4) -> dict[str, dict[str, float]]:
    from sega_rl.envs.factory import make_vec_env
    from sega_rl.train import build_model

    env = make_vec_env(cfg)
    out: dict[str, dict[str, float]] = {}
    try:
        env.reset()
        t = time.perf_counter()
        for _ in range(100):
            env.step(np.array([env.action_space.sample() for _ in range(env.num_envs)]))
        out["emulators_only"] = {"fps": 100 * env.num_envs / (time.perf_counter() - t)}
        for n in THREADS:
            model = build_model(cfg, env)
            torch.set_num_threads(n)
            timer = SplitTimer()
            model.learn(total_timesteps=rollouts * model.n_steps * env.num_envs, callback=timer)
            collect, update = float(np.mean(timer.collect[1:])), float(np.mean(timer.update))
            steps = model.n_steps * env.num_envs
            out[f"threads_{n}"] = {"collect_s": collect, "update_s": update, "fps": steps / (collect + update)}
            logger.info("threads %d: %s", n, out[f"threads_{n}"])
    finally:
        env.close()
    return out


def try_moves(env: RetroEnv, moves: list[str], names: list[str], repeats: int = REPEATS,
              frames: list | None = None) -> dict[str, dict[str, float]]:
    """Damage per use, from the start distance and from up close. Each use is a fresh game."""
    right, idle = moves.index("right"), moves.index("idle")
    out: dict[str, dict[str, float]] = {}
    for name in names:
        row = {}
        for where, walk in (("far", 0), ("near", NEAR_STEPS)):
            damage = 0.0
            for i in range(repeats):
                env.reset(seed=i)
                script = [idle] * INTRO_STEPS + [right] * walk + [moves.index(name)] + [idle] * 20  # a fireball needs time to arrive
                for j, action in enumerate(script):
                    _, _, done, truncated, info = env.step(action)
                    if j >= INTRO_STEPS + walk:
                        damage += info.get("damage_dealt", 0)
                    if frames is not None and i == 0:
                        frames.append(env.render())
                    if done or truncated:
                        break
            row[where] = damage / repeats
        out[name] = row
        logger.info("%s: %s", name, row)
    return out


def compare_play(cfg: Settings, load: Callable | None = None, play: Callable | None = None,
                 episodes: int = 20) -> dict[str, dict]:
    """Win rate of the run's newest checkpoint playing its likeliest move vs sampling (how it trains)."""
    from stable_baselines3 import PPO

    from sega_rl.evaluate import run_eval

    ckpt = sorted((cfg.run_dir / "checkpoints").glob("ckpt_*.zip"))[-1]
    model = (load or PPO.load)(ckpt)
    out: dict[str, dict] = {}
    for name, deterministic in (("likeliest", True), ("sampled", False)):
        result = (play or run_eval)(model, cfg, episodes, deterministic=deterministic,
                                    video_path=cfg.run_dir / "videos" / f"compare_{name}.mp4")
        out[name] = {"checkpoint": ckpt.name, **asdict(result)}
        logger.info("%s: %s", name, out[name])
        (cfg.run_dir / "compare.json").write_text(json.dumps(out, indent=2, default=float))
    return out


def run(cfg: Settings) -> int:
    from sega_rl.envs import street_fighter as sf

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"training": {f"envs_{n}": time_training(cfg.model_copy(update={"n_envs": n})) for n in N_ENVS}}
    if cfg.game == "sf":
        moves = list(sf.FIGHTER_MOVES)
        names = ["idle", *(moves[i] for i in sorted(sf.FIGHTER_ATTACKS))]
        frames: list = []
        env = sf.make(cfg.model_copy(update={"sf_actions": "fighter"}))
        try:
            report["moves"] = try_moves(env, moves, names, frames=frames)
        finally:
            env.close()
        write_mp4(frames, cfg.run_dir / "videos" / "moves.mp4")
    (cfg.run_dir / "bench.json").write_text(json.dumps(report, indent=2))
    (cfg.run_dir / "DONE").write_text("done")
    print(json.dumps(report))
    return 0

from __future__ import annotations

from typing import Callable

import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, VecFrameStack, VecMonitor

from sega_rl.config import Settings


def make_single(cfg: Settings, rank: int) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        from sega_rl.envs import airstriker, sonic, street_fighter

        makers = {"sf": street_fighter.make, "sonic": sonic.make, "airstriker": airstriker.make}
        env = makers[cfg.game](cfg)
        env.reset(seed=cfg.seed + rank)
        return env

    return _init


def make_vec_env(cfg: Settings, n_envs: int | None = None) -> VecEnv:
    n = n_envs or cfg.n_envs
    venv = SubprocVecEnv([make_single(cfg, i) for i in range(n)], start_method="forkserver")
    return VecFrameStack(VecMonitor(venv), cfg.vec_stack, channels_order="last")

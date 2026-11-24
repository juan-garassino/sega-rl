from __future__ import annotations

from sega_rl.config import Settings
from sega_rl.envs.common import RetroEnv

GAME_ID = "Airstriker-Genesis-v0"


def score_reward(prev: dict, info: dict) -> tuple[float, bool]:
    return (info.get("score", 0) - prev.get("score", info.get("score", 0))) / 100.0, False


def make(cfg: Settings) -> RetroEnv:
    return RetroEnv(GAME_ID, cfg.state, score_reward, combos=None,
                    obs_mode=cfg.obs_mode, frame_skip=cfg.frame_skip, noop_max=cfg.noop_max)

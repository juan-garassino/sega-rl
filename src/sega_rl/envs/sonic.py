from __future__ import annotations

from sega_rl.config import Settings
from sega_rl.envs.common import RetroEnv

GAME_ID = "SonicTheHedgehog-Genesis-v0"
DEFAULT_STATE = "GreenHillZone.Act1"
MAX_STEPS = 4500
LEVEL_BONUS = 10.0

# OpenAI Retro Contest baseline action set.
SONIC_ACTIONS: list[list[str]] = [[], ["LEFT"], ["RIGHT"], ["LEFT", "DOWN"], ["RIGHT", "DOWN"], ["DOWN"], ["B"]]

# (zone, act) -> x at which the act ends; from the integration's script.lua `level_max_x`.
LEVEL_MAX_X: dict[tuple[int, int], int] = {
    (0, 0): 0x2560, (0, 1): 0x1F60, (0, 2): 0x292A,
    (2, 0): 0x1860, (2, 1): 0x1860, (2, 2): 0x1720,
    (4, 0): 0x2360, (4, 1): 0x2960, (4, 2): 0x2B83,
    (1, 0): 0x1A50, (1, 1): 0x1150, (1, 2): 0x1CC4,
    (3, 0): 0x2060, (3, 1): 0x2060, (3, 2): 0x1F48,
    (5, 0): 0x2260, (5, 1): 0x1EE0,
}


class SonicReward:
    """Reward = new ground reached (Δ furthest x)/100, +10 at the act's end. Ends the episode on a lost life."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.start_x: int | None = None
        self.max_x = 0

    def __call__(self, prev: dict, info: dict) -> tuple[float, bool]:
        x = info["x"]
        end_x = LEVEL_MAX_X.get((info.get("zone", 0), info.get("act", 0)))
        if self.start_x is None:
            self.start_x = self.max_x = x
        reward = max(0, x - self.max_x) / 100.0
        self.max_x = max(self.max_x, x)
        complete = end_x is not None and x >= end_x
        info["max_x"] = self.max_x
        info["level_complete"] = complete
        if end_x is not None and end_x > self.start_x:
            info["progress"] = min(1.0, (self.max_x - self.start_x) / (end_x - self.start_x))
        life_lost = bool(prev) and info.get("lives", 0) < prev.get("lives", 0)
        if complete:
            reward += LEVEL_BONUS
        return reward, complete or life_lost


def make(cfg: Settings) -> RetroEnv:
    return RetroEnv(GAME_ID, cfg.state or DEFAULT_STATE, SonicReward(), combos=SONIC_ACTIONS,
                    obs_mode=cfg.obs_mode, frame_skip=cfg.frame_skip, max_steps=MAX_STEPS)

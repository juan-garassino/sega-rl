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


GENESIS_FPS = 60


def stall_frames(cfg: Settings) -> int:
    return int(cfg.sonic_stall_secs * GENESIS_FPS)


class SonicReward:
    """Reward = new ground reached (Δ furthest x)/100, +10 at the act's end.

    Ends the episode on a lost life, or after `stall_frames` raw frames without new ground: standing in
    front of an obstacle earns nothing, so a stuck episode only burns compute (eval videos showed 4 min stuck).
    """

    def __init__(self, stall_frames: int | None = None):
        self.stall_frames = stall_frames
        self.reset()

    def reset(self) -> None:
        self.start_x: int | None = None
        self.max_x = 0
        self.frames_since_progress = 0

    def __call__(self, prev: dict, info: dict) -> tuple[float, bool]:
        x = info["x"]
        end_x = LEVEL_MAX_X.get((info.get("zone", 0), info.get("act", 0)))
        if self.start_x is None:
            self.start_x = self.max_x = x
        reward = max(0, x - self.max_x) / 100.0
        first_frame = self.start_x == x and self.max_x == x and self.frames_since_progress == 0 and not prev
        self.frames_since_progress = 0 if (x > self.max_x or first_frame) else self.frames_since_progress + 1
        self.max_x = max(self.max_x, x)
        stalled = self.stall_frames is not None and self.frames_since_progress >= self.stall_frames
        info["stalled"] = stalled
        complete = end_x is not None and x >= end_x
        info["max_x"] = self.max_x
        info["level_complete"] = complete
        if end_x is not None and end_x > self.start_x:
            info["progress"] = min(1.0, (self.max_x - self.start_x) / (end_x - self.start_x))
        life_lost = bool(prev) and info.get("lives", 0) < prev.get("lives", 0)
        if complete:
            reward += LEVEL_BONUS
        return reward, complete or life_lost or stalled


def make(cfg: Settings) -> RetroEnv:
    return RetroEnv(GAME_ID, cfg.state or DEFAULT_STATE, SonicReward(stall_frames(cfg)), combos=SONIC_ACTIONS,
                    obs_mode=cfg.obs_mode, frame_skip=cfg.frame_skip, noop_max=cfg.noop_max, max_steps=MAX_STEPS)

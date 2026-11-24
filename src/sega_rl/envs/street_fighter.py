from __future__ import annotations

from typing import Literal

from sega_rl.config import Settings
from sega_rl.envs.common import RetroEnv

GAME_ID = "StreetFighterIISpecialChampionEdition-Genesis-v0"
DEFAULT_STATE = "Champion.Level1.RyuVsGuile"
MAX_HEALTH = 176


class SFReward:
    """Shaped: (damage dealt - damage taken)/176, +1 round won, -1 round lost. Score: upstream score delta."""

    def __init__(self, mode: Literal["shaped", "score"] = "shaped"):
        self.mode = mode

    def reset(self) -> None:
        pass

    def __call__(self, prev: dict, info: dict) -> tuple[float, bool]:
        if not prev:
            info.setdefault("round_won", None)
            return 0.0, False
        # Health refills at round start; only decreases count as damage.
        dealt = max(0, prev["enemy_health"] - info["enemy_health"])
        taken = max(0, prev["health"] - info["health"])
        won = info["matches_won"] > prev["matches_won"]
        lost = info["enemy_matches_won"] > prev["enemy_matches_won"]
        info["damage_dealt"] = dealt
        info["damage_taken"] = taken
        info["round_won"] = True if won else False if lost else None
        if self.mode == "score":
            return float(info["score"] - prev["score"]), False
        return (dealt - taken) / MAX_HEALTH + (1.0 if won else 0.0) - (1.0 if lost else 0.0), False


def make(cfg: Settings) -> RetroEnv:
    return RetroEnv(GAME_ID, cfg.state or DEFAULT_STATE, SFReward(cfg.reward_mode), combos=None,
                    obs_mode=cfg.obs_mode, frame_skip=cfg.frame_skip)

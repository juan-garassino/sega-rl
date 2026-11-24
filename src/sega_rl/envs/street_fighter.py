from __future__ import annotations

from typing import Literal

from sega_rl.config import Settings
from sega_rl.envs.common import RetroEnv

GAME_ID = "StreetFighterIISpecialChampionEdition-Genesis-v0"
DEFAULT_STATE = "Champion.Level1.RyuVsGuile"
MAX_HEALTH = 176

# Genesis 6-button pad: X/Y/Z = light/medium/heavy punch, A/B/C = light/medium/heavy kick.
# Independent MultiBinary buttons press UP half the time (the agent jumped constantly);
# these 21 moves keep it on the ground unless it chooses to jump.
SF_ACTIONS: list[list[str]] = [
    [], ["LEFT"], ["RIGHT"], ["DOWN"], ["DOWN", "LEFT"], ["DOWN", "RIGHT"],
    ["UP"], ["UP", "LEFT"], ["UP", "RIGHT"],
    ["X"], ["Y"], ["Z"], ["A"], ["B"], ["C"],
    ["DOWN", "X"], ["DOWN", "Y"], ["DOWN", "Z"], ["DOWN", "A"], ["DOWN", "B"], ["DOWN", "C"],
]


class SFReward:
    """Shaped: (w_d·dealt − w_t·taken)/176, +1 round won, −1 round lost. Score: upstream score delta."""

    def __init__(self, mode: Literal["shaped", "score"] = "shaped", dealt_weight: float = 1.0,
                 taken_weight: float = 1.0):
        self.mode = mode
        self.dealt_weight = dealt_weight
        self.taken_weight = taken_weight
        self.reset()

    def reset(self) -> None:
        self.between_rounds = False

    def __call__(self, prev: dict, info: dict) -> tuple[float, bool]:
        if not prev:
            info.setdefault("round_won", None)
            return 0.0, False
        # KO is -1. Between rounds the game clears both bars to 0 and then refills them; neither is damage.
        cleared = info["health"] == 0 and info["enemy_health"] == 0
        dealt = max(0, max(0, prev["enemy_health"]) - max(0, info["enemy_health"]))
        taken = max(0, max(0, prev["health"]) - max(0, info["health"]))
        if self.between_rounds or cleared:
            dealt = taken = 0
        won = info["matches_won"] > prev["matches_won"]
        lost = info["enemy_matches_won"] > prev["enemy_matches_won"]
        if won or lost:
            self.between_rounds = True
        elif info["health"] == MAX_HEALTH and info["enemy_health"] == MAX_HEALTH:
            self.between_rounds = False
        info["damage_dealt"] = dealt
        info["damage_taken"] = taken
        info["round_won"] = True if won else False if lost else None
        if self.mode == "score":
            return float(info["score"] - prev["score"]), False
        shaped = (self.dealt_weight * dealt - self.taken_weight * taken) / MAX_HEALTH
        return shaped + (1.0 if won else 0.0) - (1.0 if lost else 0.0), False


def make(cfg: Settings) -> RetroEnv:
    reward = SFReward(cfg.reward_mode, dealt_weight=cfg.sf_dealt_weight, taken_weight=cfg.sf_taken_weight)
    combos = SF_ACTIONS if cfg.sf_actions == "discrete" else None
    return RetroEnv(GAME_ID, cfg.state or DEFAULT_STATE, reward, combos=combos,
                    obs_mode=cfg.obs_mode, frame_skip=cfg.frame_skip, noop_max=cfg.noop_max)

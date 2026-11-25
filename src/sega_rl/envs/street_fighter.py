from __future__ import annotations

from collections import deque
from typing import Literal

from sega_rl.config import Settings
from sega_rl.envs.common import Move, RetroEnv

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


def _mirror(move: Move) -> Move:
    flip = {"LEFT": "RIGHT", "RIGHT": "LEFT"}
    return [([flip.get(b, b) for b in names], n) for names, n in move]


# Facing right; ↓↘→ starts from neutral, or a walk forward before it makes it the Shoryuken motion. The game data has no positions, so every directional move exists both ways and the CNN picks.
_RIGHT_MOVES: dict[str, Move] = {
    "hadouken": [([], 10), (["DOWN"], 3), (["DOWN", "RIGHT"], 3), (["RIGHT"], 2), (["RIGHT", "Y"], 4)],
    "shoryuken": [(["RIGHT"], 3), (["DOWN"], 3), (["DOWN", "RIGHT"], 2), (["DOWN", "RIGHT", "Z"], 4)],
    "tatsumaki": [([], 10), (["DOWN"], 3), (["DOWN", "LEFT"], 3), (["LEFT"], 2), (["LEFT", "B"], 4)],
    "forward_hp": [(["RIGHT", "Z"], 4)],  # a throw when close
    "forward_hk": [(["RIGHT", "C"], 4)],
    "jump_kick": [(["UP", "RIGHT"], 4), ([], 16), (["C"], 6)],
    "jump_punch": [(["UP", "RIGHT"], 4), ([], 16), (["Z"], 6)],
}
FIGHTER_MOVES: dict[str, Move] = {
    "idle": [], "left": ["LEFT"], "right": ["RIGHT"], "crouch": ["DOWN"],
    "block_low_left": ["DOWN", "LEFT"], "block_low_right": ["DOWN", "RIGHT"],
    "up": ["UP"], "jump_left": ["UP", "LEFT"], "jump_right": ["UP", "RIGHT"],
    "lp": ["X"], "mp": ["Y"], "hp": ["Z"], "lk": ["A"], "mk": ["B"], "hk": ["C"],
    "low_mp": ["DOWN", "Y"], "uppercut": ["DOWN", "Z"], "low_lk": ["DOWN", "A"], "low_mk": ["DOWN", "B"],
    "sweep": ["DOWN", "C"],
    **{f"{name}_right": m for name, m in _RIGHT_MOVES.items()},
    **{f"{name}_left": _mirror(m) for name, m in _RIGHT_MOVES.items()},
}
_MOVEMENT = {"idle", "left", "right", "crouch", "block_low_left", "block_low_right", "up", "jump_left", "jump_right"}
FIGHTER_ATTACKS = frozenset(i for i, name in enumerate(FIGHTER_MOVES) if name not in _MOVEMENT)


class SFReward:
    """Shaped: (w_d·dealt − w_t·taken)/176, round won +1 (+health left), lost −1. Score: upstream score delta.

    With `attacks`, landing a move that already landed in the last 4 hits pays half as much each time (min ¼),
    so one safe poke is not the best plan. `info["damage_dealt"]` always reports the real damage.
    """

    def __init__(self, mode: Literal["shaped", "score"] = "shaped", dealt_weight: float = 1.0,
                 taken_weight: float = 1.0, win_health_bonus: float = 0.0,
                 attacks: frozenset[int] = frozenset(), single_match: bool = False):
        self.mode = mode
        self.single_match = single_match  # end at the first match won or lost: one opponent per game
        self.win_health_bonus = win_health_bonus
        self.attacks = attacks
        self.dealt_weight = dealt_weight
        self.taken_weight = taken_weight
        self.reset()

    def reset(self) -> None:
        self.between_rounds = False
        self.health = MAX_HEALTH
        self.use: tuple[int, int] | None = None  # (move, move_id) of the last attack chosen
        self.paid_use: tuple[int, int] | None = None  # the use whose freshness is already set
        self.freshness = 1.0
        self.landed: deque[int] = deque(maxlen=4)

    def _freshness(self) -> float:
        """Set once per use of a move: its damage may drain over several frames."""
        if self.use is not None and self.use != self.paid_use:
            self.freshness = max(0.25, 0.5 ** self.landed.count(self.use[0]))
            self.landed.append(self.use[0])
            self.paid_use = self.use
        return self.freshness

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
        elif info["health"] > 0:
            self.health = info["health"]  # what is left when the KO lands; the bars are cleared before the win is counted
        if info.get("move") in self.attacks:
            self.use = (info["move"], info["move_id"])
        paid = dealt * self._freshness() if dealt and self.attacks else dealt
        won = info["matches_won"] > prev["matches_won"]
        lost = info["enemy_matches_won"] > prev["enemy_matches_won"]
        if won or lost:
            self.between_rounds = True
        elif info["health"] == MAX_HEALTH and info["enemy_health"] == MAX_HEALTH:
            self.between_rounds = False
        info["damage_dealt"] = dealt
        info["damage_taken"] = taken
        info["round_won"] = True if won else False if lost else None
        over = self.single_match and max(info["matches_won"], info["enemy_matches_won"]) >= 2
        if self.mode == "score":
            return float(info["score"] - prev["score"]), over
        shaped = (self.dealt_weight * paid - self.taken_weight * taken) / MAX_HEALTH
        bonus = 1.0 + self.win_health_bonus * self.health / MAX_HEALTH
        return shaped + (bonus if won else 0.0) - (1.0 if lost else 0.0), over


def opponents(cfg: Settings, split: str) -> list[str] | None:
    """Start states of the opponents to train on ("train") or only test on ("heldout"), from cfg.sf_states_dir."""
    if cfg.sf_states_dir is None:
        return None
    found = sorted(str(p) for p in (cfg.sf_states_dir / split).glob("*.state"))
    if not found:
        raise FileNotFoundError(f"no *.state in {cfg.sf_states_dir / split}")
    return found


def make(cfg: Settings) -> RetroEnv:
    fighter = cfg.sf_actions == "fighter"
    reward = SFReward(cfg.reward_mode, dealt_weight=cfg.sf_dealt_weight, taken_weight=cfg.sf_taken_weight,
                      win_health_bonus=cfg.sf_win_health_bonus, attacks=FIGHTER_ATTACKS if fighter and cfg.sf_stale else frozenset(),
                      single_match=cfg.sf_states_dir is not None)
    combos = list(FIGHTER_MOVES.values()) if fighter else SF_ACTIONS if cfg.sf_actions == "discrete" else None
    states = opponents(cfg, "train") if cfg.state is None else None
    return RetroEnv(GAME_ID, cfg.state or DEFAULT_STATE, reward, combos=combos, states=states, weak_spots=cfg.weak_spots,
                    log_dir=None if states is None else cfg.run_dir / "achievements",
                    obs_mode=cfg.obs_mode, frame_skip=cfg.frame_skip, noop_max=cfg.noop_max)

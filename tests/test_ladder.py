"""ladder.py on a fake Street Fighter: no emulator, no ROM."""
from __future__ import annotations

import gzip

import numpy as np

from sega_rl.ladder import FULL, climb, save_states


class _Em:
    def __init__(self, game):
        self.game = game

    def get_state(self):
        return f"fight-{self.game.fight}".encode()


class _Fight:
    """Two-round matches: any step with health pinned to 1 wins the round; ~50 frames of screens between fights."""

    buttons = ["A", "B", "C"]

    def __init__(self, fights=3):
        self.fights, self.fight, self.won, self.screen = fights, 0, 0, 0
        self.health = {"health": FULL, "enemy_health": FULL}
        self.em = _Em(self)

    def set_value(self, name, val):
        assert name in self.health
        self.health[name] = val

    def step(self, action):
        if self.screen:
            self.screen -= 1
            if not self.screen:
                self.fight, self.won = self.fight + 1, 0
                self.health = {"health": FULL, "enemy_health": FULL}
        elif self.health["enemy_health"] == 1:
            self.won += 1
            self.health = {"health": FULL, "enemy_health": FULL}
            if self.won == 2:
                self.health = {"health": 0, "enemy_health": 0}
                self.screen = 50
        info = {**self.health, "matches_won": self.won, "enemy_matches_won": 0}
        return np.full((8, 8, 3), self.fight, np.uint8), 0.0, False, False, info


def test_climb_saves_one_state_per_fight():
    fights = climb(_Fight(), max_fights=3, max_frames=5000)
    assert [f.state for f in fights] == [b"fight-0", b"fight-1", b"fight-2"]
    assert [int(f.frame[0, 0, 0]) for f in fights] == [0, 1, 2]


def test_climb_gives_up_at_the_frame_limit():
    assert len(climb(_Fight(), max_fights=99, max_frames=200)) < 5


def test_states_are_written_for_stable_retro(tmp_path):
    paths = save_states(climb(_Fight(), max_fights=2, max_frames=5000), tmp_path)
    assert [p.name for p in paths] == ["Ladder.01.state", "Ladder.02.state"]
    assert gzip.open(paths[1]).read() == b"fight-1" and (tmp_path / "Ladder.02.png").exists()

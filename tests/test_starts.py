"""Practice starts (no emulator): weak opponents come up more often, reached points are kept and shared."""
from __future__ import annotations

import gzip

import gymnasium as gym
import numpy as np
import pytest

from sega_rl.envs.common import RetroEnv
from sega_rl.envs.sonic import place
from sega_rl.envs.starts import LevelPoints, WeakSpots


def test_lost_to_opponents_come_up_more_but_nobody_disappears():
    mix, rng = WeakSpots(["Guile", "Sagat"]), np.random.default_rng(0)
    for _ in range(60):
        mix.result("Guile", True)
        mix.result("Sagat", False)
    draws = [mix.pick(rng) for _ in range(1000)]
    assert 0.75 < draws.count("Sagat") / 1000 < 0.87 and draws.count("Guile") > 100


def _walk(points, level, *places, state=b"s"):
    """Visit places, then survive long enough."""
    for place in places:
        points.reached(level, place, lambda: state)
    for _ in range(LevelPoints.SURVIVE):
        points.reached(level, places[-1], lambda: pytest.fail("nothing new here"))


def test_places_are_saved_once_and_shared(tmp_path):
    a, b = LevelPoints(tmp_path, 1.0), LevelPoints(tmp_path, 1.0)
    rng = np.random.default_rng(0)
    assert a.pick(rng, "GreenHillZone.Act1") is None  # nothing reached: the normal start
    _walk(a, "GreenHillZone.Act1", (0, 3), (1, 3), (1, 4), state=b"below")  # down counts as a place too
    assert sorted(p.name for p in tmp_path.iterdir()) == ["GreenHillZone.Act1.x001y003.state", "GreenHillZone.Act1.x001y004.state"]
    assert b.pick(rng, "MarbleZone.Act1") is None
    assert gzip.open(b.pick(rng, "GreenHillZone.Act1")).read() == b"below"
    b.reached("GreenHillZone.Act1", (0, 3), lambda: b"")
    b.reached("GreenHillZone.Act1", (1, 3), lambda: pytest.fail("known place"))


def test_a_game_that_dies_right_after_leaves_no_point(tmp_path):
    p, rng = LevelPoints(tmp_path, 1.0), np.random.default_rng(0)
    p.pick(rng, "L")
    for place in ((0, 0), (1, 0)):
        p.reached("L", place, lambda: b"doomed")
    p.pick(rng, "L")  # the next game
    for _ in range(20):
        p.reached("L", (0, 0), lambda: b"")
    assert not list(tmp_path.iterdir())


def test_practice_spreads_over_places_with_a_share_for_the_furthest(tmp_path):
    p, rng = LevelPoints(tmp_path, 0.5), np.random.default_rng(1)
    for x in range(10):
        (tmp_path / f"L.x{x:03d}y002.state").touch()
    picks = [p.pick(rng, "L") for _ in range(2000)]
    practice = [x.name for x in picks if x is not None]
    assert 920 < len(practice) < 1080
    assert 0.3 < practice.count("L.x009y002.state") / len(practice) < 0.42
    assert min(practice.count(f"L.x{x:03d}y002.state") for x in range(9)) > 40


def test_where_in_a_level():
    assert place({"x": 600, "y": 300}) == (2, 1) and place({"x": 600}) is None


class _Game:
    """A stand-in for the emulator: x grows by 20 per frame; the level is 1000 long."""

    buttons = ["B"]

    class em:
        @staticmethod
        def get_state():
            return b"state"

    def __init__(self):
        self.loaded, self.initial_state = [], None
        self.action_space = gym.spaces.MultiBinary(1)

    def load_state(self, name):
        self.initial_state = name

    def reset(self, seed=None):
        self.loaded.append(self.initial_state)
        self.x = 0
        return np.zeros((224, 320, 3), np.uint8), {"x": 0}

    def step(self, buttons):
        self.x += 20
        return np.zeros((224, 320, 3), np.uint8), 0.0, False, False, {"x": self.x}


def _reward(prev, info):
    info["level_complete"] = info["x"] >= 1000
    return 0.0, info["level_complete"]


def test_a_game_saves_points_then_practises_from_them(tmp_path):
    game = _Game()
    env = RetroEnv("g", None, _reward, [[], ["B"]], "frame", 1, inner=game, states=["L"], weak_spots=True, noop_max=0,
                   log_dir=tmp_path / "a", points=LevelPoints(tmp_path / "p", 1.0), where=lambda info: (info["x"] // 50, 0))
    env.reset(seed=0)
    assert game.loaded == ["L"] and not env.practice
    over = False
    while not over:
        _, _, over, _, _ = env.step(1)
    assert len(list((tmp_path / "p").glob("L.x*.state"))) >= 14  # every place, minus the last ones: not survived long enough
    assert env.mix.rate["L"] > 0.5 and (tmp_path / "a" / "finish_L.json").exists()
    rate = env.mix.rate["L"]
    env.reset(seed=1)
    assert game.loaded[-1] == b"state" and env.practice
    over = False
    while not over:
        _, _, over, _, _ = env.step(1)
    assert env.mix.rate["L"] == rate and (tmp_path / "a" / "practice_finish_L.json").exists()  # practice wins do not count

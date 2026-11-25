"""Practice starts (no emulator): weak opponents come up more often, reached points are kept and shared."""
from __future__ import annotations

import gzip

import gymnasium as gym
import numpy as np
import pytest

from sega_rl.envs.common import RetroEnv
from sega_rl.envs.sonic import level_fraction
from sega_rl.envs.starts import LevelPoints, WeakSpots


def test_lost_to_opponents_come_up_more_but_nobody_disappears():
    mix, rng = WeakSpots(["Guile", "Sagat"]), np.random.default_rng(0)
    for _ in range(60):
        mix.result("Guile", True)
        mix.result("Sagat", False)
    draws = [mix.pick(rng) for _ in range(1000)]
    assert 0.75 < draws.count("Sagat") / 1000 < 0.87 and draws.count("Guile") > 100


def test_points_are_saved_shared_and_limited(tmp_path):
    a, b = LevelPoints(tmp_path, 1.0), LevelPoints(tmp_path, 1.0)
    rng = np.random.default_rng(0)
    assert a.pick(rng, "GreenHillZone.Act1") is None  # nothing reached: the normal start
    a.reached("GreenHillZone.Act1", 0.02, lambda: pytest.fail("the first chunk is the normal start"))
    a.reached("GreenHillZone.Act1", 0.31, lambda: b"at 31%")
    a.reached("GreenHillZone.Act1", 0.31, lambda: pytest.fail("not further than before in this game"))
    assert b.pick(rng, "MarbleZone.Act1") is None
    f = b.pick(rng, "GreenHillZone.Act1")
    assert gzip.open(f).read() == b"at 31%" and b.furthest == 6
    b.reached("GreenHillZone.Act1", 0.30, lambda: pytest.fail("where it started"))
    for pid in ("x", "y"):
        (tmp_path / f"GreenHillZone.Act1.c09-{pid}.state").touch()
    b.reached("GreenHillZone.Act1", 0.47, lambda: pytest.fail("two states of this chunk are enough"))
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_half_the_practice_is_at_the_furthest_point(tmp_path):
    p, rng = LevelPoints(tmp_path, 0.5), np.random.default_rng(1)
    for c in (2, 5, 8, 11):
        (tmp_path / f"L.c{c:02d}-1.state").touch()
    picks = [p.pick(rng, "L") for _ in range(1000)]
    from_start = sum(x is None for x in picks)
    furthest = sum(x is not None and "c11" in x.name for x in picks)
    assert 440 < from_start < 560 and 0.55 < furthest / (1000 - from_start) < 0.70


def test_how_far_along_a_level():
    assert level_fraction({"zone": 0, "act": 0, "x": 0x2560}) == 1.0 and level_fraction({"zone": 99, "act": 0, "x": 5}) is None


class _Game:
    """A stand-in for the emulator: x grows by 100 per frame; the level is 1000 long."""

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
        self.x += 100
        return np.zeros((224, 320, 3), np.uint8), 0.0, False, False, {"x": self.x}


def _reward(prev, info):
    info["level_complete"] = info["x"] >= 1000
    return 0.0, info["level_complete"]


def test_a_game_saves_points_then_practises_from_them(tmp_path):
    game = _Game()
    env = RetroEnv("g", None, _reward, [[], ["B"]], "frame", 1, inner=game, states=["L"], weak_spots=True, noop_max=0,
                   log_dir=tmp_path / "a", points=LevelPoints(tmp_path / "p", 1.0), where=lambda info: info["x"] / 1000)
    env.reset(seed=0)
    assert game.loaded == ["L"] and not env.practice
    over = False
    while not over:
        _, _, over, _, _ = env.step(1)
    assert len(list((tmp_path / "p").glob("L.c*.state"))) == 9  # 10% .. 90%; the end itself is not a start
    assert env.mix.rate["L"] > 0.5 and (tmp_path / "a" / "finish_L.json").exists()
    rate = env.mix.rate["L"]
    env.reset(seed=1)
    assert game.loaded[-1] == b"state" and env.practice
    over = False
    while not over:
        _, _, over, _, _ = env.step(1)
    assert env.mix.rate["L"] == rate and (tmp_path / "a" / "practice_finish_L.json").exists()  # practice wins do not count

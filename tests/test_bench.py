"""bench.py on stubs: no emulator, no training."""
from __future__ import annotations

import numpy as np

from sega_rl.bench import SplitTimer, try_moves


def test_split_timer_separates_collecting_from_updating():
    ticks = iter([0.0, 2.0, 5.0, 7.0])  # collect 2 s, update 3 s, collect 2 s
    t = SplitTimer(clock=lambda: next(ticks))
    for _ in range(2):
        t._on_rollout_start()
        t._on_rollout_end()
    assert t.collect == [2.0, 2.0] and t.update == [3.0]


class _Env:
    """Move 2 does 10 damage, only after walking (near)."""

    def reset(self, seed=None):
        self.walked = 0
        return None, {}

    def step(self, action):
        self.walked += action == 1
        return None, 0.0, False, False, {"damage_dealt": 10 if action == 2 and self.walked else 0}

    def render(self):
        return np.zeros((8, 8, 3), np.uint8)


def test_moves_are_tried_from_far_and_near():
    frames = []
    out = try_moves(_Env(), ["idle", "right", "punch"], ["idle", "punch"], repeats=2, frames=frames)
    assert out == {"idle": {"far": 0.0, "near": 0.0}, "punch": {"far": 0.0, "near": 10.0}} and frames


def test_video_frames_are_enlarged(tmp_path):
    import imageio.v2 as imageio

    from sega_rl.monitoring.video import SCALE, write_mp4

    path = write_mp4([np.zeros((32, 48, 3), np.uint8)] * 3, tmp_path / "v.mp4")
    assert imageio.get_reader(path).get_meta_data()["size"] == (48 * SCALE, 32 * SCALE)

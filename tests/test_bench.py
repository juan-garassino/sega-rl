"""bench.py on stubs: no emulator, no training."""
from __future__ import annotations

import numpy as np

from sega_rl.bench import INTRO_STEPS, SplitTimer, try_moves


def test_split_timer_separates_collecting_from_updating():
    ticks = iter([0.0, 2.0, 5.0, 7.0])  # collect 2 s, update 3 s, collect 2 s
    t = SplitTimer(clock=lambda: next(ticks))
    for _ in range(2):
        t._on_rollout_start()
        t._on_rollout_end()
    assert t.collect == [2.0, 2.0] and t.update == [3.0]


class _Env:
    """Move 2 does 10 damage, only after the intro and only after walking (near)."""

    def reset(self, seed=None):
        self.walked = self.steps = 0
        return None, {}

    def step(self, action):
        self.steps += 1
        self.walked += action == 1 and self.steps > INTRO_STEPS
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


def test_compare_plays_the_newest_checkpoint_both_ways(tmp_path):
    from sega_rl.bench import compare_play
    from sega_rl.config import Settings
    from sega_rl.evaluate import EvalResult

    cfg = Settings(game="sf", run_id="r", runs_dir=tmp_path)
    ckpts = cfg.run_dir / "checkpoints"
    ckpts.mkdir(parents=True)
    for n in (1, 2):
        (ckpts / f"ckpt_{n:010d}.zip").touch()
    loaded, calls = [], []

    def load(path):
        loaded.append(path.name)
        return "model"

    def play(model, cfg, episodes, video_path=None, deterministic=True):
        calls.append(deterministic)
        return EvalResult(win_rate=0.5 if deterministic else 0.75, mean_damage_dealt=0, mean_damage_taken=0,
                          mean_return=0, rounds_done=40)

    out = compare_play(cfg, load=load, play=play)
    assert loaded == ["ckpt_0000000002.zip"] and calls == [True, False]
    assert out["sampled"]["win_rate"] == 0.75 and (cfg.run_dir / "compare.json").exists()
    assert not (cfg.run_dir / "DONE").exists()  # the run can still be resumed

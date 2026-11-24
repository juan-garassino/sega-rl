"""Win-rate evidence: enough rounds, and do-nothing / random baselines measured before training (no emulator here)."""
from __future__ import annotations

import json

import numpy as np
from gymnasium.spaces import Discrete, MultiBinary

from sega_rl.config import Settings
from sega_rl.evaluate import EvalResult, RandomPolicy, StandStill, run_baselines


def test_sf_eval_uses_enough_episodes_for_30_rounds():
    assert Settings(game="sf").n_eval_episodes == 15  # at least 2 rounds per match
    assert Settings(game="sonic").n_eval_episodes == 5
    assert Settings(game="sf", eval_episodes=2).n_eval_episodes == 2


def test_stand_still_presses_nothing():
    obs = np.zeros((1, 84, 84, 4))
    assert StandStill(Discrete(21)).predict(obs)[0].tolist() == [0]
    assert StandStill(MultiBinary(12)).predict(obs)[0].tolist() == [[0] * 12]


def test_random_policy_samples_valid_actions():
    actions = {int(RandomPolicy(Discrete(21)).predict(np.zeros((1, 1)))[0][0]) for _ in range(50)}
    assert len(actions) > 1 and max(actions) < 21


def test_baselines_are_written_and_named(tmp_path):
    seen = []

    def fake_eval(model, cfg, episodes, heartbeat=None):
        seen.append((type(model).__name__, episodes))
        return EvalResult(win_rate=0.1, mean_damage_dealt=0, mean_damage_taken=352, mean_return=-1, rounds_done=30)

    out = run_baselines(Settings(game="sf"), tmp_path, Discrete(21), eval_fn=fake_eval)
    assert seen == [("StandStill", 15), ("RandomPolicy", 15)]
    assert out["baseline/stand_still_win_rate"] == 0.1 and out["baseline/random_rounds_done"] == 30
    assert json.loads((tmp_path / "baselines.json").read_text()) == out


def test_baselines_run_once_per_run(tmp_path):
    (tmp_path / "baselines.json").write_text("{}")
    assert run_baselines(Settings(game="sf"), tmp_path, Discrete(21), eval_fn=None) == {}

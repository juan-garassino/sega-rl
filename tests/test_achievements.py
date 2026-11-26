"""Achievement log and resumed hyperparameters: no emulator, no training."""
from __future__ import annotations

import json

import numpy as np

from sega_rl.config import Settings
from sega_rl.monitoring.achievements import first


def test_an_achievement_is_written_once_for_all_processes(tmp_path):
    assert first(tmp_path, "finish_GreenHill1", "finished Green Hill 1", steps=5) is True
    assert first(tmp_path, "finish_GreenHill1", "again", steps=9) is False
    assert json.loads((tmp_path / "finish_GreenHill1.json").read_text())["steps"] == 5
    assert first(None, "x", "y") is False


def test_a_finished_level_or_won_match_is_logged_with_the_start_it_began_from(tmp_path, monkeypatch):
    from sega_rl.envs.common import RetroEnv
    from test_envs_common import _score_reward, _StubRetro

    stub = _StubRetro()
    env = RetroEnv("stub", None, _score_reward, combos=None, obs_mode="frame", frame_skip=1, inner=stub,
                   log_dir=tmp_path, states=["/fights/Ken.state"])
    stub.load_state = lambda s: None
    stub.initial_state = b""
    import sega_rl.envs.common as common

    monkeypatch.setattr(common, "load_start", lambda game, state: None)
    env.reset()
    env._celebrate({"matches_won": 2, "enemy_matches_won": 0})
    env._celebrate({"level_complete": True})
    assert sorted(p.stem for p in tmp_path.iterdir()) == ["finish_Ken", "win_Ken"]


def test_gamma_and_rollout_length_apply_to_a_loaded_model():
    from gymnasium.spaces import Box
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from sega_rl.train import apply_hyperparameters

    class _E:
        metadata = {}
        observation_space, action_space = Box(0, 255, (84, 84, 1), np.uint8), __import__("gymnasium").spaces.Discrete(3)
        def reset(self, *, seed=None, options=None): return np.zeros((84, 84, 1), np.uint8), {}
        def step(self, a): return np.zeros((84, 84, 1), np.uint8), 0.0, False, False, {}
        def render(self): return None
        def close(self): pass
        unwrapped = property(lambda s: s)
        spec = None
    import gymnasium as gym

    class E(gym.Env):
        observation_space, action_space = _E.observation_space, _E.action_space
        reset, step = _E.reset, _E.step

    env = DummyVecEnv([E, E])
    m = PPO("CnnPolicy", env, n_steps=8, batch_size=16, device="cpu")
    apply_hyperparameters(m, Settings(game="sf", gamma=0.997, n_steps=16, n_envs=2))
    assert m.gamma == 0.997 and m.rollout_buffer.gamma == 0.997 and m.n_steps == 16 and m.rollout_buffer.buffer_size == 16


def test_adam_counters_leave_the_gpu_after_a_load():
    from types import SimpleNamespace

    import torch

    from sega_rl.train import keep_counters_on_cpu

    class _OnGpu(torch.Tensor):
        moved = 0

        def cpu(self):
            _OnGpu.moved += 1
            return torch.tensor(7.0)

    state = {"a": {"step": torch.tensor(7.0).as_subclass(_OnGpu)}, "b": {}}  # "b" has not been updated yet
    keep_counters_on_cpu(SimpleNamespace(policy=SimpleNamespace(optimizer=SimpleNamespace(state=state))))
    assert _OnGpu.moved == 1 and state["a"]["step"].item() == 7.0 and state["b"] == {}

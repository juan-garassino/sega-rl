"""Growing a trained network's frame stack must not change how it plays."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from sega_rl.graft import graft_frames


class _Game(gym.Env):
    observation_space, action_space = Box(0, 255, (84, 84, 1), np.uint8), Discrete(5)

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 1.0, False, False, {}


def _model(frames):
    return PPO("CnnPolicy", VecFrameStack(DummyVecEnv([_Game]), frames, channels_order="last"), n_steps=8, batch_size=8, device="cpu")


def test_the_grown_network_plays_the_same_whatever_the_older_frames_show():
    old, new = _model(4), _model(8)
    old.num_timesteps = 123
    graft_frames(old, new)
    frames = torch.randint(0, 256, (3, 8, 84, 84), dtype=torch.uint8)  # oldest first, as VecFrameStack orders them
    with torch.no_grad():
        a = old.policy.get_distribution(frames[:, 4:]).distribution.logits
        b = new.policy.get_distribution(frames).distribution.logits
        va, vb = old.policy.predict_values(frames[:, 4:]), new.policy.predict_values(frames)
    assert torch.allclose(a, b, atol=1e-5) and torch.allclose(va, vb, atol=1e-5) and new.num_timesteps == 123


def test_the_newest_frame_is_the_last_channel():
    env = VecFrameStack(DummyVecEnv([_Game]), 8, channels_order="last")
    env.reset()
    obs, *_ = env.step([0])
    newest = obs[0, :, :, -1].copy()
    obs, *_ = env.step([0])
    assert (obs[0, :, :, -2] == newest).all()


def test_frames_cannot_be_taken_away():
    with pytest.raises(ValueError, match="cannot graft"):
        graft_frames(_model(8), _model(4))

from __future__ import annotations

import numpy as np

from sega_rl.config import Settings
from sega_rl.envs.factory import make_vec_env


def test_vec_env_airstriker():
    env = make_vec_env(Settings(game="airstriker"), n_envs=2)
    try:
        obs = env.reset()
        assert obs.shape == (2, 84, 84, 4)
        for _ in range(20):
            obs, rewards, dones, infos = env.step(np.array([env.action_space.sample() for _ in range(2)]))
        assert rewards.shape == (2,)
        assert "buttons" in infos[0]
    finally:
        env.close()

from __future__ import annotations

import numpy as np
import pytest

from sega_rl.envs.sonic import LEVEL_MAX_X, SONIC_ACTIONS, SonicReward


def _info(x, lives=3, zone=0, act=0):
    return {"x": x, "lives": lives, "zone": zone, "act": act}


def test_progress_reward_only_new_ground():
    fn = SonicReward()
    rewards = []
    prev = {}
    for x in (100, 150, 120, 160):
        info = _info(x)
        r, _ = fn(prev, info)
        rewards.append(r)
        prev = info
    assert rewards == pytest.approx([0.0, 0.5, 0.0, 0.1])
    assert prev["max_x"] == 160


def test_level_complete_bonus_and_terminates():
    fn = SonicReward()
    fn({}, _info(100))
    info = _info(LEVEL_MAX_X[(0, 0)])
    r, term = fn(_info(9000), info)
    assert term is True
    assert info["level_complete"] is True
    assert r >= 10.0


def test_life_lost_terminates():
    fn = SonicReward()
    fn({}, _info(100))
    _, term = fn(_info(100), _info(100, lives=2))
    assert term is True


def test_progress_fraction():
    fn = SonicReward()
    fn({}, _info(80))
    info = _info(80 + (LEVEL_MAX_X[(0, 0)] - 80) // 2)
    fn(_info(80), info)
    assert info["progress"] == pytest.approx(0.5, abs=0.01)


def test_action_mapping():
    from sega_rl.envs.common import RetroEnv

    class _Stub:
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        action_space = None

    env = RetroEnv.__new__(RetroEnv)
    env.buttons = _Stub.buttons
    vec = env._buttons_vec(SONIC_ACTIONS[2])
    assert vec[_Stub.buttons.index("RIGHT")] == 1 and int(np.sum(vec)) == 1
    assert len(SONIC_ACTIONS) == 7


def test_truncation_4500():
    from sega_rl.envs.sonic import MAX_STEPS

    assert MAX_STEPS == 4500


@pytest.mark.rom
def test_sonic_rom_smoke():
    import stable_retro

    from sega_rl.config import Settings
    from sega_rl.envs import sonic

    try:
        stable_retro.data.get_romfile_path(sonic.GAME_ID)
    except FileNotFoundError:
        pytest.skip("Sonic ROM not imported (make import-roms)")
    env = sonic.make(Settings(game="sonic"))
    try:
        env.reset(seed=0)
        assert env.action_space.n == 7
        for _ in range(50):
            *_, info = env.step(2)  # hold RIGHT
        assert info["max_x"] > 80
    finally:
        env.close()

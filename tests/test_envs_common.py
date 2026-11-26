from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, MultiBinary

from sega_rl.config import Settings
from sega_rl.envs import airstriker
from sega_rl.envs.common import RetroEnv, preprocess, signed_delta

BUTTONS = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]


class _StubRetro(gym.Env):
    """Counts raw steps; frame value = step index so max-pooling is observable."""

    def __init__(self):
        self.observation_space = Box(0, 255, (224, 320, 3), np.uint8)
        self.action_space = MultiBinary(12)
        self.buttons = BUTTONS
        self.calls = []
        self.n = 0

    def reset(self, *, seed=None, options=None):
        self.n = 0
        return np.zeros((224, 320, 3), np.uint8), {}

    def step(self, action):
        self.calls.append(np.array(action))
        self.n += 1
        frame = np.full((224, 320, 3), (self.n * 10) % 256, np.uint8)
        return frame, 0.0, False, False, {"score": self.n}


def _score_reward(prev, info):
    return float(info.get("score", 0) - prev.get("score", 0)), False


def test_preprocess_shape_dtype():
    frame = np.random.randint(0, 256, (224, 320, 3), dtype=np.uint8)
    out = preprocess(frame)
    assert out.shape == (84, 84, 1)
    assert out.dtype == np.uint8


def test_signed_delta_no_wrap():
    zero = np.zeros((84, 84, 1), np.uint8)
    two_hundred = np.full((84, 84, 1), 200, np.uint8)
    assert int(signed_delta(zero, two_hundred)[0, 0, 0]) == 27
    assert int(signed_delta(two_hundred, zero)[0, 0, 0]) == 227


def test_frame_skip_counts():
    stub = _StubRetro()
    env = RetroEnv("stub", None, _score_reward, combos=None, obs_mode="frame", frame_skip=4, inner=stub)
    env.reset()
    _, reward, _, _, info = env.step(np.zeros(12, np.int8))
    assert len(stub.calls) == 4
    assert reward == 4.0
    assert info["buttons"].shape == (12,)


def test_max_pool_last_two_frames():
    stub = _StubRetro()
    env = RetroEnv("stub", None, _score_reward, combos=None, obs_mode="frame", frame_skip=4, inner=stub)
    env.reset()
    obs, *_ = env.step(np.zeros(12, np.int8))
    assert int(obs.max()) == 40  # max of raw frames 3 (30) and 4 (40)


def test_combo_mapping():
    stub = _StubRetro()
    env = RetroEnv("stub", None, _score_reward, combos=[[], ["RIGHT"], ["RIGHT", "DOWN"]],
                   obs_mode="frame", frame_skip=1, inner=stub)
    assert env.action_space.n == 3
    env.reset()
    env.step(2)
    sent = stub.calls[-1]
    assert sent[BUTTONS.index("RIGHT")] == 1 and sent[BUTTONS.index("DOWN")] == 1 and sent.sum() == 2


def test_truncation_max_steps():
    stub = _StubRetro()
    env = RetroEnv("stub", None, _score_reward, combos=None, obs_mode="frame", frame_skip=1,
                   inner=stub, max_steps=3)
    env.reset()
    truncs = [env.step(np.zeros(12, np.int8))[3] for _ in range(3)]
    assert truncs == [False, False, True]


def test_airstriker_step_contract():
    env = airstriker.make(Settings(game="airstriker"))
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (84, 84, 1)
        for _ in range(10):
            obs, reward, term, trunc, info = env.step(env.action_space.sample())
            assert isinstance(reward, float)
            assert info["buttons"].shape == (12,)
    finally:
        env.close()


def test_events_on_middle_frame_survive_frame_skip():
    def events(prev, info):
        n = info["score"]
        info["damage_dealt"] = 10
        info["round_won"] = True if n == 2 else None
        info["level_complete"] = n == 3
        return 0.0, False

    stub = _StubRetro()
    env = RetroEnv("stub", None, events, combos=None, obs_mode="frame", frame_skip=4, inner=stub)
    env.reset()
    _, _, _, _, info = env.step(np.zeros(12, np.int8))
    assert info["round_won"] is True
    assert info["damage_dealt"] == 40
    assert info["level_complete"] is True


def test_noop_reset_varies_start():
    counts = []
    for seed in range(6):
        stub = _StubRetro()
        env = RetroEnv("stub", None, _score_reward, combos=None, obs_mode="frame", frame_skip=1,
                       inner=stub, noop_max=30)
        env.reset(seed=seed)
        counts.append(len(stub.calls))
    assert len(set(counts)) > 1 and max(counts) <= 30


def test_a_move_can_be_a_timed_sequence():
    stub = _StubRetro()
    hadouken = [(["DOWN"], 3), (["DOWN", "RIGHT"], 3), (["RIGHT", "Z"], 2)]
    env = RetroEnv("stub", None, _score_reward, combos=[[], hadouken], obs_mode="frame", frame_skip=4, inner=stub)
    env.reset()
    _, reward, _, _, info = env.step(1)
    assert len(stub.calls) == 8 and reward == 8.0 and info["move"] == 1
    assert stub.calls[0][BUTTONS.index("DOWN")] == 1 and stub.calls[0].sum() == 1
    assert stub.calls[4][BUTTONS.index("RIGHT")] == 1 and stub.calls[7][BUTTONS.index("Z")] == 1
    env.step(0)
    assert len(stub.calls) == 12


def test_each_game_starts_on_one_of_the_given_states():
    stub = _StubRetro()
    stub.loaded = []
    stub.load_state = stub.loaded.append
    env = RetroEnv("stub", None, _score_reward, combos=None, obs_mode="frame", frame_skip=1, inner=stub,
                   states=["A", "B", "C"])
    for seed in range(20):
        env.reset(seed=seed)
    assert set(stub.loaded) == {"A", "B", "C"}


def test_a_start_can_be_our_own_state_file(tmp_path):
    import gzip

    from sega_rl.envs.common import load_start

    f = tmp_path / "Ken.state"
    f.write_bytes(gzip.compress(b"ken"))
    stub = _StubRetro()
    load_start(stub, str(f))
    assert stub.initial_state == b"ken"


class _BusyGame:
    """A stand-in emulator whose RAM byte 0 says busy for `busy_for` frames after a press."""

    buttons = ["B"]

    def __init__(self, busy_for):
        from gymnasium.spaces import MultiBinary

        self.action_space, self.busy_for, self.left, self.pressed = MultiBinary(1), busy_for, 0, []

    def reset(self, seed=None):
        self.n = 0
        return np.zeros((224, 320, 3), np.uint8), {}

    def get_ram(self):
        return np.array([self.left > 0], np.uint8)

    def step(self, buttons):
        self.n += 1
        self.pressed.append(int(buttons[0]))
        self.left = self.busy_for if buttons[0] else max(0, self.left - 1)
        return np.full((224, 320, 3), self.n % 250, np.uint8), 0.0, False, False, {}


def _busy_env(busy_for, history=8):
    game = _BusyGame(busy_for)
    return game, RetroEnv("g", None, lambda p, i: (1.0, False), [[], ["B"]], "frame", 4, inner=game,
                          busy=lambda ram: bool(ram[0]), history=history)


def test_frames_where_the_player_cannot_act_pass_by_themselves():
    game, env = _busy_env(busy_for=20)
    env.reset()
    _, reward, _, _, _ = env.step(1)
    assert game.pressed == [1] * 4 + [0] * 20 and reward == 24.0  # every frame still counts for the reward
    env.step(0)
    assert len(game.pressed) == 28  # free to act: one ordinary move, no waiting
    game, env = _busy_env(busy_for=10_000)
    env.reset()
    env.step(1)
    assert len(game.pressed) == 4 + 120  # never stuck for good, and never holding the other games long


def test_the_history_is_game_time_not_decisions():
    game, env = _busy_env(busy_for=20)
    obs, _ = env.reset()
    assert obs.shape == (84, 84, 8) and not obs.any()
    obs, *_ = env.step(1)  # 24 frames: six of the eight history frames are new, 4 game frames apart
    assert [int(obs[0, 0, k]) for k in range(8)] == [0, 0, 4, 8, 12, 16, 20, 24] and len(env.film) == 6

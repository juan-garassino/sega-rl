from __future__ import annotations

import gzip
import logging
from typing import Callable, Literal

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete

logger = logging.getLogger(__name__)

# (prev_info, info) -> (reward, terminated_extra). May add keys to `info` (e.g. round_won).
RewardFn = Callable[[dict, dict], tuple[float, bool]]

# Per-frame events a reward fn may emit; merged across skipped frames so none is lost.
SUM_KEYS = ("damage_dealt", "damage_taken")

# A move is buttons held for one agent step, or a timed sequence of (buttons, raw frames).
Move = list[str] | list[tuple[list[str], int]]


def preprocess(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return small[:, :, None]


def signed_delta(cur: np.ndarray, prev: np.ndarray) -> np.ndarray:
    d = cur.astype(np.int16) - prev.astype(np.int16)
    return ((d + 255) // 2).astype(np.uint8)


def load_start(game, state: str) -> None:
    """A state shipped with the integration, or a .state file of our own (from `sega-rl ladder`)."""
    if state.endswith(".state"):
        with gzip.open(state, "rb") as f:
            game.initial_state = f.read()
    else:
        game.load_state(state)


def _make_retro(game_id: str, state: str | None) -> gym.Env:
    import stable_retro

    if state is not None and state.endswith(".state"):
        game = _make_retro(game_id, None)
        load_start(game, state)
        return game
    kwargs = {"use_restricted_actions": stable_retro.Actions.FILTERED, "render_mode": None}
    if state is not None:
        kwargs["state"] = state
    return stable_retro.make(game_id, **kwargs)


class RetroEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        game_id: str,
        state: str | None,
        reward_fn: RewardFn,
        combos: list[Move] | None,
        obs_mode: Literal["frame", "delta"],
        frame_skip: int,
        inner: gym.Env | None = None,
        max_steps: int | None = None,
        noop_max: int = 0,
        states: list[str] | None = None,
    ):
        super().__init__()
        cv2.setNumThreads(1)  # every env process would otherwise open a thread per host core
        self.game = inner if inner is not None else _make_retro(game_id, state)
        self.reward_fn = reward_fn
        self.obs_mode = obs_mode
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.noop_max = noop_max
        self.states = states  # each game starts on one of these, drawn at random
        self.buttons: list[str] = list(self.game.buttons)
        self.combos = combos
        if combos is None:
            self.action_space = self.game.action_space
        else:
            self.action_space = Discrete(len(combos))
            self._moves = [self._move(c) for c in combos]
        self.observation_space = Box(0, 255, (84, 84, 1), np.uint8)
        self.last_raw: np.ndarray | None = None
        self._prev_obs: np.ndarray | None = None
        self._prev_info: dict = {}
        self._steps = 0

    def _buttons_vec(self, names: list[str]) -> np.ndarray:
        vec = np.zeros(len(self.buttons), np.int8)
        for n in names:
            vec[self.buttons.index(n)] = 1
        return vec

    def _move(self, move: Move) -> list[tuple[np.ndarray, int]]:
        if move and isinstance(move[0], tuple):
            return [(self._buttons_vec(names), frames) for names, frames in move]
        return [(self._buttons_vec(move), self.frame_skip)]

    def _frames(self, action) -> list[np.ndarray]:
        """The buttons to hold on each raw frame of this action."""
        if self.combos is None:
            return [np.asarray(action, dtype=np.int8)] * self.frame_skip
        return [vec for vec, n in self._moves[int(action)] for _ in range(n)]

    def _observe(self, proc: np.ndarray) -> np.ndarray:
        if self.obs_mode == "delta":
            out = signed_delta(proc, self._prev_obs if self._prev_obs is not None else proc)
        else:
            out = proc
        self._prev_obs = proc
        return out

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.states:
            load_start(self.game, self.states[int(self.np_random.integers(len(self.states)))])
        frame, info = self.game.reset(seed=seed)
        if hasattr(self.reward_fn, "reset"):
            self.reward_fn.reset()
        # Random no-op start: otherwise every episode (and every parallel env) replays the same savestate.
        noop = np.zeros(len(self.buttons), np.int8)
        for _ in range(int(self.np_random.integers(0, self.noop_max + 1)) if self.noop_max else 0):
            frame, _, term, trunc, info = self.game.step(noop)
            if term or trunc:
                frame, info = self.game.reset()
        self._prev_info = dict(info)
        self._prev_obs = None
        self._steps = 0
        self.last_raw = frame
        return self._observe(preprocess(frame)), dict(info)

    def step(self, action):
        move = None if self.combos is None else int(action)
        total, terminated, truncated = 0.0, False, False
        frames: list[np.ndarray] = []
        info: dict = {}
        sums: dict[str, float] = {}
        round_won, level_complete = None, False
        for buttons in self._frames(action):
            frame, _, term, trunc, info = self.game.step(buttons)
            info = dict(info)
            if move is not None:
                info["move"], info["move_id"] = move, self._steps
            r, extra_term = self.reward_fn(self._prev_info, info)
            self._prev_info = info
            for k in SUM_KEYS:
                if k in info:
                    sums[k] = sums.get(k, 0) + info[k]
            if info.get("round_won") is not None:
                round_won = info["round_won"]
            level_complete = level_complete or bool(info.get("level_complete"))
            total += r
            frames.append(frame)
            terminated = terminated or term or extra_term
            truncated = truncated or trunc
            if terminated or truncated:
                break
        pooled = np.maximum(frames[-2], frames[-1]) if len(frames) >= 2 else frames[-1]
        self.last_raw = frames[-1]
        self._steps += 1
        if self.max_steps is not None and self._steps >= self.max_steps:
            truncated = True
        info.update(sums)
        if "round_won" in info:
            info["round_won"] = round_won
        if "level_complete" in info:
            info["level_complete"] = level_complete
        info["buttons"] = buttons
        return self._observe(preprocess(pooled)), float(total), terminated, truncated, info

    def render(self):
        return self.last_raw

    def close(self):
        self.game.close()

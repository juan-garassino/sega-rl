from __future__ import annotations

import logging
from typing import Callable, Literal

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete

logger = logging.getLogger(__name__)

# (prev_info, info) -> (reward, terminated_extra). May add keys to `info` (e.g. round_won).
RewardFn = Callable[[dict, dict], tuple[float, bool]]


def preprocess(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return small[:, :, None]


def signed_delta(cur: np.ndarray, prev: np.ndarray) -> np.ndarray:
    d = cur.astype(np.int16) - prev.astype(np.int16)
    return ((d + 255) // 2).astype(np.uint8)


def _make_retro(game_id: str, state: str | None) -> gym.Env:
    import stable_retro

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
        combos: list[list[str]] | None,
        obs_mode: Literal["frame", "delta"],
        frame_skip: int,
        inner: gym.Env | None = None,
        max_steps: int | None = None,
    ):
        super().__init__()
        self.game = inner if inner is not None else _make_retro(game_id, state)
        self.reward_fn = reward_fn
        self.obs_mode = obs_mode
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.buttons: list[str] = list(self.game.buttons)
        self.combos = combos
        if combos is None:
            self.action_space = self.game.action_space
        else:
            self.action_space = Discrete(len(combos))
            self._combo_vecs = [self._buttons_vec(c) for c in combos]
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

    def _to_buttons(self, action) -> np.ndarray:
        if self.combos is not None:
            return self._combo_vecs[int(action)]
        return np.asarray(action, dtype=np.int8)

    def _observe(self, proc: np.ndarray) -> np.ndarray:
        if self.obs_mode == "delta":
            out = signed_delta(proc, self._prev_obs if self._prev_obs is not None else proc)
        else:
            out = proc
        self._prev_obs = proc
        return out

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        frame, info = self.game.reset(seed=seed)
        if hasattr(self.reward_fn, "reset"):
            self.reward_fn.reset()
        self._prev_info = dict(info)
        self._prev_obs = None
        self._steps = 0
        self.last_raw = frame
        return self._observe(preprocess(frame)), dict(info)

    def step(self, action):
        buttons = self._to_buttons(action)
        total, terminated, truncated = 0.0, False, False
        frames: list[np.ndarray] = []
        info: dict = {}
        for _ in range(self.frame_skip):
            frame, _, term, trunc, info = self.game.step(buttons)
            info = dict(info)
            r, extra_term = self.reward_fn(self._prev_info, info)
            self._prev_info = info
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
        info["buttons"] = buttons
        return self._observe(preprocess(pooled)), float(total), terminated, truncated, info

    def render(self):
        return self.last_raw

    def close(self):
        self.game.close()

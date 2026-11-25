"""Give a trained network more past frames without losing what it knows.

Stacked frames are the channels of the first convolution, oldest first. The old weights move to the newest channels and
the added, older ones start at zero, so the grown network plays exactly like the old one until training uses them.
"""
from __future__ import annotations

import torch
from stable_baselines3 import PPO


def graft_frames(old: PPO, new: PPO) -> None:
    out = {}
    o = old.policy.state_dict()
    for name, target in new.policy.state_dict().items():
        source = o[name].to(target.device, target.dtype)
        if source.shape != target.shape:
            if not name.endswith("cnn.0.weight") or source.shape[1] > target.shape[1] or source.shape[2:] != target.shape[2:]:
                raise ValueError(f"cannot graft {name}: {tuple(source.shape)} -> {tuple(target.shape)}")
            grown = torch.zeros_like(target)
            grown[:, -source.shape[1]:] = source
            source = grown
        out[name] = source
    new.policy.load_state_dict(out)
    new.num_timesteps = old.num_timesteps

from __future__ import annotations

import pytest

from sega_rl.cli import build_parser, settings_from_args


def test_train_overrides():
    args = build_parser().parse_args(["train", "--game", "sf", "--steps", "500", "--n-envs", "4", "--run-id", "x"])
    cfg = settings_from_args(args)
    assert (cfg.game, cfg.steps, cfg.n_envs, cfg.run_id) == ("sf", 500, 4, "x")


def test_requires_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])

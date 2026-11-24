from __future__ import annotations

import pytest

from sega_rl.config import Settings, default_max_eur, default_total_steps


def test_defaults():
    s = Settings()
    assert s.game == "airstriker"
    assert s.n_envs == 16
    assert s.obs_mode == "frame"


def test_env_prefix(monkeypatch):
    monkeypatch.setenv("SEGARL_GAME", "sf")
    monkeypatch.setenv("SEGARL_N_ENVS", "4")
    s = Settings()
    assert s.game == "sf"
    assert s.n_envs == 4


def test_default_total_steps():
    assert default_total_steps("airstriker") == 1_000_000
    assert default_total_steps("sf") == 10_000_000
    assert default_total_steps("sonic") == 10_000_000


def test_default_max_eur():
    assert default_max_eur("sf") == 8.0
    assert default_max_eur("sonic") == 7.0
    assert default_max_eur("airstriker") == 1.0


def test_resolved_budget(monkeypatch):
    monkeypatch.setenv("SEGARL_GAME", "sonic")
    s = Settings()
    assert s.steps == 10_000_000
    assert s.budget_eur == 7.0
    monkeypatch.setenv("SEGARL_MAX_EUR", "3")
    assert Settings().budget_eur == 3.0


def test_import_stable_retro():
    import stable_retro

    assert "Airstriker-Genesis-v0" in stable_retro.data.list_games()


def test_budget_from_cost(monkeypatch):
    monkeypatch.setenv("SEGARL_GAME", "sf")  # 8 EUR cap
    monkeypatch.setenv("SEGARL_COST_PER_HR_USD", "0.5")
    assert Settings().hours_budget == 6.0  # 8 EUR buys 16.8 h > 6 h cap
    monkeypatch.setenv("SEGARL_COST_PER_HR_USD", "3.0")
    assert Settings().hours_budget == pytest.approx(8 * 1.05 / 3.0)


def test_budget_without_cost_uses_max_hours():
    assert Settings(max_hours=2.0).hours_budget == 2.0

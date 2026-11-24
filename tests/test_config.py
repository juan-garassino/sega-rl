from __future__ import annotations

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

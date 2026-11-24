from __future__ import annotations

import pytest

from sega_rl.envs.street_fighter import SFReward

BASE = {"health": 176, "enemy_health": 176, "matches_won": 0, "enemy_matches_won": 0, "score": 0}


def _info(**kw):
    return {**BASE, **kw}


def test_damage_dealt_positive():
    r, term = SFReward("shaped")(_info(), _info(enemy_health=150))
    assert r == pytest.approx(26 / 176)
    assert term is False


def test_damage_taken_negative():
    r, _ = SFReward("shaped")(_info(), _info(health=126))
    assert r == pytest.approx(-50 / 176)


def test_round_win_bonus():
    info = _info(matches_won=1)
    r, _ = SFReward("shaped")(_info(), info)
    assert r == pytest.approx(1.0)
    assert info["round_won"] is True


def test_round_loss_penalty():
    info = _info(enemy_matches_won=1)
    r, _ = SFReward("shaped")(_info(), info)
    assert r == pytest.approx(-1.0)
    assert info["round_won"] is False


def test_score_mode_matches_upstream():
    r, _ = SFReward("score")(_info(score=100), _info(score=1100, enemy_health=100))
    assert r == 1000.0


def test_health_reset_between_rounds_not_rewarded():
    r, _ = SFReward("shaped")(_info(health=20, enemy_health=-1), _info(health=176, enemy_health=176))
    assert r == 0.0


def test_first_step_without_prev_is_zero():
    r, _ = SFReward("shaped")({}, _info(enemy_health=100))
    assert r == 0.0


def test_damage_recorded_in_info():
    info = _info(enemy_health=150, health=170)
    SFReward("shaped")(_info(), info)
    assert info["damage_dealt"] == 26 and info["damage_taken"] == 6


@pytest.mark.rom
def test_sf_rom_smoke():
    import stable_retro

    from sega_rl.config import Settings
    from sega_rl.envs import street_fighter

    try:
        stable_retro.data.get_romfile_path(street_fighter.GAME_ID)
    except FileNotFoundError:
        pytest.skip("SF ROM not imported (make import-roms)")
    env = street_fighter.make(Settings(game="sf"))
    try:
        env.reset(seed=0)
        for _ in range(50):
            *_, info = env.step(env.action_space.sample())
        assert {"health", "enemy_health", "matches_won", "enemy_matches_won", "score", "continuetimer"} <= set(info)
    finally:
        env.close()


def test_sf_discrete_action_set():
    from sega_rl.envs.street_fighter import SF_ACTIONS

    assert len(SF_ACTIONS) == 21
    assert [] in SF_ACTIONS and ["UP"] in SF_ACTIONS and ["DOWN", "X"] in SF_ACTIONS
    assert not any(b in ("START", "MODE") for combo in SF_ACTIONS for b in combo)
    # UP only in the 3 jump actions (random play no longer jumps half the time)
    assert sum("UP" in combo for combo in SF_ACTIONS) == 3


def test_damage_weights():
    fn = SFReward("shaped", dealt_weight=1.0, taken_weight=0.5)
    r_dealt, _ = fn(_info(), _info(enemy_health=150))
    r_taken, _ = fn(_info(), _info(health=126))
    assert r_dealt == pytest.approx(26 / 176)
    assert r_taken == pytest.approx(-0.5 * 50 / 176)


def test_sf_defaults_discrete_and_lower_lr():
    from sega_rl.config import Settings, default_learning_rate

    assert Settings(game="sf").sf_actions == "discrete"
    assert default_learning_rate("sf") == 1e-4
    assert default_learning_rate("sonic") == 2.5e-4

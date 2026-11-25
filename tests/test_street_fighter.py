from __future__ import annotations

import pytest

from sega_rl.envs.street_fighter import FIGHTER_ATTACKS, FIGHTER_MOVES, SFReward

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


def test_winning_with_health_left_pays_more():
    fn = SFReward("shaped", win_health_bonus=1.0)
    fn({}, _info())
    fn(_info(), _info(health=88, enemy_health=-1))
    r, _ = fn(_info(health=88, enemy_health=-1), _info(health=0, enemy_health=0, matches_won=1))
    assert r == pytest.approx(1.0 + 0.5)  # the health when the KO landed, not the cleared bar


def _hit(fn, use, move, enemy_health, prev_health):
    prev, info = _info(enemy_health=prev_health), _info(enemy_health=enemy_health, move=move, move_id=use)
    r, _ = fn(prev, info)
    return r * 176, info["damage_dealt"]


def test_repeating_the_same_hit_pays_less_but_damage_is_reported_in_full():
    fn = SFReward("shaped", attacks=frozenset({3, 4}))
    fn({}, _info())
    paid = [_hit(fn, use, 3, 170 - 10 * use, 180 - 10 * use) for use in range(1, 4)]
    assert [round(p) for p, _ in paid] == [10, 5, 3] or [round(p, 1) for p, _ in paid] == [10.0, 5.0, 2.5]
    assert [d for _, d in paid] == [10, 10, 10]
    assert _hit(fn, 4, 4, 130, 140)[0] == pytest.approx(10)  # a different move pays in full


def test_one_hit_draining_over_several_frames_is_one_use():
    fn = SFReward("shaped", attacks=frozenset({3}))
    fn({}, _info())
    assert sum(_hit(fn, 1, 3, 176 - n - 1, 176 - n)[0] for n in range(8)) == pytest.approx(8)


def test_specials_start_from_neutral_so_walking_forward_does_not_change_the_motion():
    for name in ("hadouken_right", "hadouken_left", "tatsumaki_right", "tatsumaki_left"):
        assert FIGHTER_MOVES[name][0][0] == [] and FIGHTER_MOVES[name][0][1] >= 10


def test_fighter_moves_include_every_special_both_ways():
    assert len(set(FIGHTER_MOVES)) == len(FIGHTER_MOVES)
    for name in ("hadouken", "shoryuken", "tatsumaki", "forward_hp", "jump_kick"):
        assert {f"{name}_left", f"{name}_right"} <= set(FIGHTER_MOVES)
    names = list(FIGHTER_MOVES)
    assert all(names[i] not in ("idle", "left", "right", "up") for i in FIGHTER_ATTACKS)


def test_a_single_match_ends_when_either_side_wins_twice():
    fn = SFReward("shaped", single_match=True)
    fn({}, _info())
    assert fn(_info(), _info(matches_won=1))[1] is False
    assert fn(_info(matches_won=1), _info(matches_won=2))[1] is True
    assert fn(_info(enemy_matches_won=1), _info(enemy_matches_won=2))[1] is True
    assert SFReward("shaped")(_info(), _info(matches_won=2))[1] is False


def test_opponents_come_from_the_split_folders(tmp_path):
    from sega_rl.config import Settings
    from sega_rl.envs.street_fighter import opponents

    for split, names in (("train", ("Ken", "Blanka")), ("heldout", ("Sagat",))):
        (tmp_path / split).mkdir()
        for n in names:
            (tmp_path / split / f"{n}.state").touch()
    cfg = Settings(game="sf", sf_states_dir=tmp_path)
    assert [p.split("/")[-1] for p in opponents(cfg, "train")] == ["Blanka.state", "Ken.state"]
    assert len(opponents(cfg, "heldout")) == 1 and opponents(Settings(game="sf"), "train") is None


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


def test_sf_defaults_fighter_and_lower_lr():
    from sega_rl.config import Settings, default_learning_rate

    assert Settings(game="sf").sf_actions == "fighter"
    assert default_learning_rate("sf") == 1e-4
    assert default_learning_rate("sonic") == 2.5e-4


def _total(frames):
    """Run SFReward over (health, enemy_health, won, lost) frames; return summed damage and reward."""
    fn, prev, dealt, taken, reward = SFReward("shaped"), {}, 0, 0, 0.0
    for hp, ehp, won, lost in frames:
        info = _info(health=hp, enemy_health=ehp, matches_won=won, enemy_matches_won=lost)
        r, _ = fn(prev, info)
        dealt, taken, reward = dealt + info.get("damage_dealt", 0), taken + info.get("damage_taken", 0), reward + r
        prev = info
    return dealt, taken, reward


# Frames recorded from the real ROM: KO is -1, and the game clears both bars to 0 between rounds.
def test_losing_untouched_enemy_deals_nothing():
    dealt, taken, reward = _total([(176, 176, 0, 0), (33, 176, 0, 0), (-1, 176, 0, 0), (-1, 176, 0, 1),
                                   (0, 0, 0, 1), (176, 176, 0, 1)])
    assert (dealt, taken) == (0, 176)
    assert reward == pytest.approx(-2.0)


def test_winning_untouched_takes_nothing():
    dealt, taken, reward = _total([(176, 176, 0, 0), (176, -1, 0, 0), (176, -1, 1, 0), (0, 0, 1, 0), (176, 176, 1, 0)])
    assert (dealt, taken) == (176, 0)
    assert reward == pytest.approx(2.0)


def test_damage_counts_again_in_the_next_round():
    dealt, _, _ = _total([(176, 176, 0, 0), (-1, 176, 0, 0), (-1, 176, 0, 1), (0, 0, 0, 1), (176, 176, 0, 1),
                          (176, 150, 0, 1)])
    assert dealt == 26


def test_cleared_bars_without_a_round_result_are_not_damage():
    assert _total([(100, 90, 0, 0), (0, 0, 0, 0), (176, 176, 0, 0)])[:2] == (0, 0)


@pytest.mark.rom
def test_standing_still_deals_no_damage_on_the_real_game():
    import stable_retro

    from sega_rl.config import Settings
    from sega_rl.envs import street_fighter

    try:
        stable_retro.data.get_romfile_path(street_fighter.GAME_ID)
    except FileNotFoundError:
        pytest.skip("SF ROM not imported (make import-roms)")
    env = street_fighter.make(Settings(game="sf", noop_max=0))
    dealt = taken = 0
    try:
        env.reset(seed=0)
        for _ in range(1400):
            *_, term, trunc, info = env.step(0)
            dealt, taken = dealt + info.get("damage_dealt", 0), taken + info.get("damage_taken", 0)
            if term or trunc:
                break
    finally:
        env.close()
    assert dealt == 0
    assert taken == 2 * 176

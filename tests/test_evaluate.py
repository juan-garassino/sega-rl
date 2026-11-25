from __future__ import annotations

import math
import sys

import numpy as np

from sega_rl.config import Settings
from sega_rl.evaluate import EvalCallback, EvalResult, run_eval
from sega_rl.monitoring import mlflow_tracking


class _Logger:
    def __init__(self):
        self.records = {}

    def record(self, key, value, exclude=None):
        self.records[key] = value


class _RandomModel:
    def __init__(self, action_space):
        self.logger = _Logger()
        self.action_space = action_space
        self.saved = []
        self.num_timesteps = 0

    def predict(self, obs, deterministic=True):
        return np.array([self.action_space.sample() for _ in range(len(obs))]), None

    def save(self, path):
        self.saved.append(str(path))
        open(path, "w").close()

    def get_env(self):
        return None


def test_eval_airstriker_returns_result(tmp_path):
    from gymnasium.spaces import MultiBinary

    video = tmp_path / "v.mp4"
    result = run_eval(_RandomModel(MultiBinary(12)), Settings(game="airstriker"), episodes=1,
                      video_path=video, max_steps=200)
    assert isinstance(result, EvalResult)
    assert math.isnan(result.win_rate)
    assert isinstance(result.mean_return, float)
    assert video.exists() and video.stat().st_size > 0


def test_mlflow_disabled_noop(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)  # any import attempt would raise
    assert mlflow_tracking.enabled() is False
    mlflow_tracking.start(Settings())
    mlflow_tracking.log_metrics({"a": 1.0}, step=1)
    mlflow_tracking.end("done")


def test_best_model_selection(tmp_path):
    from gymnasium.spaces import MultiBinary

    results = iter([
        EvalResult(win_rate=0.2, mean_damage_dealt=50, mean_damage_taken=60, mean_return=0.0),
        EvalResult(win_rate=0.1, mean_damage_dealt=90, mean_damage_taken=10, mean_return=0.0),
        EvalResult(win_rate=0.4, mean_damage_dealt=40, mean_damage_taken=80, mean_return=0.0),
    ])
    cfg = Settings(game="sf", eval_freq=100, eval_sampled=False)
    cb = EvalCallback(cfg, tmp_path, eval_fn=lambda *a, **k: next(results))
    model = _RandomModel(MultiBinary(12))
    cb.init_callback(model)
    for steps in (100, 200, 300):
        model.num_timesteps = steps
        cb.on_step()
    assert len(model.saved) == 2
    assert (tmp_path / "best.json").exists()


def test_eval_heartbeat_called_per_step():
    from gymnasium.spaces import MultiBinary

    beats = []
    run_eval(_RandomModel(MultiBinary(12)), Settings(game="airstriker"), episodes=1, max_steps=20,
             heartbeat=lambda: beats.append(1))
    assert len(beats) >= 20


def test_best_restored_on_resume(tmp_path):
    import json as _json
    from gymnasium.spaces import MultiBinary

    (tmp_path / "best.json").write_text(_json.dumps({"step": 1, "win_rate": 0.5, "mean_damage_dealt": 0,
                                                     "mean_damage_taken": 0, "mean_return": 0.0}))
    results = iter([EvalResult(win_rate=0.3, mean_damage_dealt=0, mean_damage_taken=0, mean_return=0.0)])
    cb = EvalCallback(Settings(game="sf", eval_freq=100, eval_sampled=False), tmp_path, eval_fn=lambda *a, **k: next(results))
    model = _RandomModel(MultiBinary(12))
    cb.init_callback(model)
    model.num_timesteps = 100
    cb.on_step()
    assert model.saved == []


def test_best_model_uploaded_on_improvement(tmp_path, monkeypatch):
    from gymnasium.spaces import MultiBinary

    uploads = []
    monkeypatch.setattr(mlflow_tracking, "log_artifact", lambda p, a=None: uploads.append(str(p)))
    results = iter([EvalResult(win_rate=0.2, mean_damage_dealt=0, mean_damage_taken=0, mean_return=0.0)])
    cb = EvalCallback(Settings(game="sf", eval_freq=100, eval_sampled=False), tmp_path, eval_fn=lambda *a, **k: next(results))
    model = _RandomModel(MultiBinary(12))
    cb.init_callback(model)
    model.num_timesteps = 100
    cb.on_step()
    assert any(u.endswith("best_model.zip") for u in uploads)


def test_sonic_score_comes_from_held_out_levels_but_best_model_from_training_levels():
    from sega_rl.envs.sonic import HELD_OUT
    from sega_rl.evaluate import run_levels_eval, score

    seen = []

    def fake(model, cfg, episodes, video_path=None, max_steps=None, heartbeat=None, deterministic=True):
        seen.append(cfg.state)
        held = cfg.state in HELD_OUT
        return EvalResult(win_rate=0.25 if held else 0.75, mean_damage_dealt=0, mean_damage_taken=0, mean_return=1,
                          mean_progress=0.4 if held else 0.9)

    r = run_levels_eval(None, Settings(game="sonic"), eval_fn=fake)
    assert len(seen) == 8 and r.win_rate == 0.25 and r.train_win_rate == 0.75
    assert score("sonic", r) == (0.75, 0.9)


def test_every_check_is_also_played_sampled(tmp_path):
    from gymnasium.spaces import MultiBinary

    played = []

    def fake(model, cfg, episodes, video_path=None, heartbeat=None, deterministic=True):
        played.append(deterministic)
        return EvalResult(win_rate=0.4 if deterministic else 0.6, mean_damage_dealt=0, mean_damage_taken=0, mean_return=0)

    cb = EvalCallback(Settings(game="sf", eval_freq=100), tmp_path, eval_fn=fake)
    model = _RandomModel(MultiBinary(12))
    cb.init_callback(model)
    model.num_timesteps = 100
    cb.on_step()
    assert played == [True, False] and model.logger.records["eval_sampled/win_rate"] == 0.6
    assert model.logger.records["eval/win_rate"] == 0.4  # best_model still follows the likeliest-move score


def test_sf_is_scored_on_opponents_it_never_trained_on(tmp_path):
    from sega_rl.evaluate import run_fights_eval, score

    for split, names in (("train", ("Ken", "Guile")), ("heldout", ("Sagat",))):
        (tmp_path / split).mkdir()
        for n in names:
            (tmp_path / split / f"{n}.state").touch()
    seen = []

    def fake(model, cfg, episodes, video_path=None, max_steps=None, heartbeat=None, deterministic=True):
        seen.append(cfg.state.split("/")[-1])
        held = "Sagat" in cfg.state
        return EvalResult(win_rate=0.3 if held else 0.8, mean_damage_dealt=1, mean_damage_taken=1, mean_return=1, rounds_done=5)

    r = run_fights_eval(None, Settings(game="sf", sf_states_dir=tmp_path), eval_fn=fake)
    assert len(seen) == 3 and r.win_rate == 0.3 and r.train_win_rate == 0.8
    assert score("sf", r)[0] == 0.8  # the best model is never picked by the held-out fights

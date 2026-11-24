from __future__ import annotations

import json

import optuna
from stable_baselines3.common.vec_env import VecEnvWrapper

from sega_rl.config import Settings
from sega_rl.envs.factory import make_vec_env
from sega_rl.train import build_model, run
from sega_rl.tune import suggest_params


def _cfg(tmp_path, **kw):
    base = dict(game="airstriker", n_envs=2, total_steps=1024, eval_freq=1024, ckpt_freq=512,
                eval_episodes=1, eval_max_steps=100, runs_dir=tmp_path, run_id="t")
    base.update(kw)
    return Settings(**base)


def test_smoke_writes_artifacts(tmp_path):
    cfg = _cfg(tmp_path)
    assert run(cfg) == "done"
    rd = tmp_path / "t"
    assert list((rd / "checkpoints").glob("*.zip"))
    assert json.loads((rd / "health.json").read_text())["status"] in {"HEALTHY", "WARN"}
    assert (rd / "config.json").exists()
    assert (rd / "DONE").read_text() == "done"
    assert (rd / "best_model.zip").exists()


def test_resume_continues_timesteps(tmp_path):
    run(_cfg(tmp_path, total_steps=512, eval_freq=10_000))
    first = sorted((tmp_path / "t" / "checkpoints").glob("*.zip"))
    assert first
    run(_cfg(tmp_path, total_steps=1024, eval_freq=10_000))
    last = sorted((tmp_path / "t" / "checkpoints").glob("*.zip"))[-1]
    assert int(last.stem.split("_")[-1]) >= 1024


def test_budget_stop_writes_done(tmp_path):
    assert run(_cfg(tmp_path, max_hours=0.0, eval_freq=10_000)) == "budget"
    assert (tmp_path / "t" / "DONE").read_text() == "budget"


class _CrashAfter(VecEnvWrapper):
    def __init__(self, venv, n):
        super().__init__(venv)
        self.n = n

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        self.n -= 1
        if self.n < 0:
            raise EOFError("worker died")
        return self.venv.step_wait()


def test_env_crash_marks_critical(tmp_path):
    cfg = _cfg(tmp_path, eval_freq=10_000)
    status = run(cfg, env_factory=lambda c: _CrashAfter(make_vec_env(c), 300))
    assert status == "critical"
    health = json.loads((tmp_path / "t" / "health.json").read_text())
    assert health["status"] == "CRITICAL" and "env_crash" in health["reasons"]
    assert list((tmp_path / "t" / "checkpoints").glob("*.zip"))


def test_params_file_override(tmp_path):
    pf = tmp_path / "p.json"
    pf.write_text(json.dumps({"n_steps": 64, "gamma": 0.9}))
    cfg = _cfg(tmp_path, params_file=pf)
    env = make_vec_env(cfg)
    try:
        model = build_model(cfg, env)
        assert model.n_steps == 64 and model.gamma == 0.9
    finally:
        env.close()


def test_suggest_params_space():
    params = suggest_params(optuna.trial.FixedTrial(
        {"n_steps": 4096, "gamma": 0.95, "learning_rate": 5e-5, "clip_range": 0.2, "gae_lambda": 0.9}))
    assert params["n_steps"] == 4096 and params["learning_rate"] == 5e-5

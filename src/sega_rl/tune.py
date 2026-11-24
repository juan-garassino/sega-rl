from __future__ import annotations

import json
import logging

import optuna
from stable_baselines3 import PPO

from sega_rl.config import Settings
from sega_rl.envs.factory import make_vec_env
from sega_rl.evaluate import run_eval
from sega_rl.train import PPO_DEFAULTS

logger = logging.getLogger(__name__)


def suggest_params(trial: optuna.Trial) -> dict:
    """Nicholas Renotte's original search space."""
    return {
        "n_steps": trial.suggest_int("n_steps", 2048, 8192),
        "gamma": trial.suggest_float("gamma", 0.8, 0.9999, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.99),
    }


def run(cfg: Settings, n_trials: int = 10, trial_steps: int = 100_000) -> dict:
    def objective(trial: optuna.Trial) -> float:
        params = {**PPO_DEFAULTS, **suggest_params(trial)}
        env = make_vec_env(cfg)
        try:
            model = PPO("CnnPolicy", env, seed=cfg.seed, verbose=0, **params)
            model.learn(total_timesteps=trial_steps)
        finally:
            env.close()
        return run_eval(model, cfg, cfg.n_eval_episodes).mean_return

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.run_dir / "best_params.json"
    out.write_text(json.dumps(study.best_params, indent=2))
    logger.info("best params %s -> %s", study.best_params, out)
    return study.best_params

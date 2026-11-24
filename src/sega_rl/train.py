from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import Logger, make_output_format
from stable_baselines3.common.vec_env import VecEnv

from sega_rl.config import Settings
from sega_rl.envs.factory import make_vec_env
from sega_rl.evaluate import EvalCallback
from sega_rl.monitoring import mlflow_tracking
from sega_rl.monitoring.health import HealthCallback, HealthMonitor

logger = logging.getLogger(__name__)

# SB3 rl-zoo Atari PPO defaults; overridden by cfg.params_file (e.g. tune output).
PPO_DEFAULTS = dict(n_steps=128, batch_size=256, n_epochs=4, learning_rate=2.5e-4, clip_range=0.1,
                    ent_coef=0.01, gamma=0.99, gae_lambda=0.95)
ENV_CRASH_ERRORS = (EOFError, BrokenPipeError, ConnectionResetError)


def _linear(start: float) -> Callable[[float], float]:
    return lambda progress_remaining: start * progress_remaining


def build_model(cfg: Settings, env: VecEnv) -> PPO:
    params = dict(PPO_DEFAULTS)
    if cfg.params_file:
        params.update(json.loads(Path(cfg.params_file).read_text()))
    params["learning_rate"] = _linear(params["learning_rate"])
    return PPO("CnnPolicy", env, seed=cfg.seed, verbose=0, **params)


class CheckpointKeeper(BaseCallback):
    """Saves ckpt_<steps>.zip every cfg.ckpt_freq steps, keeping the newest cfg.keep_ckpts."""

    def __init__(self, cfg: Settings, ckpt_dir: Path):
        super().__init__()
        self.cfg = cfg
        self.ckpt_dir = ckpt_dir
        self.next_save = cfg.ckpt_freq

    def _on_training_start(self) -> None:
        self.next_save = (self.model.num_timesteps // self.cfg.ckpt_freq + 1) * self.cfg.ckpt_freq

    def save(self) -> Path:
        path = self.ckpt_dir / f"ckpt_{self.model.num_timesteps:010d}.zip"
        self.model.save(path)
        for old in sorted(self.ckpt_dir.glob("ckpt_*.zip"))[: -self.cfg.keep_ckpts]:
            old.unlink()
        return path

    def _on_step(self) -> bool:
        if self.model.num_timesteps >= self.next_save:
            self.save()
            self.next_save += self.cfg.ckpt_freq
        return True


class BudgetCallback(BaseCallback):
    """Stops training at the first rollout boundary after the wall-clock deadline."""

    def __init__(self, deadline_ts: float):
        super().__init__()
        self.deadline_ts = deadline_ts
        self.hit = False

    def _on_rollout_end(self) -> None:
        if time.time() >= self.deadline_ts:
            self.hit = True
            logger.warning("time budget reached, stopping")

    def _on_step(self) -> bool:
        return not self.hit


class HealthReporter:
    """Mirrors health into MLflow: tags on every status change, health.json at most every upload_every_s."""

    def __init__(self, health_json: Path, upload_every_s: float = 600.0, clock: Callable[[], float] = time.time):
        self.health_json = health_json
        self.upload_every_s = upload_every_s
        self.clock = clock
        self.last_key: tuple | None = None
        self.last_upload: float | None = None

    def __call__(self, report) -> None:
        key = (report.status, tuple(report.reasons))
        if key != self.last_key:
            mlflow_tracking.set_tag("health_status", report.status)
            mlflow_tracking.set_tag("health_reasons", ",".join(report.reasons))
            self.last_key = key
        now = self.clock()
        if self.last_upload is None or now - self.last_upload >= self.upload_every_s:
            mlflow_tracking.log_artifact(self.health_json)
            self.last_upload = now


def _latest_checkpoint(ckpt_dir: Path) -> Path | None:
    ckpts = sorted(ckpt_dir.glob("ckpt_*.zip"))
    return ckpts[-1] if ckpts else None


def _make_logger(run_dir: Path) -> Logger:
    tb = str(run_dir / "tb")
    formats = [make_output_format("stdout", tb), make_output_format("tensorboard", tb)]
    if mlflow_tracking.enabled():
        formats.append(mlflow_tracking.MLflowOutputFormat())
    return Logger(tb, formats)


def run(cfg: Settings, env_factory: Callable[[Settings], VecEnv] = make_vec_env) -> str:
    """Train (or resume) until cfg.steps, CRITICAL health, or the time budget. Returns done|critical|budget."""
    run_dir = cfg.run_dir
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(cfg.model_dump_json(indent=2))
    deadline = time.time() + cfg.hours_budget * 3600

    env = env_factory(cfg)
    resume_from = _latest_checkpoint(ckpt_dir)
    if resume_from:
        logger.info("resuming from %s", resume_from)
        model = PPO.load(resume_from, env=env)
    else:
        model = build_model(cfg, env)
    model.set_logger(_make_logger(run_dir))
    mlflow_tracking.start(cfg)

    health = HealthCallback(
        HealthMonitor(cfg, check_button_spam=cfg.game != "sonic"),
        run_dir / "health.json",
        on_critical=lambda r: logger.error("CRITICAL training health: %s", r.reasons),
        stall_secs=cfg.fps_zero_secs,
        on_report=HealthReporter(run_dir / "health.json"),
    )
    keeper = CheckpointKeeper(cfg, ckpt_dir)
    budget = BudgetCallback(deadline)
    callbacks = CallbackList([keeper, health, EvalCallback(cfg, run_dir), budget])

    status = "done"
    remaining = cfg.steps - model.num_timesteps
    try:
        if remaining > 0:
            model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=resume_from is None,
                        log_interval=1)
        if health.last_report and health.last_report.status == "CRITICAL":
            status = "critical"
        elif budget.hit:
            status = "budget"
    except ENV_CRASH_ERRORS as e:
        logger.exception("environment worker crashed: %s", e)
        health.write(health.monitor.mark_critical("env_crash"))
        status = "critical"
    finally:
        keeper.model = model
        final_ckpt = keeper.save()
        try:
            env.close()
        except Exception:
            logger.exception("env.close failed")

    (run_dir / "DONE").write_text(status)
    for artifact in ("health.json", "best.json", "best_model.zip", "DONE"):
        if (run_dir / artifact).exists():
            mlflow_tracking.log_artifact(run_dir / artifact)
    mlflow_tracking.log_artifact(final_ckpt, "checkpoints")
    mlflow_tracking.end(status)
    logger.info("run %s finished: %s at %d steps", cfg.run_id, status, model.num_timesteps)
    return status

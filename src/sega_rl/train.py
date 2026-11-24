from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Callable

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import Logger, make_output_format
from stable_baselines3.common.vec_env import VecEnv

from sega_rl.config import Settings, default_learning_rate
from sega_rl.envs.factory import make_vec_env
from sega_rl.evaluate import EvalCallback, run_baselines
from sega_rl.monitoring import mlflow_tracking
from sega_rl.monitoring.health import HealthCallback, HealthMonitor

logger = logging.getLogger(__name__)

# SB3 rl-zoo Atari PPO defaults; overridden by cfg.params_file (e.g. tune output).
PPO_DEFAULTS = dict(n_steps=128, batch_size=256, n_epochs=4, learning_rate=2.5e-4, clip_range=0.1,
                    ent_coef=0.01, gamma=0.99, gae_lambda=0.95)
ENV_CRASH_ERRORS = (EOFError, BrokenPipeError, ConnectionResetError)


def torch_threads(cpu_count: int, n_envs: int) -> int:
    """CPU pods: leave one core per emulator process, give torch the rest (at least 2)."""
    return max(2, cpu_count - n_envs)


def _linear(start: float) -> Callable[[float], float]:
    return lambda progress_remaining: start * progress_remaining


def build_model(cfg: Settings, env: VecEnv) -> PPO:
    params = dict(PPO_DEFAULTS)
    params["learning_rate"] = cfg.learning_rate or default_learning_rate(cfg.game)
    if cfg.params_file:
        params.update(json.loads(Path(cfg.params_file).read_text()))
    params["learning_rate"] = _linear(params["learning_rate"])
    if not torch.cuda.is_available():
        torch.set_num_threads(torch_threads(os.cpu_count() or 2, cfg.n_envs))
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
        mlflow_tracking.log_artifact(path, "checkpoints")  # durable copy: GCS access ends after 1 h
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


def _stall_handler(run_dir: Path) -> Callable:
    def on_stall(report) -> None:
        logger.critical("no training progress for too long, exiting: %s", report.reasons)
        (run_dir / "DONE").write_text("stalled")
        mlflow_tracking.set_tag("health_status", "CRITICAL")
        mlflow_tracking.set_tag("health_reasons", ",".join(report.reasons))
        mlflow_tracking.end("stalled")
        os._exit(3)  # the main thread is stuck; the entrypoint's EXIT trap syncs and removes the pod

    return on_stall


def run(cfg: Settings, env_factory: Callable[[Settings], VecEnv] = make_vec_env) -> str:
    """Train (or resume) until cfg.steps, CRITICAL health, or the time budget.

    Returns done | critical | budget | mlflow_unavailable. DONE, the final checkpoint and the MLflow
    uploads happen on every exit path.
    """
    run_dir = cfg.run_dir
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(cfg.model_dump_json(indent=2))
    deadline = time.time() + cfg.hours_budget * 3600

    if mlflow_tracking.enabled() and not mlflow_tracking.start(cfg, probe=run_dir / "config.json"):
        logger.error("MLflow is configured but unusable — refusing to train without a durable channel")
        (run_dir / "DONE").write_text("mlflow_unavailable")
        return "mlflow_unavailable"

    health = HealthCallback(
        HealthMonitor(cfg, check_button_spam=cfg.game != "sonic"),
        run_dir / "health.json",
        on_critical=lambda r: logger.error("CRITICAL training health: %s", r.reasons),
        stall_secs=cfg.fps_zero_secs,
        on_stall=_stall_handler(run_dir),
        on_report=HealthReporter(run_dir / "health.json"),
    )
    keeper = CheckpointKeeper(cfg, ckpt_dir)
    budget = BudgetCallback(deadline)

    status, model, env = "done", None, None
    try:
        env = env_factory(cfg)
        resume_from = _latest_checkpoint(ckpt_dir)
        if resume_from:
            logger.info("resuming from %s", resume_from)
            model = PPO.load(resume_from, env=env)
        else:
            model = build_model(cfg, env)
        model.set_logger(_make_logger(run_dir))
        if cfg.game in ("sf", "sonic"):
            mlflow_tracking.log_metrics(run_baselines(cfg, run_dir, env.action_space), step=0)
            mlflow_tracking.log_artifact(run_dir / "baselines.json")
        callbacks = CallbackList([keeper, health, EvalCallback(cfg, run_dir, heartbeat=health.heartbeat), budget])
        remaining = cfg.steps - model.num_timesteps
        if remaining > 0:
            model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=resume_from is None,
                        log_interval=10)
        if health.last_report and health.last_report.status == "CRITICAL":
            status = "critical"
        elif budget.hit:
            status = "budget"
    except ENV_CRASH_ERRORS as e:
        logger.exception("environment worker crashed: %s", e)
        health.write(health.monitor.mark_critical("env_crash"))
        status = "critical"
    except Exception as e:  # e.g. NaN logits raise inside predict before any health check can run
        logger.exception("training crashed: %s", e)
        health.write(health.monitor.mark_critical("crash"))
        status = "critical"
    finally:
        if model is not None:
            keeper.model = model
            keeper.save()
        if env is not None:
            try:
                env.close()
            except Exception:
                logger.exception("env.close failed")
        (run_dir / "DONE").write_text(status)
        for artifact in ("health.json", "best.json", "DONE"):
            if (run_dir / artifact).exists():
                mlflow_tracking.log_artifact(run_dir / artifact)
        mlflow_tracking.set_tag("health_status", health.last_report.status if health.last_report else "UNKNOWN")
        mlflow_tracking.end(status)

    logger.info("run %s finished: %s at %d steps", cfg.run_id, status, model.num_timesteps if model else 0)
    return status

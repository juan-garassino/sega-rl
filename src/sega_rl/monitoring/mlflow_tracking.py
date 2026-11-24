"""Optional MLflow tracking against the shared garassino-mlflow server.

The server runs with --serve-artifacts, so the pod uploads videos/models/health through it with no GCS
credentials — this is the durable channel once the pod's 1 h GCS token has expired.

Active only when MLFLOW_TRACKING_URI is set (the RunPod job); everything no-ops locally.
Auth: the API sits behind MLflow basic auth — mlflow reads MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD.
A tracker error never fails a training run.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from stable_baselines3.common.logger import KVWriter

from sega_rl.config import Settings

logger = logging.getLogger(__name__)

EXPERIMENT = "sega-rl"
_active = False  # True only after start() succeeded; otherwise mlflow would auto-create stray runs


def _require_active(fn):
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs) if _active else None

    wrapper.__name__ = fn.__name__
    return wrapper


def enabled() -> bool:
    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


def _safe(fn):
    def wrapper(*args, **kwargs):
        if not enabled():
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # tracker hiccups must never stop training
            logger.warning("mlflow %s skipped: %s: %s", fn.__name__, type(e).__name__, e)
            return None

    return wrapper


@_safe
def start(cfg: Settings) -> None:
    global _active
    import mlflow

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(EXPERIMENT)
    mlflow.start_run(run_name=cfg.run_id)
    mlflow.set_tag("game", cfg.game)
    mlflow.log_params({k: str(v) for k, v in cfg.model_dump().items()})
    _active = True


@_require_active
@_safe
def log_metrics(metrics: dict[str, float], step: int) -> None:
    import mlflow

    mlflow.log_metrics({k.replace("/", "_"): float(v) for k, v in metrics.items()}, step=step)


@_require_active
@_safe
def log_artifact(path: Path, artifact_path: str | None = None) -> None:
    import mlflow

    mlflow.log_artifact(str(path), artifact_path=artifact_path)


@_require_active
@_safe
def set_tag(key: str, value: str) -> None:
    import mlflow

    mlflow.set_tag(key, value)


@_require_active
@_safe
def end(status: str) -> None:
    import mlflow

    global _active
    mlflow.set_tag("final_status", status)
    mlflow.end_run()
    _active = False


class MLflowOutputFormat(KVWriter):
    """Streams every numeric SB3 logger value (train/*, rollout/*, eval/*, health/*) to MLflow."""

    def write(self, key_values, key_excluded, step=0):
        log_metrics({k: v for k, v in key_values.items() if isinstance(v, (int, float))}, step=step)

    def close(self):
        pass

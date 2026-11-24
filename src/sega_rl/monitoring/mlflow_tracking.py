"""Optional MLflow tracking against the shared garassino-mlflow server.

Active only when MLFLOW_TRACKING_URI is set (the RunPod job); everything no-ops locally.
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
    import mlflow

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(EXPERIMENT)
    mlflow.start_run(run_name=f"{cfg.game}-{cfg.run_id}")
    mlflow.log_params({k: str(v) for k, v in cfg.model_dump().items()})


@_safe
def log_metrics(metrics: dict[str, float], step: int) -> None:
    import mlflow

    mlflow.log_metrics({k.replace("/", "_"): float(v) for k, v in metrics.items()}, step=step)


@_safe
def log_artifact(path: Path) -> None:
    import mlflow

    mlflow.log_artifact(str(path))


@_safe
def end(status: str) -> None:
    import mlflow

    mlflow.set_tag("final_status", status)
    mlflow.end_run()


class MLflowOutputFormat(KVWriter):
    """Streams every numeric SB3 logger value (train/*, rollout/*, eval/*, health/*) to MLflow."""

    def write(self, key_values, key_excluded, step=0):
        log_metrics({k: v for k, v in key_values.items() if isinstance(v, (int, float))}, step=step)

    def close(self):
        pass

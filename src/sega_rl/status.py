"""Read a run's live state from the shared MLflow tracker (works from the laptop, no GCS creds)."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from sega_rl.monitoring.mlflow_tracking import EXPERIMENT

DEFAULT_URI = "https://garassino-mlflow-920423386248.europe-west1.run.app"
SHOWN = [("eval_win_rate", "win rate"), ("eval_mean_progress", "level progress"), ("eval_mean_return", "eval return"),
         ("rollout_ep_rew_mean", "train reward"), ("time_fps", "fps"), ("train_approx_kl", "approx_kl"),
         ("train_explained_variance", "explained var")]


def format_status(run_name: str, metrics: dict[str, float], tags: dict[str, str | None], step: int) -> str:
    lines = [f"{run_name} @ {step:,} steps — health {tags.get('health_status') or '?'}"
             + (f" ({tags['health_reasons']})" if tags.get("health_reasons") else "")]
    if tags.get("final_status"):
        lines.append(f"finished: {tags['final_status']}")
    lines += [f"  {label:<15} {metrics[key]:.4g}" for key, label in SHOWN if key in metrics]
    return "\n".join(lines)


def fetch(run_name: str) -> str:
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_URI))
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        return f"no MLflow experiment {EXPERIMENT!r}"
    runs = client.search_runs([exp.experiment_id], f"tags.mlflow.runName = '{run_name}'", max_results=1)
    if not runs:
        return f"no MLflow run named {run_name!r}"
    run = runs[0]
    step = max((h.step for h in client.get_metric_history(run.info.run_id, "time_total_timesteps")), default=0)
    return format_status(run_name, run.data.metrics, run.data.tags, step)


def _client():
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_URI))


def fetch_videos(run_name: str, dest: Path, client=None) -> list[Path]:
    """Download the run's eval replays that aren't on this machine yet into dest/<run_name>/."""
    client = client or _client()
    exp = client.get_experiment_by_name(EXPERIMENT)
    runs = client.search_runs([exp.experiment_id], f"tags.mlflow.runName = '{run_name}'", max_results=1) if exp else []
    if not runs:
        return []
    out_dir = dest / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(client.download_artifacts(runs[0].info.run_id, "videos", tmp))
        new = []
        for mp4 in sorted(src.glob("*.mp4")):
            target = out_dir / mp4.name
            if not target.exists():
                shutil.copy2(mp4, target)
                new.append(target)
    return new

# RunPod runbook

The image is built by CI (`build-push.yml`) — never `docker build` on the laptop.

## Launch

`make launch GAME=sf` (uploads that ROM to the private bucket, then dispatches **deploy-runpod**; extra inputs via
`ARGS="-f n_envs=32 -f max_eur=5"`). The workflow mints a 1 h GCS token via WIF and creates the pod.
GCS is only used in the first minutes (ROMs, resume state); after that the token may expire — live metrics, health,
eval videos, the best model and the final checkpoint go through MLflow (`--serve-artifacts`, basic auth).

## What the pod does (`entrypoint.sh`)

1. Exits (and removes itself) if `gs://garassino-ml-artifacts/sega-rl/runs/<RUN_ID>/DONE` exists.
2. Pulls ROMs from `gs://garassino-ml-artifacts/sega-rl/roms/` and **deletes them from GCS immediately**
   (uploaded right before each launch by `make launch`; the deploy workflow also deletes them if no pod was
   created), then imports and verifies them (`sega-rl check-rom`). Failure → CRITICAL `rom`, pod removed.
3. Pulls the run dir (resume from the newest checkpoint).
4. Reads its own USD/h (`pod_self.sh cost`) → `SEGARL_COST_PER_HR_USD`; training stops at
   min(`max_hours`, `max_eur` worth of hours).
5. Syncs the run dir to GCS every `SYNC_INTERVAL` (600 s) while the token lasts; failures never stop training.
6. Runs `sega-rl $MODE`; on any exit (incl. missing config): final sync, then `pod_self.sh remove`, retried 5×
   (billing stops). If even that fails, the log says so — remove the pod in the RunPod console.

## Watch a run

```bash
make status RUN=<id>       # from MLflow: health status + reasons, eval, fps
```

Everything else (videos, best_model.zip, health.json, final checkpoint): MLflow experiment `sega-rl`, run name = RUN_ID.
Needs `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` in your shell (and as GitHub secrets for the pod).

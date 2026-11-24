# RunPod runbook

The image is built by CI (`build-push.yml`) — never `docker build` on the laptop.

## Launch

GitHub → Actions → **deploy-runpod** → inputs `game` (sf|sonic|airstriker), `mode` (train), `total_steps`,
`n_envs` (≈ vCPUs), `max_hours`, `max_eur`. The workflow mints a 1 h GCS token via WIF and creates the pod;
**refresh-token** renews it every 50 min.

## What the pod does (`entrypoint.sh`)

1. Exits (and removes itself) if `gs://garassino-ml-artifacts/sega-rl/runs/<RUN_ID>/DONE` exists.
2. Pulls ROMs from `gs://garassino-ml-artifacts/sega-rl/roms/`, imports them, then **deletes them from GCS**
   (ROMs are uploaded right before each launch by `make launch`; not used for airstriker).
   Failure → `health.json` CRITICAL `rom`, pod removed.
3. Pulls the run dir (resume from the newest checkpoint).
4. Reads its own USD/h (`pod_self.sh cost`) → `SEGARL_COST_PER_HR_USD`; training stops at
   min(`max_hours`, `max_eur` worth of hours).
5. Syncs the run dir to GCS every `SYNC_INTERVAL` (600 s); sync failures never stop training.
6. Runs `sega-rl $MODE`; on any exit: final sync, then `pod_self.sh remove` (billing stops).

## Watch a run

```bash
make health RUN_ID=<id>    # health.json: HEALTHY / WARN / CRITICAL + reasons
make results RUN_ID=<id>   # DONE status + best.json
```

Videos: `gs://garassino-ml-artifacts/sega-rl/runs/<RUN_ID>/videos/`. Metrics: MLflow experiment `sega-rl`.

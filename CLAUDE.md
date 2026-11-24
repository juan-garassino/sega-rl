# CLAUDE.md — RL-sega-rl

PPO (SB3) on Sega Genesis games via stable-retro: Street Fighter II' SCE (`sf`), Sonic 1 (`sonic`), Airstriker
(`airstriker`, bundled, tests only). Trained on RunPod (archetype B); monitored via MLflow. Design + plan:
`docs/superpowers/specs/2026-11-24-sega-rl-design.md`, `docs/superpowers/plans/2026-11-24-sega-rl.md`.

## Module map (`src/sega_rl/`)

| Module | Purpose |
|---|---|
| `config.py` | `Settings` (`SEGARL_*`), per-game defaults (`default_total_steps`, `default_max_eur`), `hours_budget` (EUR cap ÷ pod USD/h) |
| `envs/common.py` | `RetroEnv`: gray 84×84, frame skip + max-pool, frame/delta obs, button combos, truncation, `info["buttons"]` |
| `envs/street_fighter.py` | `SFReward` (damage diff/176 ± round), round_won / damage info |
| `envs/sonic.py` | `SonicReward` (new ground /100, +10 at act end, ends on lost life), `SONIC_ACTIONS`, `LEVEL_MAX_X` |
| `envs/airstriker.py` | score reward, test/smoke game |
| `envs/factory.py` | `make_vec_env`: SubprocVecEnv(forkserver) → VecMonitor → VecFrameStack(4) |
| `monitoring/health.py` | `HealthMonitor` rules, `HealthCallback` (health.json, watchdog thread → exit 3 on stall) |
| `monitoring/mlflow_tracking.py` | no-op unless `MLFLOW_TRACKING_URI`; logging only after a successful `start()`; SB3 KVWriter |
| `monitoring/video.py` | mp4 writer |
| `evaluate.py` | `run_eval` (deterministic, in-process), `EvalCallback` (video, best_model.zip, best.json) |
| `train.py` | `run()`: resume, callbacks (checkpoints, health, eval, budget), env-crash handling, DONE marker |
| `tune.py` | Optuna, upstream search space → best_params.json (`SEGARL_PARAMS_FILE`) |
| `status.py` | `sega-rl status <run>`: live state from MLflow |
| `cli.py` | `sega-rl train|tune|eval|smoke|status|check-rom|import-roms` |

`runpod/`: `Dockerfile` (CUDA 12.8 devel, uv, torch 2.14.1 cu126 swapped in, import check), `entrypoint.sh`
(lifecycle, see runpod/README.md), `scripts/pod_self.sh` (cost / self-terminate with retries).
`.github/workflows/`: `ci.yml`, `build-push.yml` (GHCR), `deploy-runpod.yml` (WIF token + pod create).

## Gotchas

- **One emulator per process** (stable-retro). Training envs live in SubprocVecEnv workers, so the main process may host
  exactly one (eval). Never create two in one process; close before creating another.
- `stable_retro.make` must get `render_mode=None` (default "human" needs OpenGL).
- Torch: pyproject pins the CPU index (laptop/CI). Don't use extras for torch — SB3's transitive torch then resolves
  from PyPI (6 GB CUDA build). The image reinstalls torch from cu126.
- GCS token on the pod lasts 1 h and is not refreshed; anything that must survive goes to MLflow (experiments use
  `mlflow-artifacts:/<name>`; `train.run` refuses to start if MLflow is configured but an upload probe fails).
- `RetroEnv` merges per-frame events across frame skip (`damage_*` summed, `round_won`/`level_complete` kept) —
  reward fns may emit events on any sub-frame.
- Resets take 0–30 random no-op frames (`SEGARL_NOOP_MAX`); without it every episode replays the same savestate.
- Torch is pinned to 2.14.1 so the CPU lock and the image's cu126 wheel are the same version.
- MLflow API needs `MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD` (basic auth behind the GitHub OAuth proxy).
- Local heavy jobs one at a time (laptop froze 2026-11-09): `nice`, ≤ 2 envs, never `docker build`.

## Env vars

All `Settings` fields map to `SEGARL_<FIELD>` (see inline comments in `config.py`). Pod-only: `MODE`, `RUN_ID`,
`GCS_BUCKET`, `GCS_ACCESS_TOKEN`, `SEGARL_RUNPOD_API_KEY`, `RUNPOD_POD_ID` (injected), `SYNC_INTERVAL`, `MLFLOW_*`.

# sega-rl

PPO agents for Sega Genesis games on [stable-retro](https://github.com/Farama-Foundation/stable-retro):
**Street Fighter II' Special Champion Edition** and **Sonic The Hedgehog**, trained on RunPod with live
training-health monitoring. Part of an RL series (one repo per console: `sega-rl`, `atari-rl`, `gameboy-rl`).

Street Fighter started as a port of [nicknochnack/StreetFighterRL](https://github.com/nicknochnack/StreetFighterRL)
(gym-retro, 2022) to a maintained stack, with its bugs fixed:

| Upstream | Here |
|---|---|
| `uint8` frame delta wraps around | plain frames + 4-frame stack (`SEGARL_OBS_MODE=delta` = fixed signed delta) |
| reward = score delta | (damage dealt − damage taken)/176 ± 1 per round (`SEGARL_REWARD_MODE=score` = upstream) |
| 1 env, no frame skip | 16 parallel games, frame skip 4 with max-pooling |
| checkpoints only | eval every 250k steps, replay video, best model by win rate, health monitor |

## Games and ROMs

ROMs are **not** in this repo, the image, or any public place (a test fails if one is committed). Put your own
dumps in `roms/` (gitignored):

| Game | `SEGARL_GAME` | ROM (SHA-1 must match stable-retro) |
|---|---|---|
| Street Fighter II' SCE | `sf` | *(USA)* `a5aad1d108046d9388e33247610dafb4c6516e0b` |
| Sonic The Hedgehog | `sonic` | *(Japan, Korea)* `69e102855d4389c3fd1a8f3dc7d193f8eee5fe5b` — the USA/Europe dump does not match |
| Airstriker (bundled, free) | `airstriker` | — used by tests and the smoke run |

## Local

```bash
make install        # uv, Python 3.12, CPU torch
make test           # pytest (ROM tests skip unless ROMs are imported)
make import-roms    # optional: import roms/ for local ROM tests
make smoke          # 2 envs, 2k steps on Airstriker, ~90 s
```

## Training on RunPod

```bash
make launch GAME=sf                 # uploads the ROM to the private bucket, dispatches deploy-runpod
make status RUN=sf-20261124-180000  # health, eval win rate / progress, fps — read from MLflow
make videos RUN=sf-20261124-180000  # eval replays + best_model.zip → ~/Videos/rl-series/<run>/
make play RUN=sf-20261124-180000 GAME=sf  # replay the best model locally → local_play.mp4
make launch GAME=sf ARGS="-f run_id=<same id> -f max_eur=5"  # extend a run that stopped on its budget
```

The pod pulls the ROM and deletes it from GCS, trains, and **removes itself** when it finishes, when training health
turns CRITICAL, or at the budget cap (min of `max_hours` and `max_eur` at the pod's hourly price). Details:
[runpod/README.md](runpod/README.md).

### Health monitor

After every PPO update the run is classified HEALTHY / WARN / CRITICAL (`health.json`, MLflow tags `health_status`,
`health_reasons`, metric `health/status`):

| Signal | WARN | CRITICAL (stops the run) |
|---|---|---|
| NaN/inf in loss or weights | — | immediately |
| approx KL | > 0.05 | > 0.2 for 3 updates |
| clip fraction | > 0.3 | — |
| policy entropy | < 20% of start, in the first 25% of training | — |
| explained variance | < 0 after 10% of training | < −0.5 for 10 updates |
| episode reward | no new best over the last 20% of steps | — |
| throughput | < 50% of the first-5-min median | 0 for 5 min, or no env step for 5 min (watchdog) |
| button spam (SF) | one button in > 90% of steps | — |

## Run ledger

| Run | Game | Steps | Hours | € (RunPod) | Result | Health incidents |
|---|---|---|---|---|---|---|

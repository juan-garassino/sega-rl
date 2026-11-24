# RL-sega-rl — design

Date: 2026-11-24 · Status: approved 2026-11-24 (reworked same day: per-console repos, Sonic added)

## Intent

First repo of Juan's **RL series**: one repo per console — `RL-sega-rl` (Street Fighter + Sonic, this repo),
`RL-atari-rl`, `RL-gameboy-rl` (Pokémon) — each trained on RunPod. This repo goes first and becomes the
template the other two copy (workspace convention: copy-paste between repos, no shared package).
Both Sega games run on stable-retro and share one trainer, one image and one monitor; `GAME=sf|sonic|airstriker`.

Source: [nicknochnack/StreetFighterRL](https://github.com/nicknochnack/StreetFighterRL) (Feb 2022,
3 notebooks, gym-retro + SB3 1.x). That stack no longer installs (gym-retro needs Python ≤3.8).

The ask is "train it now, with good monitoring of a healthy training": the run must make it obvious,
while it's running, whether the agent is learning or silently broken.

## Goals

- Port the tutorial to a maintained stack: `stable-retro` + `gymnasium` + `stable-baselines3` 2.x, Python 3.12, uv.
- Keep Nick's design recognisable: PPO + `CnnPolicy`, grayscale 84×84, 4-frame stack, Optuna tuning.
- Fix the upstream defects that would waste paid GPU time (see "Changes vs upstream").
- Run on RunPod as an archetype-B job (`entrypoint.sh` + `MODE=` switch, artifacts in GCS).
- Training-health monitoring: live status, automatic warnings, auto-stop on unrecoverable failure.
- Replay videos of the agent fighting, uploaded during training.
- Tests and the free local smoke run on stable-retro's bundled **Airstriker** ROM (no ROM needed in CI).
- A Sonic adapter in the same repo (second game; first paid run is Street Fighter).

## Non-goals

- A shared package/harness across the three console repos.
- Atari / Game Boy work (later repos, copied from this one).
- Sonic level-generalisation (the Retro Contest train/test split): one level at a time here.
- Claude-in-the-loop modes (`research` / `improve` / `self-improve`) — can be added later, mirroring deep-sculpt.
- Beating a benchmark; multi-character or multi-stage curricula; a human-play UI.
- Touching the other console repos (`RL-atari-rl`, `RL-gameboy-rl`, `RL-pyboy-rl`) or `DATA-pokemon-api`.

## Anti-goals

- Paying for hours of RunPod time on a run that was broken from minute five.
- A ROM in git, in the Docker image, or in any public location.
- Service-account JSON keys, `requirements.txt`, pyenv.

## Constraints

- **ROMs** (verified, in gitignored `roms/`): SF2 `Street Fighter II' - Special Champion Edition (USA).md`,
  Sonic `Sonic The Hedgehog (Japan, Korea).md` (SHA-1 `69e102855d4389c3fd1a8f3dc7d193f8eee5fe5b`; the USA/Europe
  dump does NOT match). SF integration `StreetFighterIISpecialChampionEdition-Genesis-v0`
  (current stable-retro naming, package `stable_retro`) requires SHA-1 `a5aad1d108046d9388e33247610dafb4c6516e0b`
  (the USA release), imported with `python -m stable_retro.import <dir>`. Default state
  `Champion.Level1.RyuVsGuile`; built-in done = `continuetimer == 10` (game over).
- **Local machine**: no GPU, 8 cores, ~4 GB free RAM, froze on 2026-11-09 under load. Local runs are
  limited to tests + a small niced smoke (2 envs, ~2k steps). All real training is on RunPod.
- **Throughput is CPU-bound** (emulation), the CNN is tiny. Pod choice favours vCPU count over GPU class.
- **Cost reporting**: RunPod spend (Juan's card) and API credits (none in train mode) quoted separately.
- GCP: `garassino-ml`, bucket `gs://garassino-ml-artifacts/sega-rl/`, region `europe-west1`.

## Approaches considered

| | Approach | Trade-off |
|---|---|---|
| A | Faithful port — notebooks → scripts, only compatibility changes | Line-for-line with the video, but keeps the bugs; paid GPU on a broken setup |
| **B** | **Fixed + scaled port (chosen)** | Same core design, fixed and parallelised; template-friendly for the other 3 repos |
| C | Custom CleanRL-style PPO + fast vector env | Best $/step, but departs from both the tutorial and SB3; worse as a series template |

Chosen: **B** (proposed by Claude, not objected to).

## Changes vs upstream

| Upstream | Problem | Here |
|---|---|---|
| `frame_delta = obs - previous_frame` on `uint8` | wraps on negative differences → noise | `OBS_MODE=frame` (default): plain frames + 4-stack (motion is in the stack). `OBS_MODE=delta` kept for fidelity, computed in `int16` and rescaled to `uint8` |
| reward = score delta | ignores damage taken and round outcome; score is sparse/noisy | reward = `(enemy_health_lost − own_health_lost) / 176` + `±1` on round win/loss; `REWARD_MODE=score` keeps upstream behaviour |
| 1 env, `DummyVecEnv` | slow | `SubprocVecEnv`, `N_ENVS` (default 16) |
| no frame skip | 4× wasted compute | skip 4, max-pool last 2 frames |
| `COLOR_BGR2GRAY` on RGB frames | wrong channel weights | `COLOR_RGB2GRAY`, `INTER_AREA` resize |
| checkpoint every 10k calls, no eval | can't tell good from bad models | periodic deterministic eval, best-by-win-rate model kept |

Action space stays `MultiBinary(12)` with `use_restricted_actions=FILTERED`, as upstream.
Info keys used (stable-retro SF2 integration): `health`, `enemy_health`, `matches_won`,
`enemy_matches_won`, `score`. The Airstriker adapter maps its own reward (score) and has no health terms.

## Design

### Layout

```
RL-sega-rl/
  src/sega_rl/
    config.py              # BaseSettings, env_prefix="SEGARL_", every field commented with env var + example
    envs/
      common.py            # shared retro wrapper: preprocess (gray 84×84), frame skip 4 + max-pool, obs mode
      street_fighter.py    # SF reward (health diff + round bonus) + win-rate info
      sonic.py             # Sonic reward (x-progress) + discrete button combos
      airstriker.py        # bundled ROM, score reward; tests + smoke only
      factory.py           # make_vec_env(game, n_envs, seed) → VecFrameStack(VecMonitor(SubprocVecEnv))
    monitoring/
      health.py            # HealthCallback + rules → health.json
      mlflow_tracking.py   # copied from 027-ml-workspace pattern; no-op without MLFLOW_TRACKING_URI
      video.py             # eval episode → mp4
    train.py               # PPO build/resume/learn with callbacks
    tune.py                # Optuna study (upstream search space)
    evaluate.py            # N deterministic episodes → win rate, damage, video
    gcs.py                 # periodic sync of run dir to GCS (token-auth, like deep-sculpt)
    cli.py                 # argparse: train | tune | eval | smoke | import-rom
    __main__.py
  tests/
  runpod/{Dockerfile,entrypoint.sh,Makefile,README.md}
  Makefile  pyproject.toml  README.md  CLAUDE.md  .env.sample  .gitignore (roms/, runs/)
```

### Training

- PPO, `CnnPolicy`, SB3 Atari-zoo defaults as baseline: `n_steps=128`, `batch_size=256`, `n_epochs=4`,
  `lr=2.5e-4` linear decay, `clip_range=0.1`, `ent_coef=0.01`, `gamma=0.99`, `gae_lambda=0.95`.
- `MODE=tune` runs Optuna with Nick's search space (`n_steps`, `gamma`, `lr`, `clip_range`, `gae_lambda`),
  short trials, writes best params JSON that `train` can load via `SEGARL_PARAMS_FILE`.
- `TOTAL_STEPS` default 10M. Resume from latest checkpoint in GCS for the same `RUN_ID`.

### Health monitoring (the core requirement)

`HealthCallback` runs after every PPO update, reads SB3's logged values + episode infos, and classifies:

| Signal | WARN | CRITICAL |
|---|---|---|
| NaN/inf in loss or params | — | immediately |
| `train/approx_kl` | > 0.05 | > 0.2 for 3 updates |
| `train/clip_fraction` | > 0.3 | — |
| `train/entropy_loss` (policy entropy) | collapsed below 20% of initial before 25% of training | — |
| `train/explained_variance` | < 0 after 10% of training | < −0.5 for 10 updates |
| `rollout/ep_rew_mean` | no improvement over last 20% of steps | — |
| env FPS | < 50% of first-5-min median | 0 for 5 min (env hung) |
| action histogram | one button pressed in > 90% of steps (button-spam collapse) | — |
| win rate (eval) | logged, informs best-model selection | — |

Outputs:
- `runs/<RUN_ID>/health.json` (status, reasons, last values), rewritten each update, synced to GCS.
- WARN → logged + MLflow tag; CRITICAL → checkpoint, write status, stop the run (`SEGARL_STOP_ON_CRITICAL=true`).
- All metrics to TensorBoard (local dir) and MLflow (`garassino-mlflow`, experiment `sega-rl`, run = `RUN_ID`).
- `make health RUN_ID=...` prints the current health.json from GCS from the laptop.

Thresholds live in config so the other three repos can tune them.

### Eval + video

Every `EVAL_FREQ` (default 250k) steps: `EVAL_EPISODES` (default 5) deterministic matches → win rate,
mean damage dealt/taken; one episode recorded to mp4. Best model (by win rate, tiebreak damage diff) kept
as `best_model.zip`.

### Artifacts

`gs://garassino-ml-artifacts/sega-rl/runs/<RUN_ID>/{checkpoints/,best_model.zip,videos/,tb/,health.json,config.json}`.
Synced every `SYNC_INTERVAL` (default 600 s) and at exit. Last 3 checkpoints kept.

### ROM handling

- Never in git (gitignored `roms/`) or the image.
- Juan uploads his own dump once to `gs://garassino-ml-artifacts/sega-rl/roms/` (private bucket).
- `entrypoint.sh` pulls it at start and runs `python -m stable_retro.import`; `GAME=airstriker` skips this.

### RunPod

- Image: archetype B (`nvidia/cuda:12.8.0-devel-ubuntu22.04`, uv, non-root), pushed to GHCR
  (`ghcr.io/juan-garassino/sega-rl`). Contains no ROM, so public is fine.
- `entrypoint.sh` `MODE=train|tune|eval|smoke`, env: `GAME`, `RUN_ID`, `TOTAL_STEPS`, `N_ENVS`,
  `GCS_BUCKET`, `GCS_ACCESS_TOKEN`, `MLFLOW_TRACKING_URI`.
- GCS auth: short-lived token, same pattern as `007-deep-sculpt/runpod` (no SA keys).
- Pod choice: high vCPU count (≥16) with a cheap GPU; set `N_ENVS` ≈ vCPUs.
- `runpod/Makefile`: `build`, `push`, `pod-up`, `pod-down`, `logs`, `health`, `results`.
- **Pod lifecycle (Claude's call)**: the pod terminates itself when training ends, on CRITICAL, or when
  `MAX_HOURS` (default 6) / `MAX_EUR` (default 10, from the pod's hourly rate) is hit — final GCS sync, then
  `runpodctl remove pod $RUNPOD_POD_ID`. Ending the process alone does not stop RunPod billing.
  Container restart with a `DONE` marker in GCS for the `RUN_ID` exits instead of resuming.
- Defaults are per game: Airstriker `TOTAL_STEPS=1M`; SF `10M`. Health thresholds that depend on
  win rate are skipped for Airstriker (eval score replaces win rate there).

### Local

- `make install` (uv), `make test`, `make smoke` = Airstriker, 2 envs, ~2k steps, `nice -n 19`, under 2 min.
- One heavy local job at a time (docker build is one).

### Tests (pytest, Airstriker only, stub classes not mocks)

- wrapper obs shape/dtype; frame skip; delta mode never wraps
- reward function on stubbed `info` sequences (damage, round win/loss)
- each health rule on stubbed logger values (healthy / warn / critical)
- smoke: `train` for a few hundred steps writes checkpoint + health.json

### Launch & auth (Juan's decision)

- `RL-sega-rl` is its own GitHub repo (`juan-garassino/sega-rl`, **public** — Claude's
  call: no ROM or secrets in it, and a public GHCR image needs no pull auth on RunPod).
- Workflows copied from `007-deep-sculpt/.github/workflows/`: `ci.yml` (tests), `build-push.yml`
  (GHCR image — the laptop never builds the CUDA image), `deploy-runpod.yml` (WIF token → `runpodctl`
  pod create, inputs `GAME`, `MODE`, `TOTAL_STEPS`, `MAX_HOURS`, `MAX_EUR`), `refresh-token.yml`
  (every 50 min, matches pods by image name `sega-rl`).
- Setup: add the repo to the `garassino-op` WIF provider's trusted repos; repo secrets
  `RUNPOD_API_KEY`, `GCP_WIF_PROVIDER`, `GCP_RUNTIME_SA`.

### Delivered today (Juan's decision)

Tests + local Airstriker smoke + repo + CI image + **one paid Street Fighter run on RunPod**
(`MAX_HOURS=6`, `MAX_EUR=10`). Sonic is runnable from the same image with `GAME=sonic` once its
adapter tests pass; its first paid run is a separate go/no-go.

### Sonic adapter

- Integration `SonicTheHedgehog-Genesis-v0`, default state `GreenHillZone.Act1` (`SEGARL_STATE` overrides).
- Actions: `Discrete(7)` button combos as in the OpenAI Retro Contest baseline —
  `[], [LEFT], [RIGHT], [LEFT,DOWN], [RIGHT,DOWN], [DOWN], [B]`.
- Reward: Δ of furthest x reached (`info['x']`), scaled by 1/100, + level-complete bonus; episode cap 4500 agent steps.
- Health rules shared; "win rate" for Sonic = fraction of eval episodes reaching the level end, plus mean max-x.

## Left to the builder

- Exact Optuna trial budget defaults; MLflow artifact vs metric split; mp4 encoder choice.
- Concrete RunPod pod type at launch time (cheapest ≥16 vCPU offer that day).

## Docs to update

- `006-research-prototypes/CLAUDE.md` → "Reinforcement Learning" table + `ls RL-*` example: rows for `RL-sega-rl`,
  `RL-atari-rl`, `RL-gameboy-rl`; drop `RL-sonic-rl`/`RL-sega-atari-rl`/`RL-pokemon-*`; `DATA-pokemon-api` under Data.
- `006-research-prototypes/DOCS.md` → same renames.
- New `RL-sega-rl/README.md` (setup, ROM import, local smoke, RunPod launch, health) and `CLAUDE.md` (module map).

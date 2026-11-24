.PHONY: help install test smoke import-roms launch status clean

GCS_ROMS := gs://garassino-ml-artifacts/sega-rl/roms
ROM_sf := roms/Street Fighter II' - Special Champion Edition (USA).md
ROM_sonic := roms/Sonic The Hedgehog (Japan, Korea).md
GAME ?=

help:  ## Show targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install:  ## uv sync with dev deps (CPU torch)
	uv sync --extra dev

test:  ## Run tests (ROM tests skip when ROMs aren't imported)
	nice -n 10 uv run pytest -q

smoke:  ## 2-env Airstriker run, ~2k steps, niced
	nice -n 19 uv run sega-rl smoke

import-roms:  ## Import ROMs from roms/ into stable-retro (local runs)
	uv run sega-rl import-roms --dir roms

launch:  ## Upload one ROM to private GCS and start a pod: make launch GAME=sf|sonic [ARGS="-f n_envs=32"]
	@if [ -z "$(GAME)" ]; then echo "Usage: make launch GAME=sf|sonic"; exit 1; fi
	@if gcloud storage buckets get-iam-policy gs://garassino-ml-artifacts --format=json | grep -q allUsers; then \
		echo "bucket is public — refusing to upload ROMs"; exit 1; fi
	gcloud storage cp "$(ROM_$(GAME))" $(GCS_ROMS)/
	gh workflow run deploy-runpod.yml -f game=$(GAME) $(ARGS)

status:  ## Live run state from MLflow: make status RUN=sf-20261124-180000
	@if [ -z "$(RUN)" ]; then echo "Usage: make status RUN=<run id>"; exit 1; fi
	@uv run sega-rl status $(RUN) 2>/dev/null

clean:  ## Remove caches and local runs
	rm -rf .pytest_cache runs **/__pycache__

.PHONY: help install test smoke import-roms launch status videos play clean

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
	@policy=$$(gcloud storage buckets get-iam-policy gs://garassino-ml-artifacts --format=json) || { \
		echo "cannot read bucket policy — refusing to upload ROMs"; exit 1; }; \
	if echo "$$policy" | grep -qE 'allUsers|allAuthenticatedUsers'; then \
		echo "bucket is public — refusing to upload ROMs"; exit 1; fi
	gcloud storage cp "$(ROM_$(GAME))" $(GCS_ROMS)/
	gh workflow run deploy-runpod.yml -f game=$(GAME) $(ARGS)

status:  ## Live run state from MLflow: make status RUN=sf-20261124-180000
	@if [ -z "$(RUN)" ]; then echo "Usage: make status RUN=<run id>"; exit 1; fi
	@uv run sega-rl status $(RUN) 2>/dev/null

videos:  ## Download eval replays + best_model.zip to ~/Videos/rl-series/<run>: make videos RUN=sf-20261124-180000
	@if [ -z "$(RUN)" ]; then echo "Usage: make videos RUN=<run id>"; exit 1; fi
	@uv run sega-rl videos $(RUN)

play:  ## Replay a run's best model locally, full episode → mp4: make play RUN=sf-... GAME=sf
	@if [ -z "$(RUN)" ] || [ -z "$(GAME)" ]; then echo "Usage: make play RUN=<run id> GAME=sf|sonic"; exit 1; fi
	nice -n 19 uv run sega-rl eval --game $(GAME) --model $(HOME)/Videos/rl-series/$(RUN)/best_model.zip \
		--episodes 1 --video $(HOME)/Videos/rl-series/$(RUN)/local_play.mp4

clean:  ## Remove caches and local runs
	rm -rf .pytest_cache runs **/__pycache__

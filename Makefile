.PHONY: help install test smoke import-roms clean

help:  ## Show targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install:  ## uv sync with dev deps (CPU torch)
	uv sync --extra dev

test:  ## Run tests (ROM tests skip when ROMs aren't imported)
	nice -n 10 uv run pytest -q

smoke:  ## 2-env Airstriker run, ~2k steps, niced
	nice -n 19 uv run sega-rl smoke

import-roms:  ## Import ROMs from roms/ into stable-retro
	uv run python -m stable_retro.import roms/

clean:  ## Remove caches and local runs
	rm -rf .pytest_cache runs **/__pycache__

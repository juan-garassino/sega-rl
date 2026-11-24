"""Guard: ROMs must never be committed (the repo is public). They live only in gitignored roms/ and private GCS."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROM_PATTERN = re.compile(r"(^|/)roms/|\.(bin|gen|smd|sfc|nes|gb|gbc|gba|a26|zip|7z)$|\.md$", re.IGNORECASE)
ALLOWED_MD = re.compile(r"(^|/)(README|CLAUDE)\.md$|^docs/|^runpod/README\.md$")


def test_no_roms_tracked():
    repo = Path(__file__).resolve().parents[1]
    files = subprocess.run(["git", "ls-files"], cwd=repo, capture_output=True, text=True, check=True).stdout.split()
    offenders = [f for f in files if ROM_PATTERN.search(f) and not ALLOWED_MD.search(f)]
    assert offenders == [], f"ROM-like files tracked by git: {offenders}"

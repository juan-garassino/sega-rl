"""First time any training game does something worth naming: wins a match against X, finishes level Y.

One file per achievement, created once for all environment processes (they share the directory).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def first(directory: Path | None, key: str, text: str, **extra) -> bool:
    """True if this is the first time (and it is now on disk)."""
    if directory is None:
        return False
    directory.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(directory / f"{key}.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        json.dump({"what": text, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}, f)
    return True

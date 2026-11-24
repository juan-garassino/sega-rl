from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def write_mp4(frames: list[np.ndarray], path: Path, fps: int = 30) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, macro_block_size=16)
    return path

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


SCALE = 3  # consoles draw ~256x224: players open that tiny


def write_mp4(frames: list[np.ndarray], path: Path, fps: int = 30) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, macro_block_size=16) as writer:
        for frame in frames:
            writer.append_data(frame.repeat(SCALE, axis=0).repeat(SCALE, axis=1))
    return path

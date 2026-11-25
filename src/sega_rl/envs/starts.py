"""Where a training game starts: more often where it is weak, and sometimes partway through a level it has reached."""
from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import Callable

import numpy as np


class WeakSpots:
    """Draws opponents / levels, the ones this process loses more often. Everything keeps a share, so nothing is forgotten."""

    FLOOR, MEMORY = 0.3, 0.2

    def __init__(self, starts: list[str]):
        self.starts = starts
        self.rate = {s: 0.5 for s in starts}

    def pick(self, rng: np.random.Generator) -> str:
        w = np.array([self.FLOOR + 1.0 - self.rate[s] for s in self.starts])
        return self.starts[int(rng.choice(len(w), p=w / w.sum()))]

    def result(self, start: str, won: bool) -> None:
        self.rate[start] += self.MEMORY * (float(won) - self.rate[start])


class LevelPoints:
    """Emulator states along each level, saved the first few times a game gets there; all processes share the directory.

    A level is cut into CHUNKS; each keeps at most KEEP states. A state counts only once the game has survived SURVIVE more
    moves, so nobody practises from a moment that is already lost.
    """

    CHUNKS, KEEP, SURVIVE = 20, 2, 8

    def __init__(self, directory: Path, prob: float):
        self.directory, self.prob = Path(directory), prob
        self.furthest = 0  # the furthest chunk this game has entered
        self.pending: list[list] = []  # [moves survived, chunk, state]

    def _files(self, level: str) -> dict[int, list[Path]]:
        out: dict[int, list[Path]] = {}
        for f in self.directory.glob(f"{level}.c*.state"):
            out.setdefault(int(f.name.split(".c")[-1].split("-")[0]), []).append(f)
        return out

    def pick(self, rng: np.random.Generator, level: str) -> Path | None:
        """A state along the level, or None for the normal start. Half the practice is at the furthest point reached."""
        self.furthest, self.pending = 0, []
        if rng.random() >= self.prob or not (files := self._files(level)):
            return None
        chunks = sorted(files)
        chunk = chunks[-1] if rng.random() < 0.5 else chunks[int(rng.integers(len(chunks)))]
        self.furthest = chunk
        return files[chunk][int(rng.integers(len(files[chunk])))]

    def reached(self, level: str, fraction: float, get_state: Callable[[], bytes]) -> None:
        """Call after every move the game survived."""
        for p in self.pending:
            p[0] += 1
        while self.pending and self.pending[0][0] >= self.SURVIVE:
            _, chunk, state = self.pending.pop(0)
            self._save(level, chunk, state)
        chunk = min(self.CHUNKS - 1, int(fraction * self.CHUNKS))
        if chunk > self.furthest:
            self.furthest = chunk
            if len(self._files(level).get(chunk, [])) < self.KEEP:
                self.pending.append([0, chunk, get_state()])

    def _save(self, level: str, chunk: int, state: bytes) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.directory / f".{level}.{os.getpid()}.tmp"
        with gzip.open(tmp, "wb") as f:
            f.write(state)
        tmp.replace(self.directory / f"{level}.c{chunk:02d}-{os.getpid()}.state")  # atomic

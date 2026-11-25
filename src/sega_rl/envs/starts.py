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
    """Emulator states all over each level, one per place, saved the first time a game gets there; processes share them.

    A state counts only once the game has survived SURVIVE more moves, so nobody practises from a moment already lost.
    """

    SURVIVE, FURTHEST = 8, 0.3

    def __init__(self, directory: Path, prob: float):
        self.directory, self.prob = Path(directory), prob
        self.seen: set[tuple[int, int]] = set()  # places of this game
        self.pending: list[list] = []  # [moves survived, place, state]
        self.visits: dict[str, int] = {}

    def _file(self, level: str, place: tuple[int, int]) -> Path:
        return self.directory / f"{level}.x{place[0]:03d}y{place[1]:03d}.state"

    def pick(self, rng: np.random.Generator, level: str) -> Path | None:
        """A state in the level, or None for the normal start. Mostly places practised less, sometimes the furthest."""
        self.seen, self.pending = set(), []
        if rng.random() >= self.prob or not (files := sorted(self.directory.glob(f"{level}.x*.state"))):
            return None
        if rng.random() < self.FURTHEST:
            f = files[-1]  # the name starts with x
        else:
            w = np.array([1.0 / np.sqrt(1 + self.visits.get(f.name, 0)) for f in files])
            f = files[int(rng.choice(len(files), p=w / w.sum()))]
        self.visits[f.name] = self.visits.get(f.name, 0) + 1
        return f

    def reached(self, level: str, place: tuple[int, int], get_state: Callable[[], bytes]) -> None:
        """Call after every move the game survived."""
        for p in self.pending:
            p[0] += 1
        while self.pending and self.pending[0][0] >= self.SURVIVE:
            _, there, state = self.pending.pop(0)
            self._save(self._file(level, there), state)
        if place in self.seen:
            return
        first = not self.seen  # where the game started
        self.seen.add(place)
        if not first and not self._file(level, place).exists():
            self.pending.append([0, place, get_state()])

    def _save(self, target: Path, state: bytes) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.directory / f".{target.stem}.{os.getpid()}.tmp"
        with gzip.open(tmp, "wb") as f:
            f.write(state)
        tmp.replace(target)  # atomic

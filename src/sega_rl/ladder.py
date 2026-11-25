"""Make Street Fighter start states for every opponent (pod only): climb the ladder, save each fight's start.

Cheats on purpose: Ryu's bar stays full and the opponent's at 1, so any hit wins. Training never does this.
The states hold emulator memory, so like the ROM they never go into git, an image or a public place.
"""
from __future__ import annotations

import gzip
import logging
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from sega_rl.config import Settings
from sega_rl.monitoring.video import write_mp4

logger = logging.getLogger(__name__)
FULL = 176
SETTLE_FRAMES = 30  # both bars full this long = a round has really started
RECORD_EVERY = 60  # one frame per game second
MASH = ("A", "B", "C", "X", "Y", "Z")  # never START: it pauses the fight


@dataclass
class Fight:
    state: bytes
    frame: np.ndarray


def climb(game, max_fights: int = 12, max_frames: int = 120_000, seed: int = 0,
          frames: list | None = None) -> list[Fight]:
    rng = np.random.default_rng(seed)
    mash = [game.buttons.index(b) for b in MASH if b in game.buttons]
    move = [game.buttons.index(b) for b in ("LEFT", "RIGHT", "RIGHT", "DOWN") if b in game.buttons]
    fights: list[Fight] = []
    waiting, settled = True, 0
    for i in range(max_frames):
        buttons = np.zeros(len(game.buttons), np.int8)
        if i % 8 < 4:  # tap, do not hold: menus and win quotes need fresh presses
            buttons[rng.choice(mash)] = 1
        if move:
            buttons[move[(i // 24) % len(move)]] = 1
        if waiting:
            buttons[:] = 0 if settled else buttons  # a fight that just started: save it untouched
        frame, _, _, _, info = game.step(buttons)
        if frames is not None and i % RECORD_EVERY == 0:
            frames.append(np.array(frame))
        full = info["health"] == FULL and info["enemy_health"] == FULL
        if waiting:
            settled = settled + 1 if full and info["matches_won"] == 0 else 0
            if settled >= SETTLE_FRAMES:
                fights.append(Fight(game.em.get_state(), np.array(frame)))
                logger.info("fight %d at frame %d", len(fights), i)
                if len(fights) >= max_fights:
                    break
                waiting, settled = False, 0
        elif info["matches_won"] >= 2:
            waiting = True
        elif info["health"] > 0 and info["enemy_health"] > 0:
            game.set_value("health", FULL)  # every frame: he can be hit while the opponent is at 1
            game.set_value("enemy_health", 1)
    return fights


def save_states(fights: list[Fight], out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for n, fight in enumerate(fights, start=1):
        path = out / f"Ladder.{n:02d}.state"
        with gzip.open(path, "wb") as f:
            f.write(fight.state)
        imageio.imwrite(path.with_suffix(".png"), fight.frame)
        paths.append(path)
    return paths


def run(cfg: Settings) -> int:
    from sega_rl.envs.common import _make_retro
    from sega_rl.envs.street_fighter import DEFAULT_STATE, GAME_ID

    game = _make_retro(GAME_ID, DEFAULT_STATE)
    try:
        game.reset()
        frames: list = []
        fights = climb(game, frames=frames)
    finally:
        game.close()
    paths = save_states(fights, cfg.run_dir / "states")
    write_mp4(frames, cfg.run_dir / "ladder.mp4")
    (cfg.run_dir / "DONE").write_text("done")
    print(f"{len(paths)} fight(s) saved")
    return 0 if paths else 1

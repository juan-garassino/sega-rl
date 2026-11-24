from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from sega_rl.config import Settings

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sega-rl", description="PPO on Sega Genesis games (stable-retro).")
    sub = p.add_subparsers(dest="cmd", required=True)

    def with_overrides(sp: argparse.ArgumentParser) -> argparse.ArgumentParser:
        sp.add_argument("--game", choices=["sf", "sonic", "airstriker"])
        sp.add_argument("--steps", type=int, help="total timesteps (default: per game)")
        sp.add_argument("--n-envs", type=int)
        sp.add_argument("--run-id")
        return sp

    with_overrides(sub.add_parser("train", help="train or resume a run"))
    tune = with_overrides(sub.add_parser("tune", help="Optuna search, writes best_params.json"))
    tune.add_argument("--trials", type=int, default=10)
    tune.add_argument("--trial-steps", type=int, default=100_000)
    ev = with_overrides(sub.add_parser("eval", help="evaluate a saved model and record a video"))
    ev.add_argument("--model", type=Path, required=True)
    ev.add_argument("--episodes", type=int, default=5)
    ev.add_argument("--video", type=Path)
    sub.add_parser("smoke", help="2-env Airstriker run, ~2k steps, in a temp dir")
    st = sub.add_parser("status", help="live run state from MLflow (health, eval, fps)")
    st.add_argument("run_name", help="run id = MLflow run name, e.g. sf-20261124-180000")
    vid = sub.add_parser("videos", help="download a run's eval replay videos from MLflow")
    vid.add_argument("run_name")
    vid.add_argument("--dest", type=Path, default=Path.home() / "Videos" / "rl-series")
    sub.add_parser("check-rom", help="exit 1 unless SEGARL_GAME's ROM is imported (right dump)")
    imp = sub.add_parser("import-roms", help="import ROMs into stable-retro")
    imp.add_argument("--dir", type=Path, default=Path("roms"))
    return p


def settings_from_args(args: argparse.Namespace) -> Settings:
    overrides = {"game": args.game, "total_steps": args.steps, "n_envs": args.n_envs, "run_id": args.run_id}
    return Settings(**{k: v for k, v in overrides.items() if v is not None})


def _cmd_train(args: argparse.Namespace) -> int:
    from sega_rl import train

    status = train.run(settings_from_args(args))
    print(status)
    return 0 if status != "critical" else 2


def _cmd_tune(args: argparse.Namespace) -> int:
    from sega_rl import tune

    print(tune.run(settings_from_args(args), n_trials=args.trials, trial_steps=args.trial_steps))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from stable_baselines3 import PPO

    from sega_rl.evaluate import run_eval

    model = PPO.load(args.model)
    print(run_eval(model, settings_from_args(args), args.episodes, video_path=args.video))
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    from sega_rl import train

    with tempfile.TemporaryDirectory() as tmp:
        cfg = Settings(game="airstriker", n_envs=2, total_steps=2048, eval_freq=1024, ckpt_freq=1024,
                       eval_episodes=1, eval_max_steps=300, runs_dir=Path(tmp), run_id="smoke")
        status = train.run(cfg)
    print(status)
    return 0 if status == "done" else 2


def _cmd_status(args: argparse.Namespace) -> int:
    from sega_rl.status import fetch

    print(fetch(args.run_name))
    return 0


def rom_available(game: str) -> bool:
    import stable_retro

    from sega_rl.envs import airstriker, sonic, street_fighter

    game_id = {"sf": street_fighter.GAME_ID, "sonic": sonic.GAME_ID, "airstriker": airstriker.GAME_ID}[game]
    try:
        stable_retro.data.get_romfile_path(game_id)
        return True
    except FileNotFoundError:
        return False


def _cmd_check_rom(args: argparse.Namespace) -> int:
    game = Settings().game
    ok = rom_available(game)
    print(f"{game}: ROM {'ok' if ok else 'MISSING (not imported or wrong dump)'}")
    return 0 if ok else 1


def _cmd_videos(args: argparse.Namespace) -> int:
    from sega_rl.status import fetch_videos

    new = fetch_videos(args.run_name, args.dest)
    print(f"{len(new)} new video(s) in {args.dest / args.run_name}")
    for p in new:
        print(f"  {p}")
    return 0


def _cmd_import_roms(args: argparse.Namespace) -> int:
    return subprocess.call([sys.executable, "-m", "stable_retro.import", str(args.dir)])


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    handlers = {"train": _cmd_train, "tune": _cmd_tune, "eval": _cmd_eval, "smoke": _cmd_smoke, "status": _cmd_status, "check-rom": _cmd_check_rom, "videos": _cmd_videos,
                "import-roms": _cmd_import_roms}
    return handlers[args.cmd](args)

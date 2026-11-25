from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Game = Literal["sf", "sonic", "airstriker"]

_TOTAL_STEPS = {"airstriker": 1_000_000, "sf": 10_000_000, "sonic": 10_000_000}
_MAX_EUR = {"airstriker": 1.0, "sf": 8.0, "sonic": 7.0}
_EVAL_EPISODES = {"airstriker": 5, "sf": 15, "sonic": 5}
_LEARNING_RATE = {"airstriker": 2.5e-4, "sf": 1e-4, "sonic": 2.5e-4}  # sf: clip_fraction ran high at 2.5e-4


def default_total_steps(game: str) -> int:
    return _TOTAL_STEPS[game]


def default_max_eur(game: str) -> float:
    return _MAX_EUR[game]


def default_learning_rate(game: str) -> float:
    return _LEARNING_RATE[game]


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SEGARL_", extra="ignore")

    game: Game = "airstriker"  # SEGARL_GAME=sf
    state: str | None = None  # SEGARL_STATE=GreenHillZone.Act2 (None = integration default)
    run_id: str = Field(default_factory=_utc_run_id)  # SEGARL_RUN_ID=20261124-150000
    total_steps: int | None = None  # SEGARL_TOTAL_STEPS=10000000 (None = per-game default)
    n_envs: int = 16  # SEGARL_N_ENVS=16
    obs_mode: Literal["frame", "delta"] = "frame"  # SEGARL_OBS_MODE=delta
    reward_mode: Literal["shaped", "score"] = "shaped"  # SEGARL_REWARD_MODE=score
    sf_actions: Literal["fighter", "discrete", "multibinary"] = "fighter"  # SEGARL_SF_ACTIONS=discrete (no specials)
    sf_dealt_weight: float = 1.0  # SEGARL_SF_DEALT_WEIGHT=1.0
    sf_taken_weight: float = 0.5  # SEGARL_SF_TAKEN_WEIGHT=0.5 (<1 favours engaging over fleeing)
    sf_win_health_bonus: float = 1.0  # SEGARL_SF_WIN_HEALTH_BONUS=1.0 (round won: + this × health left)
    sonic_levels: Literal["single", "multi"] = "multi"  # SEGARL_SONIC_LEVELS=single (multi: 13 acts, 4 held out)
    sonic_actions: Literal["jump9", "retro7"] = "jump9"  # SEGARL_SONIC_ACTIONS=retro7 (sonic-v2, sonic-multi-1 models)
    sonic_level_episodes: int = 2  # SEGARL_SONIC_LEVEL_EPISODES=2 (multi: games per level and eval)
    sonic_stall_secs: float = 30.0  # SEGARL_SONIC_STALL_SECS=30 (end a Sonic episode after this long without new ground)
    learning_rate: float | None = None  # SEGARL_LEARNING_RATE=1e-4 (None = per-game default)
    frame_skip: int = 4  # SEGARL_FRAME_SKIP=4
    frame_stack: int = 4  # SEGARL_FRAME_STACK=4
    noop_max: int = 30  # SEGARL_NOOP_MAX=30 (random no-op frames at reset; 0 = always the same start)
    seed: int = 0  # SEGARL_SEED=0
    device: str = "auto"  # SEGARL_DEVICE=mps (Apple GPU; auto = NVIDIA or CPU)
    params_file: Path | None = None  # SEGARL_PARAMS_FILE=runs/x/best_params.json
    eval_freq: int = 250_000  # SEGARL_EVAL_FREQ=250000
    eval_sampled: bool = True  # SEGARL_EVAL_SAMPLED=false (also check sampled play, as training does: eval_sampled/*)
    eval_episodes: int | None = None  # SEGARL_EVAL_EPISODES=5 (None = per game: sf 15 so an eval has >= 30 rounds)
    eval_max_steps: int = 4500  # SEGARL_EVAL_MAX_STEPS=4500 (agent steps per eval episode)
    ckpt_freq: int = 250_000  # SEGARL_CKPT_FREQ=250000
    keep_ckpts: int = 3  # SEGARL_KEEP_CKPTS=3
    runs_dir: Path = Path("runs")  # SEGARL_RUNS_DIR=/workspace/runs
    stop_on_critical: bool = True  # SEGARL_STOP_ON_CRITICAL=false
    max_hours: float = 6.0  # SEGARL_MAX_HOURS=6
    max_eur: float | None = None  # SEGARL_MAX_EUR=8 (None = per-game default)
    usd_per_eur: float = 1.05  # SEGARL_USD_PER_EUR=1.05
    cost_per_hr_usd: float | None = None  # SEGARL_COST_PER_HR_USD=0.44 (set by the RunPod entrypoint)

    kl_warn: float = 0.05  # SEGARL_KL_WARN=0.05
    kl_crit: float = 0.2  # SEGARL_KL_CRIT=0.2
    clipfrac_warn: float = 0.3  # SEGARL_CLIPFRAC_WARN=0.3
    entropy_collapse_frac: float = 0.2  # SEGARL_ENTROPY_COLLAPSE_FRAC=0.2
    ev_warn_after: float = 0.1  # SEGARL_EV_WARN_AFTER=0.1 (fraction of training)
    ev_crit: float = -0.5  # SEGARL_EV_CRIT=-0.5
    plateau_frac: float = 0.2  # SEGARL_PLATEAU_FRAC=0.2
    fps_warn_frac: float = 0.5  # SEGARL_FPS_WARN_FRAC=0.5
    fps_zero_secs: float = 300.0  # SEGARL_FPS_ZERO_SECS=300
    button_spam_frac: float = 0.9  # SEGARL_BUTTON_SPAM_FRAC=0.9

    @property
    def steps(self) -> int:
        return self.total_steps if self.total_steps is not None else default_total_steps(self.game)

    @property
    def n_eval_episodes(self) -> int:
        return self.eval_episodes if self.eval_episodes is not None else _EVAL_EPISODES[self.game]

    @property
    def budget_eur(self) -> float:
        return self.max_eur if self.max_eur is not None else default_max_eur(self.game)

    @property
    def hours_budget(self) -> float:
        """Wall-clock cap: max_hours, tightened by the EUR cap when the pod's hourly cost is known."""
        if not self.cost_per_hr_usd:
            return self.max_hours
        return min(self.max_hours, self.budget_eur * self.usd_per_eur / self.cost_per_hr_usd)

    @property
    def run_dir(self) -> Path:
        return self.runs_dir / self.run_id


settings = Settings()

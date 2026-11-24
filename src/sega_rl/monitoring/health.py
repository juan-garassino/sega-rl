from __future__ import annotations

import json
import logging
import math
import os
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from sega_rl.config import Settings

logger = logging.getLogger(__name__)

Status = Literal["HEALTHY", "WARN", "CRITICAL"]
_RANK = {"HEALTHY": 0, "WARN": 1, "CRITICAL": 2}


@dataclass
class HealthSnapshot:
    step: int
    frac_done: float
    approx_kl: float | None
    clip_fraction: float | None
    entropy: float | None
    explained_variance: float | None
    ep_rew_mean: float | None
    fps: float | None
    button_freq: list[float] | None
    losses_finite: bool
    wall_time: float


@dataclass
class HealthReport:
    status: Status
    reasons: list[str]
    last: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=float)


class HealthMonitor:
    """Classifies PPO training health from a stream of per-update snapshots (thresholds from Settings)."""

    def __init__(self, cfg: Settings, check_button_spam: bool = True):
        self.cfg = cfg
        self.check_button_spam = check_button_spam
        self.history: list[HealthSnapshot] = []
        self.initial_entropy: float | None = None
        self.kl_crit_streak = 0
        self.ev_crit_streak = 0
        self.fps_zero_since: float | None = None
        self.critical: list[str] = []

    def mark_critical(self, reason: str, last: dict | None = None) -> HealthReport:
        if reason not in self.critical:
            self.critical.append(reason)
        return HealthReport("CRITICAL", list(self.critical), last or {})

    def update(self, s: HealthSnapshot) -> HealthReport:
        c = self.cfg
        self.history.append(s)
        warn: list[str] = []
        crit: list[str] = []

        if not s.losses_finite:
            crit.append("nan")

        if s.approx_kl is not None:
            self.kl_crit_streak = self.kl_crit_streak + 1 if s.approx_kl > c.kl_crit else 0
            if self.kl_crit_streak >= 3:
                crit.append("approx_kl")
            elif s.approx_kl > c.kl_warn:
                warn.append("approx_kl")

        if s.clip_fraction is not None and s.clip_fraction > c.clipfrac_warn:
            warn.append("clip_fraction")

        if s.entropy is not None:
            if self.initial_entropy is None:
                self.initial_entropy = s.entropy
            elif s.frac_done < 0.25 and s.entropy < c.entropy_collapse_frac * self.initial_entropy:
                warn.append("entropy_collapse")

        if s.explained_variance is not None and s.frac_done > c.ev_warn_after:
            self.ev_crit_streak = self.ev_crit_streak + 1 if s.explained_variance < c.ev_crit else 0
            if self.ev_crit_streak >= 10:
                crit.append("explained_variance")
            elif s.explained_variance < 0:
                warn.append("explained_variance")

        if self._plateau(s):
            warn.append("reward_plateau")

        warn += self._fps_rules(s, crit)

        if self.check_button_spam and s.button_freq and max(s.button_freq) > c.button_spam_frac:
            warn.append("button_spam")

        for r in crit:
            if r not in self.critical:
                self.critical.append(r)
        last = asdict(s)
        if self.critical:
            return HealthReport("CRITICAL", list(self.critical) + warn, last)
        return HealthReport("WARN" if warn else "HEALTHY", warn, last)

    def _plateau(self, s: HealthSnapshot) -> bool:
        if s.ep_rew_mean is None or s.frac_done < 2 * self.cfg.plateau_frac or s.frac_done <= 0:
            return False
        window_start = s.step - self.cfg.plateau_frac * (s.step / s.frac_done)
        before = [h.ep_rew_mean for h in self.history if h.step < window_start and h.ep_rew_mean is not None]
        inside = [h.ep_rew_mean for h in self.history if h.step >= window_start and h.ep_rew_mean is not None]
        return bool(before and inside) and max(inside) <= max(before)

    def _fps_rules(self, s: HealthSnapshot, crit: list[str]) -> list[str]:
        if s.fps is None:
            return []
        if s.fps <= 0:
            self.fps_zero_since = self.fps_zero_since if self.fps_zero_since is not None else s.wall_time
            if s.wall_time - self.fps_zero_since >= self.cfg.fps_zero_secs:
                crit.append("fps_zero")
            return []
        self.fps_zero_since = None
        t0 = self.history[0].wall_time
        if s.wall_time - t0 <= 300:
            return []
        baseline = [h.fps for h in self.history if h.fps and h.wall_time - t0 <= 300]
        if baseline and s.fps < self.cfg.fps_warn_frac * statistics.median(baseline):
            return ["fps_drop"]
        return []


def _finite(value) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.isfinite(np.asarray(value, dtype=np.float64)).all())


def _exit_on_stall(report: HealthReport) -> None:
    logger.critical("training stalled for too long, exiting: %s", report.reasons)
    os._exit(3)


class HealthCallback(BaseCallback):
    """Feeds HealthMonitor after every rollout, writes health.json atomically, stops on CRITICAL.

    A watchdog thread catches hangs (a stuck emulator never ends its rollout, so per-rollout checks can't).
    """

    def __init__(
        self,
        monitor: HealthMonitor,
        out_path: Path,
        on_critical: Callable[[HealthReport], None],
        stall_secs: float | None = 300.0,
        on_stall: Callable[[HealthReport], None] = _exit_on_stall,
    ):
        super().__init__()
        self.monitor = monitor
        self.out_path = Path(out_path)
        self.on_critical = on_critical
        self.stall_secs = stall_secs
        self.on_stall = on_stall
        self._stop = False
        self._buttons_sum: np.ndarray | None = None
        self._buttons_n = 0
        self._last_progress = time.monotonic()
        self._last_rollout = (time.monotonic(), 0)
        self._watchdog_stop = threading.Event()
        self._watchdog: threading.Thread | None = None
        self.last_report: HealthReport | None = None

    def write(self, report: HealthReport) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.out_path.with_name(self.out_path.name + ".tmp")
        tmp.write_text(report.to_json())
        os.replace(tmp, self.out_path)
        self.last_report = report

    def _on_training_start(self) -> None:
        self._last_progress = time.monotonic()
        self._last_rollout = (time.monotonic(), self.model.num_timesteps)
        if self.stall_secs is not None:
            self._watchdog = threading.Thread(target=self._watch, daemon=True)
            self._watchdog.start()

    def _watch(self) -> None:
        interval = min(5.0, self.stall_secs / 4)
        while not self._watchdog_stop.wait(interval):
            if time.monotonic() - self._last_progress > self.stall_secs:
                report = self.monitor.mark_critical("stalled")
                self.write(report)
                self.on_stall(report)
                return

    def _on_step(self) -> bool:
        self._last_progress = time.monotonic()
        for info in self.locals.get("infos", []):
            b = info.get("buttons")
            if b is None:
                continue
            b = np.asarray(b, dtype=np.float64)
            self._buttons_sum = b if self._buttons_sum is None else self._buttons_sum + b
            self._buttons_n += 1
        return not self._stop

    def _on_rollout_end(self) -> None:
        values = self.logger.name_to_value
        now = time.monotonic()
        t_prev, steps_prev = self._last_rollout
        steps = self.model.num_timesteps
        fps = (steps - steps_prev) / (now - t_prev) if now > t_prev else None
        self._last_rollout = (now, steps)
        loss = values.get("train/loss")
        finite = (loss is None or math.isfinite(loss)) and all(
            _finite(v) for group in self.model.get_parameters().values() for v in group.values()
        )
        rewards = [ep["r"] for ep in self.model.ep_info_buffer]
        entropy_loss = values.get("train/entropy_loss")
        freq = (self._buttons_sum / self._buttons_n).tolist() if self._buttons_n else None
        self._buttons_sum, self._buttons_n = None, 0
        snap = HealthSnapshot(
            step=steps,
            frac_done=steps / self.monitor.cfg.steps,
            approx_kl=values.get("train/approx_kl"),
            clip_fraction=values.get("train/clip_fraction"),
            entropy=-entropy_loss if entropy_loss is not None else None,
            explained_variance=values.get("train/explained_variance"),
            ep_rew_mean=float(np.mean(rewards)) if rewards else None,
            fps=fps,
            button_freq=freq,
            losses_finite=finite,
            wall_time=now,
        )
        report = self.monitor.update(snap)
        self.write(report)
        self.logger.record("health/status", _RANK[report.status])
        if report.status != "HEALTHY":
            logger.warning("health %s: %s", report.status, ", ".join(report.reasons))
        if report.status == "CRITICAL" and not self._stop:
            self.on_critical(report)
            self._stop = self.monitor.cfg.stop_on_critical

    def _on_training_end(self) -> None:
        self._watchdog_stop.set()

from __future__ import annotations

import json
import math
import threading

import numpy as np

from sega_rl.config import Settings
from sega_rl.monitoring.health import HealthCallback, HealthMonitor, HealthSnapshot

TOTAL = 1_000_000


def snap(step, **kw):
    base = dict(step=step, frac_done=step / TOTAL, approx_kl=0.01, clip_fraction=0.1, entropy=8.0,
                explained_variance=0.5, ep_rew_mean=float(step) / 1000, fps=1000.0, button_freq=[0.5] * 12,
                losses_finite=True, wall_time=step / 1000)
    base.update(kw)
    return HealthSnapshot(**base)


def run(monitor, snaps):
    report = None
    for s in snaps:
        report = monitor.update(s)
    return report


def healthy_series(n=50, start=0):
    return [snap(start + (i + 1) * 10_000) for i in range(n)]


def test_healthy_baseline():
    report = run(HealthMonitor(Settings()), healthy_series())
    assert report.status == "HEALTHY", report.reasons


def test_nan_is_critical():
    m = HealthMonitor(Settings())
    report = run(m, healthy_series(3) + [snap(40_000, losses_finite=False)])
    assert report.status == "CRITICAL" and "nan" in report.reasons


def test_kl_warn():
    report = run(HealthMonitor(Settings()), healthy_series(3) + [snap(40_000, approx_kl=0.08)])
    assert report.status == "WARN" and "approx_kl" in report.reasons


def test_kl_critical_after_3():
    m = HealthMonitor(Settings())
    report = run(m, healthy_series(3) + [snap(40_000 + i, approx_kl=0.3) for i in range(2)])
    assert report.status == "WARN"
    report = m.update(snap(40_010, approx_kl=0.3))
    assert report.status == "CRITICAL" and "approx_kl" in report.reasons


def test_clipfrac_warn():
    report = run(HealthMonitor(Settings()), healthy_series(3) + [snap(40_000, clip_fraction=0.4)])
    assert "clip_fraction" in report.reasons


def test_entropy_collapse_warn_early_only():
    early = run(HealthMonitor(Settings()), healthy_series(3) + [snap(100_000, entropy=1.0)])
    assert "entropy_collapse" in early.reasons
    late = run(HealthMonitor(Settings()), healthy_series(3) + [snap(500_000, entropy=1.0)])
    assert "entropy_collapse" not in late.reasons


def test_explained_variance_warn_and_critical():
    m = HealthMonitor(Settings())
    assert "explained_variance" not in m.update(snap(50_000, explained_variance=-0.2)).reasons  # before 10%
    report = m.update(snap(200_000, explained_variance=-0.2))
    assert report.status == "WARN" and "explained_variance" in report.reasons
    for i in range(10):
        report = m.update(snap(210_000 + i, explained_variance=-0.8))
    assert report.status == "CRITICAL"


def test_reward_plateau_warn():
    m = HealthMonitor(Settings())
    series = [snap(i * 10_000, ep_rew_mean=min(i, 20) * 1.0) for i in range(1, 60)]
    report = run(m, series)
    assert "reward_plateau" in report.reasons


def test_fps_drop_warn():
    m = HealthMonitor(Settings())
    series = [snap(i * 10_000, wall_time=i * 10.0, fps=1000.0) for i in range(1, 40)]
    run(m, series)
    report = m.update(snap(400_000, wall_time=400.0, fps=300.0))
    assert "fps_drop" in report.reasons


def test_fps_zero_is_critical():
    m = HealthMonitor(Settings())
    run(m, [snap(i * 10_000, wall_time=i * 10.0) for i in range(1, 5)])
    assert m.update(snap(50_000, wall_time=100.0, fps=0.0)).status != "CRITICAL"
    report = m.update(snap(50_000, wall_time=401.0, fps=0.0))
    assert report.status == "CRITICAL" and "fps_zero" in report.reasons


def test_button_spam_warn():
    freq = [0.5] * 12
    freq[0] = 0.95
    report = run(HealthMonitor(Settings()), healthy_series(3) + [snap(40_000, button_freq=freq)])
    assert "button_spam" in report.reasons
    report = run(HealthMonitor(Settings(), check_button_spam=False), healthy_series(3) + [snap(40_000, button_freq=freq)])
    assert "button_spam" not in report.reasons


def test_report_json_roundtrip():
    report = run(HealthMonitor(Settings()), healthy_series(2))
    data = json.loads(report.to_json())
    assert data["status"] == "HEALTHY" and "last" in data


class _StubLogger:
    def __init__(self, values):
        self.name_to_value = values
        self.records = {}

    def record(self, key, value, exclude=None):
        self.records[key] = value


class _StubParam:
    def __init__(self, ok=True):
        self.data = np.array([1.0 if ok else math.nan])


class _StubModel:
    def __init__(self, ok=True):
        self.num_timesteps = 0
        self.ep_info_buffer = [{"r": 1.0}, {"r": 3.0}]
        self.logger = _StubLogger({"train/approx_kl": 0.01, "train/clip_fraction": 0.1,
                                   "train/entropy_loss": -8.0, "train/explained_variance": 0.5,
                                   "train/loss": 0.3})
        self._ok = ok

    def get_env(self):
        return None

    def get_parameters(self):
        return {"policy": {"w": np.array([1.0 if self._ok else math.nan])}}


def test_callback_writes_json_atomically(tmp_path):
    out = tmp_path / "health.json"
    reports = []
    cb = HealthCallback(HealthMonitor(Settings(total_steps=TOTAL)), out, on_critical=reports.append,
                        stall_secs=None)
    model = _StubModel()
    cb.init_callback(model)
    cb.locals = {"infos": [{"buttons": np.ones(12, np.int8)}, {"buttons": np.zeros(12, np.int8)}]}
    cb.on_step()
    model.num_timesteps = 2048
    cb.on_rollout_end()
    data = json.loads(out.read_text())
    assert data["status"] in {"HEALTHY", "WARN"}
    assert data["last"]["ep_rew_mean"] == 2.0
    assert data["last"]["button_freq"][0] == 0.5
    assert not list(tmp_path.glob("*.tmp"))
    assert "health/status" in model.logger.records


def test_callback_stops_on_critical(tmp_path):
    reports = []
    cb = HealthCallback(HealthMonitor(Settings(total_steps=TOTAL)), tmp_path / "h.json",
                        on_critical=reports.append, stall_secs=None)
    model = _StubModel(ok=False)
    cb.init_callback(model)
    cb.locals = {"infos": []}
    model.num_timesteps = 2048
    cb.on_rollout_end()
    assert reports and reports[0].status == "CRITICAL"
    assert cb.on_step() is False


def test_watchdog_fires_on_stall(tmp_path):
    fired = threading.Event()
    cb = HealthCallback(HealthMonitor(Settings(total_steps=TOTAL)), tmp_path / "h.json",
                        on_critical=lambda r: None, stall_secs=0.2, on_stall=lambda r: fired.set())
    cb.init_callback(_StubModel())
    cb.on_training_start({}, {})
    assert fired.wait(2.0)
    data = json.loads((tmp_path / "h.json").read_text())
    assert data["status"] == "CRITICAL" and "stalled" in data["reasons"]
    cb.on_training_end()


def test_heartbeat_keeps_watchdog_quiet(tmp_path):
    import time as _time

    fired = threading.Event()
    cb = HealthCallback(HealthMonitor(Settings(total_steps=TOTAL)), tmp_path / "h.json",
                        on_critical=lambda r: None, stall_secs=0.4, on_stall=lambda r: fired.set())
    cb.init_callback(_StubModel())
    cb.on_training_start({}, {})
    for _ in range(8):
        _time.sleep(0.1)
        cb.heartbeat()
    assert not fired.is_set()
    cb.on_training_end()

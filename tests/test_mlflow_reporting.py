from __future__ import annotations

from sega_rl.config import Settings
from sega_rl.monitoring import mlflow_tracking
from sega_rl.monitoring.health import HealthReport
from sega_rl.status import format_status
from sega_rl.train import HealthReporter


class _Rec:
    def __init__(self):
        self.tags, self.artifacts = {}, []

    def set_tag(self, k, v):
        self.tags[k] = v

    def log_artifact(self, path, artifact_path=None):
        self.artifacts.append((str(path), artifact_path))


def test_health_reporter_tags_on_change_and_throttles_uploads(tmp_path, monkeypatch):
    rec = _Rec()
    monkeypatch.setattr(mlflow_tracking, "set_tag", rec.set_tag)
    monkeypatch.setattr(mlflow_tracking, "log_artifact", rec.log_artifact)
    health_json = tmp_path / "health.json"
    health_json.write_text("{}")
    clock = iter([0.0, 10.0, 700.0])
    reporter = HealthReporter(health_json, upload_every_s=600, clock=lambda: next(clock))

    reporter(HealthReport("HEALTHY", []))
    reporter(HealthReport("WARN", ["approx_kl"]))
    reporter(HealthReport("WARN", ["approx_kl"]))

    assert rec.tags["health_status"] == "WARN"
    assert rec.tags["health_reasons"] == "approx_kl"
    assert len(rec.artifacts) == 2  # t=0 and t=700, not t=10


def test_format_status():
    text = format_status("sf-x", {"health_status": 1.0, "eval_win_rate": 0.25, "rollout_ep_rew_mean": 1.5,
                                   "time_fps": 900.0},
                         {"health_status": "WARN", "health_reasons": "approx_kl", "final_status": None}, step=500_000)
    assert "WARN" in text and "approx_kl" in text and "500,000" in text and "0.25" in text


def test_mlflow_new_helpers_noop_when_disabled():
    assert mlflow_tracking.enabled() is False
    mlflow_tracking.set_tag("a", "b")
    mlflow_tracking.log_artifact(__file__, "x")


class _FakeMlflow:
    def __init__(self, fail_start):
        self.fail_start = fail_start
        self.calls = []

    def set_tracking_uri(self, uri):
        self.calls.append("uri")

    def get_experiment_by_name(self, name):
        return object()

    def set_experiment(self, name):
        if self.fail_start:
            raise RuntimeError("401 not authenticated")

    def start_run(self, run_name):
        self.calls.append("start")

    def log_metrics(self, metrics, step=0):
        self.calls.append("metrics")

    def set_tag(self, k, v):
        self.calls.append("tag")

    def log_params(self, p):
        pass

    def end_run(self):
        self.calls.append("end")


def test_failed_start_disables_all_logging(monkeypatch):
    fake = _FakeMlflow(fail_start=True)
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x")
    monkeypatch.setattr(mlflow_tracking, "_active", False)
    mlflow_tracking.start(Settings())
    mlflow_tracking.log_metrics({"a": 1.0}, step=1)
    mlflow_tracking.set_tag("k", "v")
    assert "metrics" not in fake.calls and "tag" not in fake.calls  # no stray auto-created runs


def test_successful_start_enables_logging(monkeypatch):
    fake = _FakeMlflow(fail_start=False)
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x")
    monkeypatch.setattr(mlflow_tracking, "_active", False)
    mlflow_tracking.start(Settings())
    mlflow_tracking.log_metrics({"a": 1.0}, step=1)
    mlflow_tracking.end("done")
    assert fake.calls.count("metrics") == 1 and "end" in fake.calls
    assert mlflow_tracking._active is False


class _ExpFakeMlflow(_FakeMlflow):
    def __init__(self, fail_artifact=False):
        super().__init__(fail_start=False)
        self.fail_artifact = fail_artifact
        self.created = None

    def get_experiment_by_name(self, name):
        return None

    def create_experiment(self, name, artifact_location=None):
        self.created = (name, artifact_location)

    def log_artifact(self, path, artifact_path=None):
        if self.fail_artifact:
            raise RuntimeError("upload failed")
        self.calls.append("artifact")


def test_start_creates_experiment_with_proxied_artifacts(monkeypatch, tmp_path):
    fake = _ExpFakeMlflow()
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x")
    monkeypatch.setattr(mlflow_tracking, "_active", False)
    probe = tmp_path / "config.json"
    probe.write_text("{}")
    assert mlflow_tracking.start(Settings(), probe=probe) is True
    assert fake.created == ("sega-rl", "mlflow-artifacts:/sega-rl")
    assert "artifact" in fake.calls
    mlflow_tracking.end("done")


def test_start_returns_false_when_probe_upload_fails(monkeypatch, tmp_path):
    fake = _ExpFakeMlflow(fail_artifact=True)
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x")
    monkeypatch.setattr(mlflow_tracking, "_active", False)
    probe = tmp_path / "config.json"
    probe.write_text("{}")
    assert mlflow_tracking.start(Settings(), probe=probe) is False
    assert mlflow_tracking._active is False


class _StubClient:
    def __init__(self, tmp):
        self.tmp = tmp

    def get_experiment_by_name(self, name):
        return type("E", (), {"experiment_id": "7"})()

    def search_runs(self, ids, flt, max_results=1):
        info = type("I", (), {"run_id": "abc"})()
        return [type("R", (), {"info": info})()]

    def download_artifacts(self, run_id, path, dst_path):
        out = self.tmp / "dl" / path
        out.mkdir(parents=True, exist_ok=True)
        (out / "eval_0000250000.mp4").write_bytes(b"x")
        return str(out)


def test_fetch_videos_copies_new_files(tmp_path):
    from sega_rl.status import fetch_videos

    dest = tmp_path / "Videos"
    got = fetch_videos("sf-1", dest, client=_StubClient(tmp_path))
    assert [p.name for p in got] == ["eval_0000250000.mp4"]
    assert (dest / "sf-1" / "eval_0000250000.mp4").exists()
    assert fetch_videos("sf-1", dest, client=_StubClient(tmp_path)) == []  # already downloaded

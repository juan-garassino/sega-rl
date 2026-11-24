"""Entrypoint behaviour with stubbed gsutil / curl / sega-rl on PATH (no cloud, no pod)."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

GSUTIL = r'''#!/usr/bin/env bash
echo "gsutil $*" >> "$CALLS"
case "$*" in
  *stat*DONE*) [ "${STUB_DONE:-0}" = 1 ] && exit 0 || exit 1 ;;
  *cp*roms*) [ "${STUB_ROM_FAIL:-0}" = 1 ] && exit 1 || exit 0 ;;
  *rsync*) [ "${STUB_RSYNC_FAIL:-0}" = 1 ] && exit 1 || exit 0 ;;
esac
exit 0
'''
CURL = r'''#!/usr/bin/env bash
echo "curl $*" >> "$CALLS"
case "$*" in
  *costPerHr*) echo '{"data":{"pod":{"costPerHr":0.5}}}' ;;
  *) echo '{"data":{"podTerminate":null}}' ;;
esac
'''
SEGA_RL = r'''#!/usr/bin/env bash
echo "sega-rl $* cost=${SEGARL_COST_PER_HR_USD:-}" >> "$CALLS"
exit "${STUB_TRAIN_RC:-0}"
'''


def _trainer_ran(calls: str) -> bool:
    return any(line.startswith("sega-rl ") for line in calls.splitlines())


def _run(tmp_path: Path, **env_overrides) -> tuple[int, str, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in {"gsutil": GSUTIL, "curl": CURL, "sega-rl": SEGA_RL}.items():
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    calls = tmp_path / "calls.log"
    calls.touch()
    ws = tmp_path / "ws"
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}", "CALLS": str(calls), "WORKSPACE_DIR": str(ws),
        "APP_DIR": str(REPO), "RUN_ID": "t", "GCS_BUCKET": "bucket", "MODE": "train",
        "SEGARL_GAME": "airstriker", "RUNPOD_POD_ID": "pod1", "RUNPOD_API_KEY": "k", "SYNC_INTERVAL": "3600",
        "HOME": str(tmp_path),
    }
    env.update(env_overrides)
    proc = subprocess.run(["bash", str(REPO / "runpod" / "entrypoint.sh")], env=env, capture_output=True,
                          text=True, timeout=60)
    return proc.returncode, calls.read_text(), ws


def test_entrypoint_done_marker_exits(tmp_path):
    rc, calls, _ = _run(tmp_path, STUB_DONE="1")
    assert rc == 0
    assert "podTerminate" in calls
    assert not _trainer_ran(calls)


def test_entrypoint_rom_missing_terminates(tmp_path):
    rc, calls, ws = _run(tmp_path, SEGARL_GAME="sf", STUB_ROM_FAIL="1")
    assert rc != 0
    assert "podTerminate" in calls
    assert not _trainer_ran(calls)
    health = json.loads((ws / "runs" / "t" / "health.json").read_text())
    assert health["status"] == "CRITICAL" and health["reasons"] == ["rom"]


def test_sync_failure_does_not_stop_training(tmp_path):
    rc, calls, _ = _run(tmp_path, STUB_RSYNC_FAIL="1")
    assert rc == 0
    assert "sega-rl train" in calls
    assert "podTerminate" in calls


def test_cost_exported_to_trainer(tmp_path):
    _, calls, _ = _run(tmp_path)
    assert "sega-rl train cost=0.5" in calls


def test_trainer_failure_still_syncs_and_terminates(tmp_path):
    rc, calls, _ = _run(tmp_path, STUB_TRAIN_RC="3")
    assert rc == 3
    lines = calls.splitlines()
    train_idx = next(i for i, line in enumerate(lines) if line.startswith("sega-rl "))
    after = "\n".join(lines[train_idx:])
    assert "rsync" in after and "podTerminate" in after


def test_local_run_without_pod_id_skips_terminate(tmp_path):
    rc, calls, _ = _run(tmp_path, RUNPOD_POD_ID="")
    assert rc == 0
    assert "podTerminate" not in calls

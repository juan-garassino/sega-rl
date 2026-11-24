#!/usr/bin/env bash
# RunPod entrypoint for sega-rl.   MODE=train|smoke (default train)
#
# Lifecycle: exit if the run is already DONE -> pull ROMs, delete them from GCS, import + verify ->
# pull the run dir (resume) -> export pod cost -> periodic GCS sync -> sega-rl $MODE.
# The EXIT trap is installed first, so on ANY exit (finish, CRITICAL health, crash, stall exit 3,
# missing config) the run dir is synced and this pod terminates itself — billing always stops.
#
# GCS works only while the 1 h GCS_ACCESS_TOKEN lasts (startup + first syncs); after that the durable
# channel is MLflow (health, videos, checkpoints, best model). RunPod injects RUNPOD_POD_ID.
set -uo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
APP_DIR="${APP_DIR:-/app}"
POD_SELF="${APP_DIR}/runpod/scripts/pod_self.sh"
GCS_TIMEOUT="${GCS_TIMEOUT:-300}"
SYNC_INTERVAL="${SYNC_INTERVAL:-600}"
SYNC_PID=""
SYNC_ENABLED=0

log() { echo "=== $* ==="; }

gcs() {
    local token_file="${WORKSPACE_DIR}/control/gcs_token"
    if [ -s "$token_file" ]; then
        CLOUDSDK_AUTH_ACCESS_TOKEN="$(cat "$token_file")" timeout "$GCS_TIMEOUT" gsutil "$@"
    else
        timeout "$GCS_TIMEOUT" gsutil "$@"
    fi
}

sync_up() {
    [ "$SYNC_ENABLED" = 1 ] || return 0
    gcs -m -q rsync -r "$RUN_DIR" "${GCS_ROOT}/runs/${RUN_ID}" || log "sync to GCS failed (token expired? MLflow has the artifacts)"
}

finish() {
    local rc=$?
    trap - EXIT
    [ -n "$SYNC_PID" ] && kill "$SYNC_PID" 2>/dev/null
    log "final sync (exit code $rc)"
    sync_up
    bash "$POD_SELF" remove || log "POD SELF-TERMINATION FAILED — remove it in the RunPod console"
    exit "$rc"
}
trap finish EXIT

for var in SEGARL_GAME RUN_ID GCS_BUCKET; do
    [ -n "${!var:-}" ] || { log "$var not set"; exit 2; }
done
MODE="${MODE:-train}"
GAME="$SEGARL_GAME"
GCS_ROOT="gs://${GCS_BUCKET}/sega-rl"
RUN_DIR="${WORKSPACE_DIR}/runs/${RUN_ID}"
export SEGARL_RUNS_DIR="${WORKSPACE_DIR}/runs" SEGARL_RUN_ID="$RUN_ID"
# A slow tracker must not stall training long enough to trip the 5-min watchdog.
export MLFLOW_HTTP_REQUEST_MAX_RETRIES="${MLFLOW_HTTP_REQUEST_MAX_RETRIES:-2}"
export MLFLOW_HTTP_REQUEST_TIMEOUT="${MLFLOW_HTTP_REQUEST_TIMEOUT:-20}"
mkdir -p "$RUN_DIR" "${WORKSPACE_DIR}/control" "${WORKSPACE_DIR}/roms"

if [ -n "${GCS_ACCESS_TOKEN:-}" ]; then
    (umask 077 && printf '%s' "$GCS_ACCESS_TOKEN" > "${WORKSPACE_DIR}/control/gcs_token")
fi

write_health() {  # status reason
    printf '{"status": "%s", "reasons": ["%s"], "last": {}}\n' "$1" "$2" > "${RUN_DIR}/health.json"
}

if gcs -q stat "${GCS_ROOT}/runs/${RUN_ID}/DONE" 2>/dev/null; then
    log "run ${RUN_ID} already DONE, nothing to do"
    exit 0
fi
SYNC_ENABLED=1

if [ "$GAME" != "airstriker" ]; then
    log "pulling ROMs"
    pulled=0
    gcs -m -q cp "${GCS_ROOT}/roms/*" "${WORKSPACE_DIR}/roms/" && pulled=1
    # ROMs are uploaded right before each launch and must never stay in GCS (Juan's rule).
    gcs -m -q rm "${GCS_ROOT}/roms/*" || log "could not delete ROMs from GCS — remove them by hand"
    if [ "$pulled" != 1 ] || ! python -m stable_retro.import "${WORKSPACE_DIR}/roms" || ! sega-rl check-rom; then
        log "ROM missing, wrong dump, or import failed"
        write_health CRITICAL rom
        exit 1
    fi
fi

log "pulling run dir (resume)"
gcs -m -q rsync -r "${GCS_ROOT}/runs/${RUN_ID}" "$RUN_DIR" 2>/dev/null || log "no prior state, fresh run"

SEGARL_COST_PER_HR_USD="$(bash "$POD_SELF" cost)"
export SEGARL_COST_PER_HR_USD
log "pod cost ${SEGARL_COST_PER_HR_USD} USD/h"

sync_loop() {
    # Kill our own sleep on TERM, otherwise it outlives the loop and keeps stdout open.
    local sleep_pid=""
    trap '[ -n "$sleep_pid" ] && kill "$sleep_pid" 2>/dev/null; exit 0' TERM
    while true; do
        sleep "$SYNC_INTERVAL" &
        sleep_pid=$!
        wait "$sleep_pid"
        sync_up
    done
}
sync_loop &
SYNC_PID=$!

log "sega-rl ${MODE} (game=${GAME}, run=${RUN_ID})"
sega-rl "$MODE"
exit $?

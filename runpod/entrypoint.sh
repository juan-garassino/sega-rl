#!/usr/bin/env bash
# RunPod entrypoint for sega-rl.
#
#   MODE=train|tune|eval|smoke   (default train)
#
# Lifecycle: GCS auth -> exit if the run is already DONE -> pull ROMs + import ->
# pull the run dir (resume) -> export pod cost -> periodic sync -> sega-rl $MODE ->
# final sync -> terminate this pod. The EXIT trap guarantees the last two steps, so
# billing stops whatever happens (finish, CRITICAL health, crash, stall exit code 3).
#
# Required env: RUN_ID, GCS_BUCKET, SEGARL_GAME; GCS_ACCESS_TOKEN (short-lived, refreshed by
# refresh-token.yml into $WORKSPACE_DIR/control/gcs_token). RunPod injects RUNPOD_POD_ID.
set -uo pipefail

MODE="${MODE:-train}"
GAME="${SEGARL_GAME:?SEGARL_GAME not set}"
RUN_ID="${RUN_ID:?RUN_ID not set}"
GCS_BUCKET="${GCS_BUCKET:?GCS_BUCKET not set}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
APP_DIR="${APP_DIR:-/app}"
SYNC_INTERVAL="${SYNC_INTERVAL:-600}"

GCS_ROOT="gs://${GCS_BUCKET}/sega-rl"
RUN_DIR="${WORKSPACE_DIR}/runs/${RUN_ID}"
TOKEN_FILE="${WORKSPACE_DIR}/control/gcs_token"
POD_SELF="${APP_DIR}/runpod/scripts/pod_self.sh"

export SEGARL_RUNS_DIR="${WORKSPACE_DIR}/runs" SEGARL_RUN_ID="$RUN_ID"
mkdir -p "$RUN_DIR" "$(dirname "$TOKEN_FILE")" "${WORKSPACE_DIR}/roms"

log() { echo "=== $* ==="; }

if [ -n "${GCS_ACCESS_TOKEN:-}" ]; then
    umask 077
    printf '%s' "$GCS_ACCESS_TOKEN" > "${TOKEN_FILE}.new" && mv "${TOKEN_FILE}.new" "$TOKEN_FILE"
    umask 022
fi

gcs() {
    # Re-read the token before every call: refresh-token.yml rewrites it every 50 min.
    if [ -s "$TOKEN_FILE" ]; then
        CLOUDSDK_AUTH_ACCESS_TOKEN="$(cat "$TOKEN_FILE")" gsutil "$@"
    else
        gsutil "$@"
    fi
}

sync_up() {
    gcs -m -q rsync -r "$RUN_DIR" "${GCS_ROOT}/runs/${RUN_ID}" || log "sync to GCS failed (retrying next interval)"
}

write_health() {  # status reason
    printf '{"status": "%s", "reasons": ["%s"], "last": {}}\n' "$1" "$2" > "${RUN_DIR}/health.json"
}

SYNC_PID=""
finish() {
    local rc=$?
    trap - EXIT
    [ -n "$SYNC_PID" ] && kill "$SYNC_PID" 2>/dev/null
    log "final sync (exit code $rc)"
    sync_up
    bash "$POD_SELF" remove
    exit "$rc"
}

if gcs -q stat "${GCS_ROOT}/runs/${RUN_ID}/DONE" 2>/dev/null; then
    log "run ${RUN_ID} already DONE, nothing to do"
    bash "$POD_SELF" remove
    exit 0
fi

trap finish EXIT

if [ "$GAME" != "airstriker" ]; then
    log "pulling ROMs"
    if ! gcs -m -q cp "${GCS_ROOT}/roms/*" "${WORKSPACE_DIR}/roms/" \
        || ! python -m stable_retro.import "${WORKSPACE_DIR}/roms"; then
        log "ROM pull/import failed"
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

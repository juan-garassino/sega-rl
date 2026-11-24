#!/usr/bin/env bash
# Receives a fresh GCS access token on stdin (from refresh-token.yml via `runpodctl exec`)
# and swaps it into the control file atomically.
set -euo pipefail

TOKEN_FILE="${WORKSPACE_DIR:-/workspace}/control/gcs_token"
mkdir -p "$(dirname "$TOKEN_FILE")"
umask 077

TOKEN="$(cat)"
[ -n "$TOKEN" ] || { echo "ERROR: empty token on stdin" >&2; exit 1; }
printf '%s' "$TOKEN" > "${TOKEN_FILE}.new"
mv "${TOKEN_FILE}.new" "$TOKEN_FILE"
echo "gcs_token refreshed at $(date -Iseconds)"

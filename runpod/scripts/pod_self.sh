#!/usr/bin/env bash
# The pod acting on itself through the RunPod GraphQL API.
#   pod_self.sh cost    -> prints this pod's USD/hour (0 when unknown)
#   pod_self.sh remove  -> terminates this pod (stops billing); no-op outside RunPod
set -uo pipefail

CMD="${1:?usage: pod_self.sh cost|remove}"
POD_ID="${RUNPOD_POD_ID:-}"
API_KEY="${RUNPOD_API_KEY:-${SEGARL_RUNPOD_API_KEY:-}}"

if [ -z "$POD_ID" ] || [ -z "$API_KEY" ]; then
    echo "pod_self: not on RunPod (no RUNPOD_POD_ID/API key), skipping '$CMD'" >&2
    [ "$CMD" = cost ] && echo 0
    exit 0
fi

gql() {
    curl -s -X POST https://api.runpod.io/graphql -H "Authorization: Bearer ${API_KEY}" \
        -H "Content-Type: application/json" -d "$1"
}

case "$CMD" in
    cost)
        gql "{\"query\":\"query { pod(input: {podId: \\\"${POD_ID}\\\"}) { costPerHr } }\"}" \
            | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["data"]["pod"]["costPerHr"] or 0)
except Exception: print(0)'
        ;;
    remove)
        echo "pod_self: terminating pod ${POD_ID}" >&2
        gql "{\"query\":\"mutation { podTerminate(input: {podId: \\\"${POD_ID}\\\"}) }\"}" >/dev/null
        ;;
    *)
        echo "unknown command: $CMD" >&2
        exit 2
        ;;
esac

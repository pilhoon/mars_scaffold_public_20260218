#!/usr/bin/env bash
set -euo pipefail
TASK_ID="${1:?usage: ./bin/run_task.sh <task_id>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUNDLE_TASK_YAML="$ROOT/tmp/$TASK_ID/task.yaml"
if [[ -f "$BUNDLE_TASK_YAML" ]]; then
  RUN_DIR="$ROOT/tmp/$TASK_ID/runs/$TASK_ID"
else
  RUN_DIR="$ROOT/runs/$TASK_ID"
fi

mkdir -p "$RUN_DIR"
echo "Run dir prepared: $RUN_DIR"

#!/usr/bin/env bash
set -euo pipefail
TASK_ID="${1:?usage: ./bin/start.sh <task_id>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LEGACY_TASK_YAML="$ROOT/tasks/$TASK_ID/task.yaml"
BUNDLE_TASK_YAML="$ROOT/tmp/$TASK_ID/task.yaml"
if [[ -f "$BUNDLE_TASK_YAML" ]]; then
  TASK_YAML="$BUNDLE_TASK_YAML"
  RUN_DIR="${RUN_DIR:-$ROOT/tmp/$TASK_ID/runs/$TASK_ID}"
else
  TASK_YAML="$LEGACY_TASK_YAML"
  RUN_DIR="${RUN_DIR:-$ROOT/runs/$TASK_ID}"
fi

if [[ ! -f "$TASK_YAML" ]]; then
  echo "task yaml not found: $TASK_YAML"
  exit 1
fi

MAX_ITERS="${MAX_ITERS:-100}"

exec "$ROOT/bin/run_supervised.sh" "$RUN_DIR" "$TASK_YAML" "$MAX_ITERS"

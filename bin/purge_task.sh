#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: ./bin/purge_task.sh <task_id> [--yes]"
  exit 1
fi

TASK_ID="$1"
CONFIRM="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUNDLE_DIR="$ROOT/tmp/$TASK_ID"
TASK_DEF_DIR="$ROOT/tasks/$TASK_ID"

mapfile -t BUNDLE_RUN_DIRS < <(find "$BUNDLE_DIR/runs" -maxdepth 1 -mindepth 1 -type d -name "${TASK_ID}*" 2>/dev/null | sort || true)
mapfile -t LEGACY_RUN_DIRS < <(find "$ROOT/runs" -maxdepth 1 -mindepth 1 -type d -name "${TASK_ID}*" 2>/dev/null | sort || true)

echo "task_id=$TASK_ID"
echo "will purge:"
echo "  - $BUNDLE_DIR"
for d in "${BUNDLE_RUN_DIRS[@]}"; do
  echo "  - $d"
done
for d in "${LEGACY_RUN_DIRS[@]}"; do
  echo "  - $d"
done
echo "will keep:"
echo "  - $TASK_DEF_DIR (task definition)"

if [[ "$CONFIRM" != "--yes" ]]; then
  echo "re-run with --yes to execute"
  exit 0
fi

for d in "${BUNDLE_RUN_DIRS[@]}"; do
  "$ROOT/bin/stop.sh" "$d" --clean || true
done
for d in "${LEGACY_RUN_DIRS[@]}"; do
  "$ROOT/bin/stop.sh" "$d" --clean || true
done

rm -rf "$BUNDLE_DIR"
for d in "${LEGACY_RUN_DIRS[@]}"; do
  rm -rf "$d"
done
echo "purged."

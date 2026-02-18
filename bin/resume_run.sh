#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: ./bin/resume_run.sh <run_id|run_dir> [task_yaml] [max_iters]"
  echo "example: ./bin/resume_run.sh demo_run_20260214_231124"
  exit 1
fi

RUN_ID_OR_DIR="$1"
TASK_YAML="${2:-}"
MAX_ITERS="${3:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$RUN_ID_OR_DIR" = /* ]]; then
  RUN_DIR="$RUN_ID_OR_DIR"
else
  LEGACY_RUN_DIR="$ROOT/runs/$RUN_ID_OR_DIR"
  if [[ -d "$LEGACY_RUN_DIR" ]]; then
    RUN_DIR="$LEGACY_RUN_DIR"
  else
    mapfile -t MATCHING_RUN_DIRS < <(find "$ROOT/tmp" -maxdepth 4 -type d -path "*/runs/$RUN_ID_OR_DIR" 2>/dev/null || true)
    if [[ "${#MATCHING_RUN_DIRS[@]}" -eq 1 ]]; then
      RUN_DIR="${MATCHING_RUN_DIRS[0]}"
    else
      RUN_DIR="$LEGACY_RUN_DIR"
    fi
  fi
fi
RUN_BASENAME="$(basename "$RUN_DIR")"

LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "run dir not found: $RUN_DIR"
  exit 1
fi

if [[ -z "$TASK_YAML" && -f "$LOG_DIR/controller.log" ]]; then
  TASK_YAML="$(sed -n 's/.*task_yaml=\([^ ]*\) default_conf=.*/\1/p' "$LOG_DIR/controller.log" | head -n 1)"
fi

# Fallback 1: infer from run id suffix (e.g. <base>_YYYYMMDD_HHMMSS -> tmp/<base>/task.yaml)
if [[ -z "$TASK_YAML" ]]; then
  RUN_BASE_NO_TS="$(printf '%s' "$RUN_BASENAME" | sed -E 's/_[0-9]{8}_[0-9]{6}$//')"
  CANDIDATE="$ROOT/tmp/$RUN_BASE_NO_TS/task.yaml"
  if [[ -f "$CANDIDATE" ]]; then
    TASK_YAML="$CANDIDATE"
  fi
fi

# Fallback 2: if a single tmp task.yaml matches run id prefix, use it.
if [[ -z "$TASK_YAML" ]]; then
  mapfile -t CANDIDATES < <(find "$ROOT/tmp" -maxdepth 3 -type f -name task.yaml -path "*$(printf '%s' "$RUN_BASENAME" | sed -E 's/_[0-9]{8}_[0-9]{6}$//')*" 2>/dev/null || true)
  if [[ "${#CANDIDATES[@]}" -eq 1 ]]; then
    TASK_YAML="${CANDIDATES[0]}"
  fi
fi

if [[ -z "$TASK_YAML" ]]; then
  echo "task_yaml is required (could not infer from $LOG_DIR/controller.log)"
  echo "hint: pass it explicitly:"
  echo "  ./bin/resume_run.sh \"$RUN_DIR\" \"$ROOT/tmp/<task_name>/task.yaml\""
  echo "available task.yaml candidates under $ROOT/tmp:"
  find "$ROOT/tmp" -maxdepth 3 -type f -name task.yaml -print || true
  exit 1
fi

if [[ -z "$MAX_ITERS" && -f "$LOG_DIR/controller.log" ]]; then
  MAX_ITERS="$(sed -n 's/.* max_iters=\([0-9][0-9]*\) worker_id=.*/\1/p' "$LOG_DIR/controller.log" | head -n 1)"
fi
MAX_ITERS="${MAX_ITERS:-100}"

exec "$ROOT/bin/run_supervised.sh" "$RUN_DIR" "$TASK_YAML" "$MAX_ITERS"

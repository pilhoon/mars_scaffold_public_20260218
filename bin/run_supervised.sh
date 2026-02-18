#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: ./bin/run_supervised.sh <run_dir> <task_yaml> [max_iters]"
  exit 1
fi

RUN_DIR="$1"
TASK_YAML="$2"
MAX_ITERS="${3:-100}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src"

LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

LLM_N="${LLM_N:-1}"
EXEC_N="${EXEC_N:-1}"
START_MONITOR="${START_MONITOR:-0}"
LIVE_TAIL="${LIVE_TAIL:-1}"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "run dir not found: $RUN_DIR"
  exit 1
fi
if [[ ! -f "$TASK_YAML" ]]; then
  echo "task yaml not found: $TASK_YAML"
  exit 1
fi

existing_pids="$(pgrep -f "services\\.(controller_main|llm_worker_main|exec_worker_main|monitor_main).*--run-dir $RUN_DIR" || true)"
if [[ -n "$existing_pids" ]]; then
  echo "existing run processes detected for $RUN_DIR:"
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    ps -fp "$pid" || true
  done <<< "$existing_pids"
  echo "stop them first: ./bin/stop.sh \"$RUN_DIR\""
  exit 1
fi

declare -a CHILD_PIDS=()
declare -a PID_FILES=()

collect_descendants() {
  local root_pid="$1"
  local child=""
  while IFS= read -r child; do
    [[ -n "$child" ]] || continue
    echo "$child"
    collect_descendants "$child"
  done < <(pgrep -P "$root_pid" || true)
}

cleanup_children() {
  local sig="${1:-TERM}"
  local pid=""
  local child=""
  declare -A seen=()
  declare -a descendants=()

  for pid in "${CHILD_PIDS[@]}"; do
    while IFS= read -r child; do
      [[ -n "$child" ]] || continue
      if [[ -z "${seen[$child]:-}" ]]; then
        seen[$child]=1
        descendants+=("$child")
      fi
    done < <(collect_descendants "$pid")
  done

  for child in "${descendants[@]}"; do
    if kill -0 "$child" 2>/dev/null; then
      kill "-$sig" "$child" 2>/dev/null || true
    fi
  done

  for pid in "${CHILD_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "-$sig" "$pid" 2>/dev/null || true
    fi
  done
}

remove_pid_files() {
  local f=""
  for f in "${PID_FILES[@]}"; do
    rm -f "$f"
  done
}

on_exit() {
  local exit_code="$1"
  trap - INT TERM EXIT
  cleanup_children TERM
  sleep 1
  cleanup_children KILL
  remove_pid_files
  exit "$exit_code"
}

trap 'on_exit 130' INT
trap 'on_exit 143' TERM
trap 'on_exit $?' EXIT

start_proc() {
  local name="$1"
  shift
  local logfile="$LOG_DIR/$name.log"
  local pidfile="$LOG_DIR/$name.pid"

  {
    printf '\n===== %s start %s =====\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$name"
  } >> "$logfile"

  "$@" >> "$logfile" 2>&1 &
  local pid=$!
  CHILD_PIDS+=("$pid")
  PID_FILES+=("$pidfile")
  echo "$pid" > "$pidfile"
  echo "started: $name (pid=$pid)"
}

start_proc "llm-1" \
  python -m services.llm_worker_main \
    --run-dir "$RUN_DIR" \
    --worker-id "llm-1"

if (( LLM_N > 1 )); then
  for i in $(seq 2 "$LLM_N"); do
    start_proc "llm-$i" \
      python -m services.llm_worker_main \
        --run-dir "$RUN_DIR" \
        --worker-id "llm-$i"
  done
fi

start_proc "exec-1" \
  python -m services.exec_worker_main \
    --run-dir "$RUN_DIR" \
    --worker-id "exec-1"

if (( EXEC_N > 1 )); then
  for i in $(seq 2 "$EXEC_N"); do
    start_proc "exec-$i" \
      python -m services.exec_worker_main \
        --run-dir "$RUN_DIR" \
        --worker-id "exec-$i"
  done
fi

if [[ "$START_MONITOR" == "1" ]]; then
  start_proc "monitor" \
    python -m services.monitor_main \
      --run-dir "$RUN_DIR"
fi

start_proc "controller" \
  python -m services.controller_main \
    --task-yaml "$TASK_YAML" \
    --run-dir "$RUN_DIR" \
    --worker-id "controller-1" \
    --max-iters "$MAX_ITERS"

echo "running supervised: run_dir=$RUN_DIR task_yaml=$TASK_YAML max_iters=$MAX_ITERS"
if [[ "$LIVE_TAIL" == "1" ]]; then
  shopt -s nullglob
  LOG_FILES=("$LOG_DIR"/*.log)
  shopt -u nullglob
  if (( ${#LOG_FILES[@]} > 0 )); then
    tail -n 0 -F "${LOG_FILES[@]}" &
    TAIL_PID=$!
    CHILD_PIDS+=("$TAIL_PID")
    echo "live log tail attached (${#LOG_FILES[@]} files)"
  else
    echo "live log tail skipped (no log files found)"
  fi
else
  echo "live log tail disabled (set LIVE_TAIL=1 to enable)"
  echo "logs: tail -f $LOG_DIR/controller.log"
fi
echo "press Ctrl-C to stop all child processes"

CONTROLLER_PID="$(cat "$LOG_DIR/controller.pid")"
set +e
wait "$CONTROLLER_PID"
controller_rc=$?
set -e

echo "controller exited (code=$controller_rc); stopping remaining workers"
exit "$controller_rc"

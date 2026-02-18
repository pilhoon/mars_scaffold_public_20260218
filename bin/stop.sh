#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: ./bin/stop.sh <task_id|run_dir> [--clean]"
  exit 1
fi

TARGET="$1"
DO_CLEAN="${2:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$TARGET" = /* ]]; then
  RUN_DIR="$TARGET"
else
  BUNDLE_RUN_DIR="$ROOT/tmp/$TARGET/runs/$TARGET"
  LEGACY_RUN_DIR="$ROOT/runs/$TARGET"
  if [[ -d "$BUNDLE_RUN_DIR" ]]; then
    RUN_DIR="$BUNDLE_RUN_DIR"
  else
    RUN_DIR="$LEGACY_RUN_DIR"
  fi
fi

LOG_DIR="$RUN_DIR/logs"
if [[ ! -d "$RUN_DIR" ]]; then
  echo "run dir not found: $RUN_DIR"
  exit 0
fi

declare -a PIDS=()

collect_pid() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    PIDS+=("$pid")
  fi
}

collect_descendants() {
  local root_pid="$1"
  local child=""
  while IFS= read -r child; do
    [[ -n "$child" ]] || continue
    echo "$child"
    collect_descendants "$child"
  done < <(pgrep -P "$root_pid" || true)
}

if [[ -d "$LOG_DIR" ]]; then
  for pidfile in "$LOG_DIR"/*.pid; do
    [[ -f "$pidfile" ]] || continue
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    collect_pid "$pid"
  done
fi

while IFS= read -r pid; do
  [[ -n "$pid" ]] || continue
  collect_pid "$pid"
done < <(pgrep -f "services\\.(controller_main|llm_worker_main|exec_worker_main|monitor_main).*--run-dir $RUN_DIR" || true)

# Unique pid list
declare -A seen=()
declare -a uniq_pids=()
for pid in "${PIDS[@]}"; do
  if [[ -z "${seen[$pid]:-}" ]]; then
    seen[$pid]=1
    uniq_pids+=("$pid")
  fi
done

if [[ "${#uniq_pids[@]}" -eq 0 ]]; then
  echo "no running processes for run_dir=$RUN_DIR"
else
  declare -A seen_desc=()
  declare -a descendant_pids=()
  for pid in "${uniq_pids[@]}"; do
    while IFS= read -r child; do
      [[ -n "$child" ]] || continue
      if [[ -z "${seen_desc[$child]:-}" ]]; then
        seen_desc[$child]=1
        descendant_pids+=("$child")
      fi
    done < <(collect_descendants "$pid")
  done

  echo "stopping ${#uniq_pids[@]} processes for run_dir=$RUN_DIR"
  for pid in "${descendant_pids[@]}"; do
    cmd="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    echo "  TERM child pid=$pid ${cmd:+cmd=\"$cmd\"}"
    kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${uniq_pids[@]}"; do
    cmd="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    echo "  TERM pid=$pid ${cmd:+cmd=\"$cmd\"}"
    kill -TERM "$pid" 2>/dev/null || true
  done

  deadline=$((SECONDS + 10))
  while :; do
    any_alive=0
    for pid in "${uniq_pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        any_alive=1
        break
      fi
    done
    if [[ "$any_alive" -eq 0 || "$SECONDS" -ge "$deadline" ]]; then
      break
    fi
    sleep 0.2
  done

  for pid in "${descendant_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "  KILL child pid=$pid"
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${uniq_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "  KILL pid=$pid"
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
fi

if [[ -d "$LOG_DIR" ]]; then
  rm -f "$LOG_DIR"/*.pid
fi

if [[ "$DO_CLEAN" == "--clean" ]]; then
  echo "cleaning run directory state for $RUN_DIR"
  rm -rf "$RUN_DIR"/spool/llm/running "$RUN_DIR"/spool/exec/running
  mkdir -p "$RUN_DIR"/spool/llm/running "$RUN_DIR"/spool/exec/running

  if [[ -d "$LOG_DIR" ]]; then
    rm -f "$LOG_DIR"/*.log
  fi
fi

echo "stopped."

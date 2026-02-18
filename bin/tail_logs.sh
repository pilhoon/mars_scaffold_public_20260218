#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Tail logs under legacy runs/ and task bundles under tmp/*/runs/
TAIL_CMD=(
  find
  "$ROOT/runs"
  "$ROOT/tmp"
  -type f
  \( -path "*/logs/*.log" \)
  -maxdepth 5
  -print0
)

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  "${TAIL_CMD[@]}" | xargs -0 tail -n 50 -f | awk '
    BEGIN {
      reset = sprintf("%c[0m", 27);
      c_model = sprintf("%c[1;36m", 27);  # cyan
      c_profile = sprintf("%c[1;34m", 27);# blue
      c_count = sprintf("%c[1;33m", 27);  # yellow
    }
    {
      line = $0;
      gsub(/model=[^ ]+/, c_model "&" reset, line);
      gsub(/llm_model=[^ ]+/, c_model "&" reset, line);
      gsub(/profile=[^ ]+/, c_profile "&" reset, line);
      gsub(/llm_profile=[^ ]+/, c_profile "&" reset, line);
      gsub(/profile_index=[0-9]+/, c_profile "&" reset, line);
      gsub(/llm_call_seq=[0-9]+/, c_count "&" reset, line);
      gsub(/attempt=[0-9]+/, c_count "&" reset, line);
      print line;
      fflush();
    }
  '
else
  "${TAIL_CMD[@]}" | xargs -0 tail -n 50 -f
fi

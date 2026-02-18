#!/usr/bin/env bash
set -euo pipefail

prompt="$(cat)"

if [[ "$prompt" == *"You are a Task Preparation agent."* ]]; then
  cat <<'EOF'
1) Metric and optimization direction assumptions.
- Optimize the declared validation metric from task spec.

2) Metadata/data handling plan (including validation split checks).
- Confirm train/valid/test split files and schema consistency.

3) EDA checklist for first run.
- Verify row counts, missing values, and label distribution.

4) Candidate model families and lightweight baseline order.
- Baseline first, then incremental model complexity.

5) Risk checklist (leakage, metric mismatch, runtime bottlenecks).
- Watch for leakage, regex mismatch, and long-running paths.
EOF
  exit 0
fi

if [[ "$prompt" == *"You are an Initial Idea Generation Agent."* ]] || [[ "$prompt" == *"You are an Idea Improvement Agent."* ]]; then
  cat <<'EOF'
Model
- Keep simple baseline logic in runfile.py.
Data
- Reuse available input format.
Training
- Keep runtime short.
Evaluation
- Preserve final metric print format.
EOF
  exit 0
fi

if [[ "$prompt" == *"You are a modular decomposition agent."* ]]; then
  cat <<'EOF'
{
  "modules": [
    {
      "path": "src/pipeline.py",
      "purpose": "keep helper pipeline utilities",
      "interfaces": ["build_pipeline"],
      "unit_test_focus": "module import and return type"
    }
  ],
  "orchestration_notes": ["wire module in runfile.py if needed"]
}
EOF
  exit 0
fi

if [[ "$prompt" == *"You are implementing one module from a decomposition plan."* ]]; then
  # Return an empty patch to keep integration run deterministic.
  exit 0
fi

if [[ "$prompt" == *"You are finalizing orchestration after module implementations."* ]]; then
  exit 0
fi

if [[ "$prompt" == *"You are a coding agent implementing a new draft solution."* ]]; then
  exit 0
fi

if [[ "$prompt" == *"You are a coding agent improving an existing solution."* ]]; then
  exit 0
fi

if [[ "$prompt" == *"You are a debugging agent."* ]]; then
  exit 0
fi

if [[ "$prompt" == *"You evaluate whether the reported final validation metric is valid."* ]]; then
  cat <<'EOF'
{"summary":"Output format looks valid.","metric":null,"valid_metric":true}
EOF
  exit 0
fi

if [[ "$prompt" == *"You distill a reusable solution lesson."* ]]; then
  cat <<'EOF'
Title: Keep metric output stable
Summary: The run completed and reported a parseable metric.
Empirical Findings: Runtime remained short and output retained expected format.
Key Lesson: Preserve metric line format while making incremental changes.
EOF
  exit 0
fi

if [[ "$prompt" == *"You distill a reusable debugging lesson."* ]]; then
  cat <<'EOF'
Title: Track repeated execution failures
Explanation: Repeated failures should be summarized with concrete error signatures.
Detection: Monitor exit code and metric parse failures to trigger focused fixes.
EOF
  exit 0
fi

if [[ "$prompt" == *"Determine whether the New Lesson duplicates Existing Lessons."* ]]; then
  cat <<'EOF'
{"reasoning":"No strong semantic overlap detected.","duplicate":false}
EOF
  exit 0
fi

# Default: no output, success.
exit 0

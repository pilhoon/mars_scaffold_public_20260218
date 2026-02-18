# mars_scaffold

Minimal multi-process scaffold inspired by [MARS](https://arxiv.org/abs/2602.02660) (Budget-aware MCTS + Modular repo construction + Comparative reflective memory).

- No Python multiprocessing: run independent processes from shell.
- Inter-process communication: file-based spool queues + SQLite state.
- LLM calls: configurable `argv`/`stdin` CLI invocation (default: `codex ... exec` with stdin prompt).

Core MARS loop is implemented (UCT selection, efficiency-guided reward, draft/improve/debug actions, reflective lessons, and queue-based multi-process execution). Task Preparation now includes local metadata/EDA signal extraction (dataset file profiling + split hints + entrypoint preview) and an LLM synthesis pass.

## Quick start (toy demo)

```bash
cd mars_scaffold
./bin/run_task.sh demo_task
./bin/start.sh demo_task
./bin/tail_logs.sh
```

The demo task uses a template repo that just prints a dummy metric. Replace the template and `task.yaml` for real tasks.

## Task Isolation

- Legacy mode: `tasks/<task_id>/task.yaml` + run artifacts under `runs/<task_id>/`.
- Bundle mode (recommended for real experiments): `tmp/<task_id>/task.yaml`.
  - In this mode, run artifacts are isolated under `tmp/<task_id>/runs/<task_id>/`.
  - This keeps experiment files removable as a single task bundle.
- Cleanup helper:
  - `./bin/purge_task.sh <task_id> --yes` removes `tmp/<task_id>` and legacy `runs/<task_id>` artifacts.

## Processes

- controller: plans actions (Draft/Improve/Debug), schedules jobs, updates MCTS stats.
- llm-worker: runs configured LLM CLI for prompt jobs (`codex` default; `gemini`, `claude` also supported).
- exec-worker: runs the task entrypoint and parses the metric.
- monitor: optional status printer.

## LLM backend switching

Edit `conf/default.yaml`:

- Default (`codex` stdin mode):
  - Equivalent to `cat prompt.txt | codex -m gpt-5.3-codex-spark -c 'model_reasoning_effort=high' exec --ephemeral --sandbox danger-full-access --skip-git-repo-check`
  - Global LLM start-rate limit is configurable via `llm.min_call_interval_sec` (seconds). You can also override at runtime with `LLM_MIN_CALL_INTERVAL_SEC=<sec>`.
- Gemini (`argv` mode):
  - set `llm.cli: "gemini"`
  - set `llm.args: []`
  - set `llm.mode: "argv"` (equivalent to `gemini "PROMPT"`)
- Claude (`stdin` mode):
  - set `llm.cli: "claude"`
  - set `llm.args: ["-p", "--no-session-persistence", "--dangerously-skip-permissions", "--verbose", "-"]`
  - set `llm.mode: "stdin"` (equivalent to `cat prompt.txt | claude -p --no-session-persistence --dangerously-skip-permissions --verbose -`)

### LLM profile selection mode

When `llm.profiles` contains multiple backends/models, you can choose how the controller selects a ready profile:

- `llm.profile_selection_mode: "sticky"` (default)
  - Keep using the current profile while it is ready.
  - Switch only when the current one is rate-limited/cooling down.
- `llm.profile_selection_mode: "round_robin"`
  - Rotate across ready profiles in round-robin order.

Runtime override is supported via `best_state.llm_profile.selection_mode` in run DB (`<run_dir>/state/mars.sqlite`):

```bash
RUN_DIR=/path/to/runs/<run_id>

# switch to round robin
sqlite3 "$RUN_DIR/state/mars.sqlite" \
"INSERT INTO best_state(key,value) VALUES('llm_profile.selection_mode','round_robin')
 ON CONFLICT(key) DO UPDATE SET value=excluded.value;"

# switch back to sticky
sqlite3 "$RUN_DIR/state/mars.sqlite" \
"INSERT INTO best_state(key,value) VALUES('llm_profile.selection_mode','sticky')
 ON CONFLICT(key) DO UPDATE SET value=excluded.value;"
```

## Current Search Behavior

- UCT-based node selection with root re-activation after configurable non-improving valid nodes.
- Task Preparation pass runs before search and stores a reusable summary context for downstream prompts.
- Task Preparation stores local context artifacts under `artifacts/task_prep/` and uses them as fallback context even if the LLM prep call fails.
- Expansion actions: `Draft` (root), `Improve` (valid nodes), and in-iteration `Debug` loop for failures (up to `max_debug_attempts`).
- Draft flow follows `Idea -> Module Plan -> Module-by-Module Patches -> Orchestration Patch` before execution.
- Module plan prompt requests JSON structure and the controller parses module specs robustly from structured or free-form outputs.
- Efficiency-guided reward: normalized objective score modulated by execution time penalty.
- Controller honors task wallclock budget and stops when exhausted (also constrains per-run execution timeout by remaining budget).
- Reflective memory:
  - Solution lessons distilled from run outcomes and diffs.
  - Debug lessons distilled from failure/fix traces.
  - LLM-based semantic dedup check before lesson insertion.
- Execution-review step validates metric quality (`valid_metric`) before reward update.

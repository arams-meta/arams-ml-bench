# ml-bench-template

A template repository for creating [ML-Bench](https://github.com/codimango/ml-bench-template) tasks — ML-focused coding challenges for evaluating AI coding agents.

ML-Bench tasks use the [Codimango](https://www.codimango.com) task format for model training, algorithm implementation, RL, and research tasks.

## Getting Started

1. Click **"Use this template"** on GitHub to create your own repo under `codimango`
2. Clone your new repo and create a task folder for each task
3. Install [Codimango](https://www.codimango.com): `uv tool install codimango` (or `pip install codimango`)

## Task Structure

Every task uses the same directory layout:

```
my-task/
├── instruction.md              # The prompt given to the agent
├── task.toml                   # Task metadata, timeouts, resource limits
├── README.md                   # Completion rates, model analysis
├── environment/
│   ├── Dockerfile              # Docker sandbox setup (bake ML deps at build time)
│   └── references/             # Domain reference materials (optional)
├── solution/
│   ├── solve.sh                # Oracle solution (must pass all tests)
│   └── (supporting files)      # train.py, solution.patch, etc.
└── tests/
    ├── test.sh                 # Orchestrator: smoke → schema → reference → baseline → scoring → holdout
    ├── test_smoke.py           # Fast gate: output exists or code imports
    ├── test_main.py            # Primary scoring tests
    ├── categorize_failure      # Failure mode classification
    ├── test_schema.py          # (optional) Format or API contract checks
    ├── test_reference.py       # (optional) Known-correct output on toy input
    ├── test_baseline.py        # (optional) Beats trivial baseline
    ├── test_holdout.py         # (optional) Generalization verification
    ├── reference/              # (optional) Pre-computed expected values
    └── data/                   # (optional) Task-specific data files
```

For tasks wrapping a third-party test runner (SWE-Bench style), also add `config.json`, `run_script.sh`, and `parser.py`.

## Test Infrastructure Toolkit

ML-Bench tasks draw from these capabilities as needed:

| Capability | Purpose |
|------------|---------|
| Failure Instrumentation | Categorize failures (oom, nan_loss, tolerance_miss, etc.) |
| Cost & Compute Logging | Record wall-clock, peak memory, set resource limits |
| Smoke Check | Fast gate before expensive scoring |
| Schema Validation | Validate output format or API contract |
| Reference Anchor | Compare against known-correct output on toy input |
| Baseline Floor | Solution must beat a trivial baseline |
| Holdout / Generalization | Verify solution generalizes beyond visible evaluation |
| Determinism | Seed the full RNG stack for reproducibility |

## Validating Your Task

```bash
# Structural validation (check files, config format)
codimango bench validate -p my-task

# Oracle (must score 1.0)
codimango bench run -p my-task -a oracle

# Nop — no solution applied (must score 0.0)
codimango bench run -p my-task -a nop

# Flakiness check (3 runs)
codimango bench run -p my-task -a oracle -k 3

# Multi-model differentiation (5 runs per model)
codimango bench run -p my-task -a claude-code -m claude-opus-4-6 -k 5
```

After pushing, monitor cloud validation with `codimango api tasks watch my-task`.

## Included Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| `hello-world` | Easy | Train a linear regression model and produce predictions |

## Creating a New Task

Use the `/mlbench-hackathon-guide` [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skill (from the Team AAI plugin) for step-by-step authoring guidance — test design, infrastructure, RL/research extensions, environment setup, validation, and submission.

1. Run `codimango task init "my-task"` or copy the `hello-world/` directory
2. Update `task.toml` with metadata, timeouts, and resource limits
3. Write your task prompt in `instruction.md` (do NOT use Claude/GPT for this file)
4. Set up `environment/Dockerfile` — bake all heavy deps at build time
5. Write tests using the toolkit capabilities your task needs
6. Write `solution/solve.sh` — the oracle reference solution
7. Validate: oracle 3/3, nop 0.0, multi-model runs show expected ordering

## Reviewing a Task

Use the `/review-task-mlbench-v1` [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skill (from the Team AAI plugin) to evaluate task quality.

```
/review-task-mlbench-v1 my-task
```

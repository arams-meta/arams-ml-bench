# FB Events Multi-Judge LLM-as-Judge Eval

## Description
Product-specific eval framework for FB Events agent answers. Agents generate natural
language answers to FB Events queries like "best Tech events near Austin next week"
with geo/time/category filters. Task: build a panel of 4 specialized LLM-as-a-judge
evaluators scoring for real product risks: events not in inventory, wrong
city/category/time, spam and duplicate clusters, unhelpfulness.

## Data
- `data/events.jsonl` — 649 synthetic inventory rows. Fields: `id`, `title`,
  `description`, `category`, `city`, `venue`, `lat`, `lng`, `start_time`, `end_time`,
  `popularity`, `is_spam`, `cluster_id`.
- `data/judge_dataset.json` — 20 visible query/answer pairs, no labels.
- `tests/data/judge_dataset_with_scores.json` — the same 20 with hidden
  `category`, `flaw_type`, `human_score`. Verifier only.
- `tests/data/holdout.json` — 20 unseen, same label schema.

Flaw distribution, identical in both the visible set and the holdout:

| flaw_type | count | human_score |
|---|---|---|
| none (good) | 10 | 5 |
| spam | 5 | 2 |
| wrong_city | 5 | 2 |

`human_score` is binary (5 or 2). There are **no** `hallucination` and **no**
`wrong_time` items in either set, though `instruction.md` and `reference_toy.json`
both describe them — see Known Limitations.

Dataset synthetic, human-authored + Llama-generated with human review (no 3P tokens),
baked into Docker per provenance rules.

## Capability Tested
- Reasoning over product constraints (geo, time, category, dedup, spam)
- Tool fluency: OpenAI-compatible Llama API, prompt engineering for 4 judges
- End-to-end eval framework: `judges/` modules, orchestrator, aggregation, CSV
- Generalization: correlation against hidden human scores

## Scoring
Binary `reward_type=binary`. Reward is gated by `tests/test_main.py`.

Aggregation: `final_score = round(0.35*accuracy + 0.30*filter + 0.20*spam + 0.15*helpfulness)`

Gates:
- directional `avg_good - avg_bad >= 1.0`
- correlation of `final_score` vs hidden `human_score` `>= 0.6`
- sub-gates: hallucination `accuracy_score <= 2`, spam `spam_score <= 2`,
  wrong_time/wrong_city `filter_score <= 3`
- no hardcoding: `>= 2` unique scores and std `> 0.5`

Logs written to `/logs/verifier/`: `baseline_score.txt` (margin),
`holdout_score.txt` (correlation), `ctrf.json`, `reward.txt`.

## Oracle behaviour

Measured on a `--no-cache` build of `python:3.10-slim`, commit `472d813`:

```
reward=1   failure_mode=none
avg_good=5.00  avg_bad=4.00  margin=1.00
corr=1.000
test_main: 8/8 passed
```

Per-dimension, `accuracy / filter / spam / helpfulness -> final`:

| item | scores | final |
|---|---|---|
| good | 5 / 5 / 5 / 5 | 5 |
| bad, spam | 5 / 5 / 1 / 5 | 4 |
| bad, wrong_city | 5 / 2 / 5 / 5 | 4 |

The oracle is honest: it computes verdicts from `events.jsonl` and the answer text
only. It does not read `human_score`, `flaw_type`, or `category`. The hidden-label
oracle described in earlier revisions of this file was removed in `632427a`.

Each judge calls the LLM first via `solution/judges/_llm.py` and falls back to a
deterministic inventory check when no API key is configured, which is what keeps
oracle validation reproducible without API access.

## Reproducibility
Seed 42, `PYTHONHASHSEED=42`, `task.toml` `seed = 42`, `deterministic = true`.
`environment/requirements.txt` is fully pinned including transitive deps, resolved
with `uv pip compile --python-version 3.10` to match `python:3.10-slim`.
LLM path uses temperature 0 against `Llama-3.3-70B-Instruct`.

## Anti-Cheating Measures
- Human scores and flaw labels live only in `tests/data/`, never copied into the image
- Ids are opaque (`q_00327`) and carry no label; row order is shuffled, so neither
  the key nor the position of a record reveals its category or flaw type
- Correlation gate rejects constant or random scoring
- `test_no_hardcoded_scores` requires score variance
- Reference anchors: 3 toy cases in `data/reference_toy.json`, +-1 tolerance

## Known Limitations

Current scope caveats.

1. **The LLM requirement is not enforced.** The spec requires each judge to
   call the LLM API, but no test verifies it, and a purely rule-based panel passes.
   Nothing in the Dockerfile or `task.toml` declares `OPENAI_API_BASE` /
   `OPENAI_API_KEY`, so the LLM path is not exercised during validation.
2. **The holdout is a secondary check.** `tests/test_holdout.py` evaluates an
   optional `/output/holdout_judgments.csv` and does not gate reward; the visible
   correlation and directional gates carry the anti-overfit weight.
3. **`human_score` is binary** (5 or 2), so the correlation gate and the
   directional margin gate measure the same separation.
4. **Deterministic judges are parser-coupled.** `solution/judges/_common.py`
   resolves listings by regex against the answer phrasing, so it is sensitive to
   answer format changes. The LLM path is the intended primary route.

## Files
- `data/judge_dataset.json` — 20 visible query/answer pairs
- `data/events.jsonl` — 649-row inventory
- `data/reference_toy.json` — 3 hand-crafted anchors
- `references/rubric.md`, `references/rubric_multi_judge.md` — single and 4-judge rubrics
- `tests/data/holdout.json`, `tests/data/judge_dataset_with_scores.json` — hidden labels
- `solution/solve.sh` — oracle: installs `/app/judges/`, writes `/app/judge.py`, runs the panel
- `solution/judges/` — the four judge modules plus `_llm.py` and `_common.py`
- `solution/judge_llm.py` — reference prompt-based Llama judge
- `tests/test.sh` — verifier orchestrator
- `tests/test_main.py` — directional, correlation, per-flaw detection, multi-judge presence
- `gen_realistic_event_answers.py` — deterministic answer generator
- `gen_judge_data.py` — Llama API generator for natural phrasing
- `eval_best_effort.py` — author-side scratch evaluator, not part of grading

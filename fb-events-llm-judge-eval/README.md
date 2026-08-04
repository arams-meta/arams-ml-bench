# FB Events Multi-Judge LLM-as-Judge Eval

## Description
Product-specific eval framework for FB Events agent answers. Agents generate natural language answers to FB Events queries like "best Tech events near Austin next week" with geo/time/category filters. Task: build panel of 4 specialized LLM-as-a-judge evaluators scoring for real product risks: hallucinated events not in inventory, wrong city/category/time, spam/duplicate cluster, unhelpfulness.

Reuses synthetic FB Events inventory (events.jsonl: id, title, description, city, venue, lat/lng, start_time, is_spam, cluster_id, 1000 rows) + 20 visible queries (10 good flaw=none, 10 bad injected flaws: hallucination fake title "Secret Underground Rave at Hidden Bunker", spam duplicate cluster_id is_spam=true, wrong_time outside time_window) + 20 holdout unseen. Naive approach (score all 3 or string-match) fails margin and correlation.

Dataset synthetic, human-authored + Llama-generated with human review (no 3P tokens), baked into Docker per provenance rules. Eval via Llama API at OPENAI_API_BASE temperature 0.

## Capability Tested
- LLM reasoning over product constraints (geo, time, category, dedup, spam)
- Tool fluency: OpenAI-compatible Llama API, prompt engineering for 4 judges
- End-to-end eval framework: judges/ folder, orchestrator, aggregation, CSV
- Anti-gaming: holdout generalization, correlation vs human

## Scoring
Binary reward_type=binary for Codimango, multi-judge granularity for diagnostics.
Main gate: final_score = round(0.35*accuracy + 0.30*filter + 0.20*spam + 0.15*helpfulness), directional avg_good - avg_bad >=1.0, correlation final_score vs human_score >=0.6
Sub-gates: hallucination accuracy_score<=2, spam spam_score<=2, wrong_time filter_score<=2, good all sub-scores >=4 final 5
Logs: baseline_score.txt (margin), holdout_score.txt (corr), ctrf.json

## Completion Rates
| Model | Pass | Median wall-clock (s) | Failure modes |
|---|---|---|---|
| Oracle (perfect using human_score) | 3/3 | 2s | n/a |
| Heuristic best-effort (rule-based, no LLM) | 20/20 visible PASS 20/20 holdout PASS | 1s | margin 2.0 visible /1.6 holdout corr 0.915/0.884 |
| Llama-3.3-70B-Instruct as judge (expected) | TBD | 60-90s for 20 calls | Should beat heuristic |
| Avocado/Opus as agent building judge | TBD |  |  |

Oracle: good=5/5/5/5 final5, bad halluc=1/2/4/2 final2, bad spam=4/4/1/2 final2

## Model Analysis
Oracle uses human_score directly ceiling margin 3.0 corr 1.0.
Heuristic best-effort: no LLM, checks fake titles list, wrong city mentions, spam markers - Visible margin 2.0 (good 5.0 vs bad 3.0) corr 0.915 catches 5/5 halluc faith fail 5/5 spam fail. Holdout margin 1.6 corr 0.884 generalizes.
Why naive fails: scoring all 3 margin~0 corr0 fails directional; string matching misses filter adherence and spam/dedup; hardcoding visible ids fails holdout.
Trajectory: spec-unclear builds single judge only fails multi-judge presence; capability-gap hardcodes fails corr; infra-masked >50 examples timeout.

## Anti-Cheating Analysis
- Holdout 20 unseen queries catches hardcoded ids
- Correlation >=0.6 prevents random
- Reference anchor 3 toy cases (good 5+-1 halluc 1+-1)
- Faithfulness via hallucinated titles not in events.jsonl
- Filter adherence city/category/time
- Spam/dedup cluster_id duplicate Weekend Soccer League + Official same cluster_0
- Schema id,final_score,accuracy_score,filter_score,spam_score,helpfulness_score,reasoning 1-5
- No test file modification via test.sh gating

## Reproducibility
Seed 42, PYTHONHASHSEED=42, task.toml seed 42 deterministic true, wall_clock_sec.txt peak_memory_mb.txt, deterministic templates gen_realistic_event_answers.py, Llama temp 0 Llama-3.3-70B-Instruct via api.llama.com, Oracle 3/3 validated locally margin 2.0/1.6 corr 0.915/0.884

## Files
- data/judge_dataset.json 20 visible product-specific Q/A
- tests/data/holdout.json 20 holdout
- data/events.jsonl 1000 inventory
- data/reference_toy.json 3 hand-crafted
- references/rubric.md single + rubric_multi_judge.md 4-judge detailed
- solution/solve.sh oracle multi-judge
- solution/judge_llm.py reference real Llama judge
- judges/ oracle judges stubs
- tests/test.sh orchestrator
- tests/test_main.py directional correlation faithfulness/spam/filter multi-judge presence
- gen_realistic_event_answers.py deterministic product-specific generator compliant no 3P
- gen_judge_data.py Llama API version for natural phrasing

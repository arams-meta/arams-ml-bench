# FB Events Multi-Judge Rubric (4 specialized judges)

You are building a panel of 4 LLM-as-Judge evaluators for FB Events agent answers. Each judge specializes in one product dimension.

## Inventory
- /app/data/events.jsonl: id, title, description, city, venue, category, start_time, is_spam, cluster_id
- /app/data/judge_dataset.json: 24 items with user_question, query_filters, agent_answer, ground_truth_ids

## Judge 1: Accuracy / Hallucination Judge (accuracy_judge.py)
Checks if agent_answer mentions only real events from inventory.
- PASS: all event titles/ids in answer exist in events.jsonl
- FAIL: includes fake title like "Secret Underground Rave" not in inventory, or invents venue
- Score 1-5: 5=no hallucination, 1=all hallucinated
- Prompt: "Given inventory list {titles}, does answer contain any event not in inventory? List hallucinated events if any."

## Judge 2: Filter Adherence Judge (filter_judge.py)
Checks city, category, time_window, radius_km from query_filters.
- PASS: respects all filters (city matches, category matches, start_time within window)
- FAIL: wrong city (e.g. Portland when query city=New York), ignores time, wrong category
- Score 1-5: 5=all filters respected, 1=ignores all

## Judge 3: Spam / Dedup Judge (spam_judge.py)
Checks spam and duplicate handling.
- PASS: no is_spam=true events, deduped by cluster_id (no duplicate cluster listed twice)
- FAIL: includes spam flagged event, lists Weekend Soccer League + Weekend Soccer League (Official) same cluster_0 twice
- Score 1-5

## Judge 4: Helpfulness Judge (helpfulness_judge.py)
Checks if answer actually answers user_question helpfully.
- PASS: concise, lists relevant events with venue/date, explains why match query
- FAIL: "No events found" when relevant ids exist, or unhelpful generic text, or too verbose without listing

## Aggregation (judge.py orchestrator)
- Reads all 4 judge scores per item
- final_score = weighted average: 0.35*accuracy + 0.30*filter + 0.20*spam + 0.15*helpfulness (round to int 1-5)
- Also output per-judge scores and reasoning

## Output Schema: /output/judgments.csv
id,final_score,accuracy_score,filter_score,spam_score,helpfulness_score,reasoning,faithfulness,filter_adherence,spam_dedup,helpfulness

- final_score 1-5 (aggregated)
- accuracy_score 1-5 (hallucination)
- filter_score 1-5
- spam_score 1-5
- helpfulness_score 1-5
- reasoning: aggregated explanation referencing all 4 checks
- faithfulness/filter_adherence/spam_dedup/helpfulness: pass/fail per check (for backward compat)

## Evaluation (tests)
- Good answers (flaw=none) should have final_score 5 and all sub-scores >=4
- Bad hallucination: accuracy_score <=2
- Bad spam: spam_score <=2
- Bad wrong_time/wrong_city: filter_score <=3
- Directional: avg final_score good > bad by >=1.0
- Correlation: final_score vs human_score >=0.6

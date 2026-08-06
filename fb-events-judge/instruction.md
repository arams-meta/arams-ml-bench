You are to build a multi-judge LLM-as-a-judge eval framework for FB Events agent answers - panel of 4 specialized judges.

Context:
- Event inventory lives in /app/data/events.jsonl (id, title, description, category, city, venue, lat, lng, start_time, end_time, popularity, is_spam, cluster_id, 649 rows)
- User queries + agent answers to judge are in /app/data/judge_dataset.json (24 items: user_question, query_filters, agent_answer, ground_truth_ids) - NO human_score/flaw_type here (hidden from agent)
- Rubrics in /app/references/rubric.md (single judge) and /app/references/rubric_multi_judge.md (detailed multi-judge) - read both carefully
- Reference toy examples in /app/data/reference_toy.json (3 hand-crafted)
- Hidden ground truth for evaluation (human scores, flaw labels) lives in /tests/data/judge_dataset_with_scores.json and holdout.json - NOT visible to agent at /app, only verifier uses it

Task - Multi-Judge Panel:
- Create 4 specialized judge modules in /app/judges/:
  - /app/judges/accuracy_judge.py: checks if answer hallucinates events not in events.jsonl (e.g. fake title "Secret Underground Rave" not in inventory, event accuracy)
  - /app/judges/filter_judge.py: checks city, category, time_window_start/end, radius_km adherence
  - /app/judges/spam_judge.py: checks is_spam=false and dedup by cluster_id (no duplicate cluster listed twice)
  - /app/judges/helpfulness_judge.py: checks if answer actually answers query helpfully with venue/date
  - Each judge calls LLM API (OpenAI compatible at os.getenv("OPENAI_API_BASE") with key OPENAI_API_KEY, model env JUDGE_MODEL or "Llama-3.3-70B-Instruct", temperature=0) with specialized prompt per rubric_multi_judge.md and returns score 1-5 + reasoning + pass/fail

- Create orchestrator /app/judge.py that:
  - Reads /app/data/judge_dataset.json (agent-visible, no scores)
  - Calls all 4 judges per item
  - Aggregates final_score = round(0.35*accuracy + 0.30*filter + 0.20*spam + 0.15*helpfulness)
  - Writes /output/judgments.csv with columns id,final_score,accuracy_score,filter_score,spam_score,helpfulness_score,reasoning,faithfulness,filter_adherence,spam_dedup,helpfulness,score
  - Note: final_score is primary, score column is alias for backward compat (must be same as final_score)
  - For holdout generalization, same judge logic is run on hidden holdout data in verifier (agent does NOT need to produce separate holdout file; but if /output/holdout_judgments.csv exists it will be checked informationally)

Constraints:
- No hardcoded scores, each judge must call LLM API (must NOT read hidden files in /tests/data/ or flaw_type/human_score)
- All scores 1-5 int, reasoning non-empty, pass/fail per product dimension
- Use seed 42, determinism where possible
- Temperature 0 for LLM calls to keep deterministic

Expected output: /output/judgments.csv with 24 judgments where good answers > bad answers by >=1 margin for final_score, correlation of final_score vs hidden human_score >=0.6, and sub-judges: hallucination answers accuracy_score <=2, spam answers spam_score <=2, wrong_time/wrong_city filter_score <=3. Good answers should have all sub-scores >=4 and final_score 5.

Your multi-judge panel will be tested on visible set + hidden holdout set to ensure product-specific accuracy and generalization (holdout not visible to agent).

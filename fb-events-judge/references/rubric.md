# FB Events LLM-as-Judge Rubric

You are judging an AI agent that answers user queries about Facebook Events.

## Input
- user_question: natural language query (e.g. "best Tech events near New York next month")
- query_filters: {"city": "...", "category": "...", "time_window_start": "...", "time_window_end": "...", "radius_km": ...}
- agent_answer: natural language response listing/recommending events

## Ground Truth (not visible to judge at runtime, for evaluator only)
- relevant_event_ids: ids from events.jsonl that truly match filters
- events inventory in /app/data/events.jsonl with fields: id, title, description, city, category, start_time, is_spam, cluster_id

## Scoring 1-5

5 = Perfect:
- Only mentions events that exist in events.jsonl (no hallucinated titles/ids)
- Respects ALL filters: city, category, time window
- No spam (is_spam=true excluded)
- No duplicate cluster_id (deduped)
- Helpful summary, answers query

4 = Mostly good, minor flaw (e.g. 1 extra irrelevant event but still helpful)

3 = Partial: mixes good and bad, or misses key constraint but still somewhat relevant

2 = Major flaw:
- Wrong city, or includes spam, or hallucinates event not in inventory, or ignores time window
- OR duplicate cluster not deduped

1 = Completely wrong / hallucinated / unsafe: all events invented or opposite city/category

## Output JSON Schema (required)
{
  "score": int 1-5,
  "reasoning": "brief explanation referencing which filters passed/failed, which events checked",
  "checks": {
    "faithfulness": "pass/fail - all events in inventory?",
    "filter_adherence": "pass/fail - city/category/time respected?",
    "spam_dedup": "pass/fail - no spam, deduped?",
    "helpfulness": "pass/fail"
  }
}

The judge must output valid JSON with these fields.

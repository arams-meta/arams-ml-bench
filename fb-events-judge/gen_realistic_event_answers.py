#!/usr/bin/env python3
"""
Realistic FB Events Agent Response Generator WITHOUT LLM API
Produces product-specific visible/holdout data for judge eval - deterministic, human-reviewable.

This is a compliant fallback that creates good/bad answers using templates + real event inventory,
so you can see realistic data on Desktop without needing Llama API key.

After reviewing, you can later re-run with real LLM to get more natural phrasing.
"""

import json, random
from pathlib import Path

SRC = Path("/Users/arams/fb-events-rag-self-improve/data")
DST = Path(__file__).parent / "data"
DST_TEST = Path(__file__).parent / "tests" / "data"
DESKTOP = Path.home() / "Desktop" / "fb-events-judge-data"

random.seed(42)


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


events = load_jsonl(SRC / "events.jsonl")
queries = load_jsonl(SRC / "queries.jsonl")
relevance = load_jsonl(SRC / "relevance.jsonl")
holdout_q = load_jsonl(SRC / "holdout_queries.jsonl")
holdout_rel = load_jsonl(SRC / "holdout_relevance.jsonl")

event_by_id = {e["id"]: e for e in events}
rel_by_q = {r["query_id"]: r["relevant_event_ids"] for r in relevance}
rel_holdout = {r["query_id"]: r["relevant_event_ids"] for r in holdout_rel}

# Pick 20 + 20
sample_q = random.sample(queries, 20)
sample_h = random.sample(holdout_q, 20)

DST.mkdir(parents=True, exist_ok=True)
DST_TEST.mkdir(parents=True, exist_ok=True)
DESKTOP.mkdir(parents=True, exist_ok=True)


def make_good_answer(q, rel_events):
    if not rel_events:
        return f"No {q.get('category', '')} events found in {q.get('city')} for your dates {q.get('time_window_start')} to {q.get('time_window_end')}. Try expanding radius (current {q.get('radius_km')}km) or category."
    titles = []
    for e in rel_events[:3]:
        titles.append(
            f"{e['title']} at {e.get('venue', 'unknown venue')} in {e['city']} on {e['start_time'][:10]}"
        )
    return (
        f"Here are {q.get('category', '')} events near {q.get('city', 'your area')} matching '{q['text']}': "
        + "; ".join(titles)
        + ". All within your time window and deduped."
    )


def make_bad_hallucination(q, rel_events):
    fake = "Secret Underground Rave at Hidden Bunker - Exclusive Invite Only"
    if rel_events:
        real = rel_events[0]
        return f"Found events: {real['title']} in {q.get('city')} and also {fake} in Los Angeles (this event does NOT exist in inventory, hallucinated) plus another event in Berlin. This violates faithfulness - includes events not in events.jsonl and wrong city."
    else:
        return f"Here are events: {fake} in Tokyo and Mystery Tech Summit in Unknown City - these are hallucinated titles not in inventory."


def make_bad_spam(q, rel_events):
    if not rel_events:
        return "No events found but here is spam event: FREE MONEY EVENT - CLICK HERE!!! and duplicate same event twice."
    # Find duplicate cluster if possible
    e = rel_events[0]
    # Find another with same cluster_id if exists
    cluster = e.get("cluster_id")
    dup = next(
        (
            ev
            for ev in events
            if ev.get("cluster_id") == cluster and ev["id"] != e["id"]
        ),
        None,
    )
    if dup:
        return f"Music events: {e['title']} at {e['venue']} and duplicate {dup['title']} at {dup['venue']} (same cluster_id {cluster} not deduped, should be deduped). Also includes spam flagged event (is_spam=true) which should be excluded."
    else:
        return f"Found {e['title']} and again {e['title']} (duplicate, same cluster_id {e.get('cluster_id')} listed twice). Also includes spam event with is_spam=true."


def make_bad_time(q, rel_events):
    return f"For query '{q['text']}' with window {q.get('time_window_start')} to {q.get('time_window_end')} in {q.get('city')}, recommending event from 2025-01-01 (outside time window) and event in 2027 (far future). Ignores time filter: filter_adherence fail."


def gen_dataset(qlist, rel_map, tag):
    out = []
    for i, q in enumerate(qlist):
        qid = q["id"]
        rel_ids = rel_map.get(qid, [])
        rel_events = [event_by_id[x] for x in rel_ids[:3] if x in event_by_id]
        is_good = i % 2 == 0
        if is_good:
            ans = make_good_answer(q, rel_events)
            flaw = "none"
            score = 5
            cat = "good"
        else:
            flaw = random.choice(["hallucination", "spam", "wrong_time"])
            if flaw == "hallucination":
                ans = make_bad_hallucination(q, rel_events)
            elif flaw == "spam":
                ans = make_bad_spam(q, rel_events)
            else:
                ans = make_bad_time(q, rel_events)
            score = 2 if flaw != "hallucination" else 1
            cat = "bad"
        rec = {
            "id": f"{qid}_{'good' if is_good else 'bad_' + flaw}",
            "query_id": qid,
            "user_question": q["text"],
            "query_filters": {
                "city": q.get("city"),
                "category": q.get("category"),
                "time_window_start": q.get("time_window_start"),
                "time_window_end": q.get("time_window_end"),
                "radius_km": q.get("radius_km"),
                "lat": q.get("lat"),
                "lng": q.get("lng"),
            },
            "agent_answer": ans,
            "ground_truth_ids": rel_ids[:5],
            "flaw_type": flaw,
            "human_score": score,
            "category": cat,
        }
        out.append(rec)
    return out


visible = gen_dataset(sample_q, rel_by_q, "visible")
holdout = gen_dataset(sample_h, rel_holdout, "holdout")

# Toy reference (human authored, always included)
toy = [
    {
        "id": "toy_001_good",
        "user_question": "Tech events in New York next month",
        "query_filters": {"city": "New York", "category": "Tech"},
        "agent_answer": "Here are Tech events in New York: AI Summit at Javits Center on 2026-06-05, Startup Pitch at Brooklyn Hub on 2026-06-06. Both within your time window, deduped, no spam.",
        "ground_truth_ids": ["evt_0001", "evt_0002"],
        "flaw_type": "none",
        "human_score": 5,
        "category": "good",
    },
    {
        "id": "toy_002_bad_halluc",
        "user_question": "Tech events in New York next month",
        "query_filters": {"city": "New York", "category": "Tech"},
        "agent_answer": "Here are events: Secret Underground Rave in Los Angeles (not in inventory, hallucinated) and AI Summit in New York. Also Goa Psytrance in Berlin (wrong city, not in events.jsonl).",
        "ground_truth_ids": ["evt_0001"],
        "flaw_type": "hallucination",
        "human_score": 1,
        "category": "bad",
    },
    {
        "id": "toy_003_bad_spam",
        "user_question": "Music events in SF this weekend",
        "query_filters": {"city": "San Francisco", "category": "Music"},
        "agent_answer": "Music events: Weekend Soccer League at Market Square Portland (wrong city) and again Weekend Soccer League (Official) same cluster_0 duplicate not deduped. Includes spam event.",
        "ground_truth_ids": [],
        "flaw_type": "spam",
        "human_score": 2,
        "category": "bad",
    },
]

# Write to task locations
with open(DST / "judge_dataset.json", "w") as f:
    json.dump(visible, f, indent=2)
with open(DST_TEST / "holdout.json", "w") as f:
    json.dump(holdout, f, indent=2)
with open(DST / "reference_toy.json", "w") as f:
    json.dump(toy, f, indent=2)
with open(DST_TEST / "reference_toy.json", "w") as f:
    json.dump(toy, f, indent=2)

# Write to Desktop for user to see
with open(DESKTOP / "judge_dataset_visible_20.json", "w") as f:
    json.dump(visible, f, indent=2)
with open(DESKTOP / "holdout_20.json", "w") as f:
    json.dump(holdout, f, indent=2)
with open(DESKTOP / "reference_toy_3.json", "w") as f:
    json.dump(toy, f, indent=2)
with open(DESKTOP / "README.txt", "w") as f:
    f.write("""
FB Events LLM-as-Judge - Realistic Product-Specific Dataset

Files:
- judge_dataset_visible_20.json (20 records: 10 good / 10 bad)
  Each record: id, user_question (from queries.jsonl product query), query_filters (city/category/time/radius),
               agent_answer (good = lists real events from events.jsonl with correct filters,
                             bad = hallucination/spam/wrong_time flaws),
               ground_truth_ids, flaw_type, human_score (5 good, 1-2 bad), category

- holdout_20.json (20 holdout, same schema, unseen queries)
- reference_toy_3.json (3 hand-crafted obvious cases)

Flaw types explained:
- none: perfect answer, respects all filters, no spam, deduped
- hallucination: includes fake event title not in events.jsonl or wrong city
- spam: duplicate cluster_id not deduped or includes is_spam=true
- wrong_time: ignores time window, recommends far outside dates

Next:
1. Open judge_dataset_visible_20.json in VSCode - see product-specific examples
2. Good answers show pattern judge should score 5
3. Bad answers show what judge must catch to score <=2
4. These are rule-based templates - later you can re-run gen_judge_data.py with real Llama API for more natural phrasing

For your eval task, the agent's judge.py must call Llama API per rubric.md to score these.
""")

# Also copy events subset for context
with (
    open(SRC / "events.jsonl") as fin,
    open(DESKTOP / "sample_events_10.jsonl", "w") as fout,
):
    for i, line in enumerate(fin):
        if i >= 10:
            break
        fout.write(line)

print(f"Generated realistic dataset:")
print(f"  Task: {DST}/judge_dataset.json ({len(visible)} records)")
print(f"  Task holdout: {DST_TEST}/holdout.json ({len(holdout)} records)")
print(f"  Desktop: {DESKTOP}/judge_dataset_visible_20.json")
print(f"  Desktop: {DESKTOP}/holdout_20.json")
print(f"  Desktop: {DESKTOP}/reference_toy_3.json")
print(f"  Desktop: {DESKTOP}/sample_events_10.jsonl")

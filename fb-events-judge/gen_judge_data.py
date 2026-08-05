#!/usr/bin/env python3
"""
FB Events Agent Response Generation for LLM-as-Judge Eval
Generates synthetic Q/A dataset on top of existing event synthetic data.

This script uses Llama (1P via OPENAI_API_BASE) to generate good/bad agent answers
for product-specific FB Events queries. Human must review outputs before baking into Docker.

Usage:
  export OPENAI_API_BASE="https://api.llama.com/compat/v1/"
  export OPENAI_API_KEY="..."
  python3 gen_judge_data.py --sample 20 --seed 42

Outputs:
  data/events_subset.jsonl (copied events for judge)
  data/judge_dataset.json (20 visible: 10 good + 10 bad)
  tests/data/holdout.json (20 holdout)
  data/reference_toy.json (3 hand-crafted reference cases)
"""

import os, json, random, argparse
from pathlib import Path
from collections import defaultdict

# Source data from your existing task
SRC_DATA = Path("/Users/arams/fb-events-rag-self-improve/data")
DST_DATA = Path(__file__).parent / "data"
DST_TEST_DATA = Path(__file__).parent / "tests" / "data"

# Llama client (1P - allowed for container content generation with human review)
try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("openai not installed, using template mode")


def get_client():
    if not HAS_OPENAI:
        return None
    base = os.getenv("OPENAI_API_BASE")
    key = os.getenv("OPENAI_API_KEY")
    if not base or not key:
        print(f"Missing OPENAI_API_BASE or KEY, using template mode")
        return None
    return OpenAI(base_url=base, api_key=key)


def load_jsonl(p):
    rows = []
    with open(p) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def call_llama(client, system, user, model="Llama-3.3-70B-Instruct", temp=0.2):
    if client is None:
        # Template fallback - human must replace
        return f"[TEMPLATE] {user[:100]}..."
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temp,
            max_tokens=400,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM call failed {e}, fallback template")
        return f"[FALLBACK] {user[:100]}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        type=int,
        default=40,
        help="total queries to use (20 visible + 20 holdout)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    client = get_client()

    print(f"Loading source data from {SRC_DATA}")
    events = load_jsonl(SRC_DATA / "events.jsonl")
    queries = load_jsonl(SRC_DATA / "queries.jsonl")
    relevance = load_jsonl(SRC_DATA / "relevance.jsonl")
    holdout_queries = load_jsonl(SRC_DATA / "holdout_queries.jsonl")
    holdout_rel = load_jsonl(SRC_DATA / "holdout_relevance.jsonl")

    # Build maps
    event_by_id = {e["id"]: e for e in events}
    rel_by_q = {r["query_id"]: r["relevant_event_ids"] for r in relevance}
    rel_holdout_by_q = {r["query_id"]: r["relevant_event_ids"] for r in holdout_rel}

    # Sample queries for visible set (first 20) and holdout (from holdout_queries)
    sampled_queries = random.sample(queries, min(20, len(queries)))
    sampled_holdout = random.sample(holdout_queries, min(20, len(holdout_queries)))

    # Prepare output dirs
    DST_DATA.mkdir(parents=True, exist_ok=True)
    DST_TEST_DATA.mkdir(parents=True, exist_ok=True)

    # Copy subset of events that are relevant to sampled queries (to keep Docker small but faithful)
    needed_ids = set()
    for q in sampled_queries + sampled_holdout:
        qid = q["id"]
        ids = rel_by_q.get(qid, []) + rel_holdout_by_q.get(qid, [])
        needed_ids.update(ids[:5])  # top 5 per query for context
    # Also add some spam / duplicate examples for bad-case detection
    spam_examples = [e["id"] for e in events if e.get("is_spam")][:5]
    needed_ids.update(spam_examples)
    subset_events = [event_by_id[i] for i in needed_ids if i in event_by_id]
    print(f"Subset events: {len(subset_events)} needed for judge eval")

    with open(DST_DATA / "events_subset.jsonl", "w") as f:
        for e in subset_events:
            f.write(json.dumps(e) + "\n")
    # Full events also baked for completeness (optional, 1000 rows is ok)
    # Copy relevant files
    with open(DST_DATA / "events.jsonl", "w") as out:
        for e in events[:1000]:  # cap at 1000 for size
            out.write(json.dumps(e) + "\n")

    # Generation prompts
    SYSTEM_GOOD = "You are a helpful FB Events assistant. Answer concisely, list only real events from provided context, respect city/category/time filters, never include spam, dedup by cluster."
    SYSTEM_BAD_HALLUC = "You are FB Events assistant but you hallucinate: include one event title that does NOT exist in inventory and wrong city."
    SYSTEM_BAD_SPAM = "You are FB Events assistant but you are sloppy: include a spam event and duplicate same cluster_id twice."
    SYSTEM_BAD_TIME = "You are FB Events assistant but ignore time window: recommend event outside the time window."

    def build_good_prompt(q, rel_ids):
        rel_events = [event_by_id[i] for i in rel_ids[:3] if i in event_by_id]
        ctx = "\n".join(
            [
                f"- {e['id']}: {e['title']} in {e['city']} ({e['category']}) at {e['start_time']} venue {e['venue']}"
                for e in rel_events
            ]
        )
        if not ctx:
            ctx = "(No relevant events found, say no events found and be helpful)"
        return f"Query: {q['text']}\nFilters: city={q.get('city')} category={q.get('category')} time={q.get('time_window_start')} to {q.get('time_window_end')}\nRelevant inventory (use only these):\n{ctx}\n\nWrite 2-3 sentence helpful answer listing events by title and venue. Keep city/time correct."

    def build_bad_prompt(q, rel_ids, flaw):
        base = build_good_prompt(q, rel_ids)
        if flaw == "hallucination":
            return (
                base
                + "\nIMPORTANT: Add one extra event with fake title like 'Secret Underground Rave' that is NOT in inventory, and say it's in wrong city."
            )
        elif flaw == "spam":
            return (
                base
                + "\nIMPORTANT: Include duplicate titles for same cluster and include a spammy event."
            )
        else:
            return (
                base
                + f"\nIMPORTANT: Violate time filter - recommend event far outside window. Flaw={flaw}"
            )

    judge_dataset = []
    holdout_dataset = []

    # Visible set: 20 queries -> 10 good + 10 bad = 20 records? Actually per spec 10 good +10 bad =20 total. We'll do 20 queries *1 version =20 records (10 good /10 bad)
    for i, q in enumerate(sampled_queries):
        qid = q["id"]
        rel_ids = rel_by_q.get(qid, [])
        is_good = i % 2 == 0  # alternate good/bad to get 10/10
        flaw = (
            "none"
            if is_good
            else random.choice(["hallucination", "spam", "wrong_time"])
        )
        human_score = 5 if is_good else 2

        if is_good:
            prompt = build_good_prompt(q, rel_ids)
            answer = call_llama(client, SYSTEM_GOOD, prompt, temp=0.2)
        else:
            if flaw == "hallucination":
                system = SYSTEM_BAD_HALLUC
            elif flaw == "spam":
                system = SYSTEM_BAD_SPAM
            else:
                system = SYSTEM_BAD_TIME
            prompt = build_bad_prompt(q, rel_ids, flaw)
            answer = call_llama(client, system, prompt, temp=0.7)

        record = {
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
            "agent_answer": answer,
            "ground_truth_ids": rel_ids[:5],
            "flaw_type": flaw,
            "human_score": human_score,  # for testing correlation, hidden from judge at runtime for holdout
            "category": "good" if is_good else "bad",
        }
        judge_dataset.append(record)

    # Holdout: 20 queries, 10 good 10 bad
    for i, q in enumerate(sampled_holdout):
        qid = q["id"]
        rel_ids = rel_holdout_by_q.get(qid, [])
        is_good = i % 2 == 0
        flaw = (
            "none"
            if is_good
            else random.choice(["hallucination", "spam", "wrong_time"])
        )
        human_score = 5 if is_good else 2

        prompt = (
            build_good_prompt(q, rel_ids)
            if is_good
            else build_bad_prompt(q, rel_ids, flaw)
        )
        system = SYSTEM_GOOD if is_good else SYSTEM_BAD_HALLUC
        answer = call_llama(client, system, prompt, temp=0.2 if is_good else 0.7)

        record = {
            "id": f"{qid}_{'good' if is_good else 'bad_' + flaw}",
            "query_id": qid,
            "user_question": q["text"],
            "query_filters": {
                "city": q.get("city"),
                "category": q.get("category"),
                "time_window_start": q.get("time_window_start"),
                "time_window_end": q.get("time_window_end"),
            },
            "agent_answer": answer,
            "ground_truth_ids": rel_ids[:5],
            "flaw_type": flaw,
            "human_score": human_score,
            "category": "good" if is_good else "bad",
        }
        holdout_dataset.append(record)

    # Write outputs
    with open(DST_DATA / "judge_dataset.json", "w") as f:
        json.dump(judge_dataset, f, indent=2)
    with open(DST_TEST_DATA / "holdout.json", "w") as f:
        json.dump(holdout_dataset, f, indent=2)

    # Reference toy: 3 hand-crafted obvious cases (human-authored, must be reviewed)
    toy = [
        {
            "id": "toy_001_good",
            "user_question": "Tech events in New York next month",
            "query_filters": {"city": "New York", "category": "Tech"},
            "agent_answer": "Here are Tech events in New York: AI Summit at Javits Center, Startup Pitch at Brooklyn Hub.",
            "ground_truth_ids": ["evt_0001", "evt_0002"],
            "flaw_type": "none",
            "human_score": 5,
            "category": "good",
        },
        {
            "id": "toy_002_bad_halluc",
            "user_question": "Tech events in New York next month",
            "query_filters": {"city": "New York", "category": "Tech"},
            "agent_answer": "Here are events: Secret Underground Rave in Los Angeles (not in inventory) and AI Summit in New York. Also Goa Psytrance in Berlin.",
            "ground_truth_ids": ["evt_0001"],
            "flaw_type": "hallucination",
            "human_score": 1,
            "category": "bad",
        },
        {
            "id": "toy_003_bad_spam",
            "user_question": "Music events in SF this weekend",
            "query_filters": {"city": "San Francisco", "category": "Music"},
            "agent_answer": "Music events: Weekend Soccer League at Market Square Portland (wrong city) - same event listed twice Weekend Soccer League and Weekend Soccer League (Official) which share cluster_0.",
            "ground_truth_ids": [],
            "flaw_type": "spam",
            "human_score": 2,
            "category": "bad",
        },
    ]
    with open(DST_DATA / "reference_toy.json", "w") as f:
        json.dump(toy, f, indent=2)
    with open(DST_TEST_DATA / "reference_toy.json", "w") as f:
        json.dump(toy, f, indent=2)

    print(f"\nGenerated:")
    print(
        f"  {DST_DATA}/judge_dataset.json - {len(judge_dataset)} records (10 good/10 bad)"
    )
    print(f"  {DST_TEST_DATA}/holdout.json - {len(holdout_dataset)} records")
    print(f"  {DST_DATA}/events.jsonl + events_subset.jsonl")
    print(f"  {DST_DATA}/reference_toy.json - 3 reference cases")
    print(
        "\nIMPORTANT: Review all generated agent_answer texts manually - edit flaws to be obvious, fix city/category mismatches, ensure good answers truly good."
    )
    print(
        "Then bake into Docker: files already under data/ will be COPY'd via Dockerfile."
    )


if __name__ == "__main__":
    main()

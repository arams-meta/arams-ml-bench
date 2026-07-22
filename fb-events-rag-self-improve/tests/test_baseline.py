"""
Baseline floor: beats trivial popularity heuristic that ignores facet.
Baseline filters by city + category + time window + geo radius + dedup + not spam, then sorts by popularity only (no facet/semantic).
Since relevance now requires facet match, this baseline will have low recall (~0.2-0.3) and improved (filter+facet+semantic) will beat it.
"""

import json
import math
from datetime import datetime


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def recall_at_k(retrieved, relevant, k=10):
    if not relevant:
        return None
    hits = len(set(retrieved[:k]) & set(relevant))
    return hits / len(relevant)


def trivial_baseline():
    # Baseline: filter by city + category + time window + geo radius + dedup + not spam, sort by popularity only (no facet/semantic)
    events = load_jsonl("/app/data/events.jsonl")
    queries = load_jsonl("/app/data/queries.jsonl")
    relevance = {
        r["query_id"]: r["relevant_event_ids"]
        for r in load_jsonl("/app/data/relevance.jsonl")
    }

    # Pre-filter valid events
    cutoff = datetime(2026, 2, 1)
    valid_events = [
        e
        for e in events
        if not e.get("is_spam") and datetime.fromisoformat(e["start_time"]) >= cutoff
    ]
    from collections import defaultdict

    events_by_city = defaultdict(list)
    for e in valid_events:
        events_by_city[e["city"]].append(e)

    recalls = []
    for q in queries:
        qid = q["id"]
        rel = relevance.get(qid, [])
        if not rel:
            continue
        ws = datetime.fromisoformat(q["time_window_start"])
        we = datetime.fromisoformat(q["time_window_end"])
        q_lat, q_lng = q["lat"], q["lng"]
        radius = q["radius_km"]
        city = q["city"]
        cat = q.get("category")

        cand = events_by_city.get(city, [])
        # city already matched, filter category, time, geo
        filtered = []
        for e in cand:
            if cat and e["category"] != cat:
                continue
            ev_start = datetime.fromisoformat(e["start_time"])
            if not (ws <= ev_start <= we):
                continue
            if haversine(q_lat, q_lng, e["lat"], e["lng"]) > radius:
                continue
            filtered.append(e)

        # dedup via cluster_id keeping highest popularity
        cluster_best = {}
        singletons = []
        for e in filtered:
            cid = e.get("cluster_id")
            if cid:
                if (
                    cid not in cluster_best
                    or e["popularity"] > cluster_best[cid]["popularity"]
                ):
                    cluster_best[cid] = e
            else:
                singletons.append(e)
        deduped = list(cluster_best.values()) + singletons
        # sort by popularity only
        sorted_by_pop = sorted(deduped, key=lambda x: x["popularity"], reverse=True)
        retrieved = [e["id"] for e in sorted_by_pop[:10]]
        r = recall_at_k(retrieved, rel, k=10)
        if r is not None:
            recalls.append(r)
    return sum(recalls) / len(recalls) if recalls else 0.0


def test_beats_trivial():
    import os, json

    # Agent's improved recall
    with open("/output/eval_report.json") as f:
        report = json.load(f)
    improved = report.get("improved_recall_at_10") or report.get("recall_at_10") or 0.0

    base = trivial_baseline()
    # Log
    os.makedirs("/logs/verifier", exist_ok=True)
    with open("/logs/verifier/baseline_score.txt", "w") as bf:
        bf.write(f"agent={improved:.3f} baseline_popularity={base:.3f}\n")

    assert improved > base + 0.05, (
        f"Agent {improved:.3f} should beat trivial popularity baseline {base:.3f} by 0.05"
    )

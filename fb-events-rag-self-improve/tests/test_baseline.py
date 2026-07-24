"""
Baseline floor: beats trivial popularity heuristic that ignores facet.
Baseline filters by city + category + time window + geo radius + dedup + not spam, then sorts by popularity only (no facet/semantic).
Since relevance now requires facet match, this baseline will have low recall (~0.2-0.3) and improved (filter+facet+semantic) will beat it.
"""

import json
import math
import os
from datetime import datetime
from collections import defaultdict


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


def dedup_by_cluster_ids(rel_ids, ev_map):
    cluster_best = {}
    single = []
    for eid in rel_ids:
        ev = ev_map.get(eid)
        if not ev:
            continue
        cid = ev.get("cluster_id")
        if cid:
            pop = ev.get("popularity", 0)
            if cid not in cluster_best or pop > cluster_best[cid][0]:
                cluster_best[cid] = (pop, eid)
        else:
            single.append(eid)
    deduped = [v[1] for v in cluster_best.values()] + single
    deduped_sorted = sorted(
        deduped, key=lambda eid: ev_map.get(eid, {}).get("popularity", 0), reverse=True
    )
    return deduped_sorted[:30]


def compute_relevance_for_query(q, valid_events, ev_map, by_city):
    q_lat = q["lat"]
    q_lng = q["lng"]
    ws = datetime.fromisoformat(q["time_window_start"])
    we = datetime.fromisoformat(q["time_window_end"])
    facet = q.get("facet")
    city = q["city"]
    cat = q.get("category")
    radius = q["radius_km"]
    rel = []
    cands = by_city.get(city, [])
    for ev in cands:
        if cat and ev["category"] != cat:
            continue
        if facet and ev.get("topic") != facet:
            continue
        ev_start = datetime.fromisoformat(ev["start_time"])
        if not (ws <= ev_start <= we):
            continue
        if haversine(q_lat, q_lng, ev["lat"], ev["lng"]) > radius:
            continue
        rel.append(ev["id"])
    return dedup_by_cluster_ids(rel, ev_map)


def trivial_baseline():
    events = load_jsonl("/app/data/events.jsonl")
    queries = load_jsonl("/app/data/queries.jsonl")

    # Try multiple relevance locations, fallback to recompute (avoids info-leakage requirement)
    relevance = {}
    for rp in [
        "/opt/eval/relevance.jsonl",
        "/app/data/relevance.jsonl",
        "/tests/data/relevance.jsonl",
    ]:
        if os.path.exists(rp):
            relevance = {r["query_id"]: r["relevant_event_ids"] for r in load_jsonl(rp)}
            break

    cutoff = datetime(2026, 2, 1)
    valid_events = [
        e
        for e in events
        if not e.get("is_spam") and datetime.fromisoformat(e["start_time"]) >= cutoff
    ]
    ev_map = {e["id"]: e for e in valid_events}
    events_by_city = defaultdict(list)
    for e in valid_events:
        events_by_city[e["city"]].append(e)

    # If no baked relevance, recompute on-the-fly (facet rule)
    recompute = not relevance
    if recompute:
        for q in queries:
            relevance[q["id"]] = compute_relevance_for_query(
                q, valid_events, ev_map, events_by_city
            )

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
        sorted_by_pop = sorted(deduped, key=lambda x: x["popularity"], reverse=True)
        retrieved = [e["id"] for e in sorted_by_pop[:10]]
        r = recall_at_k(retrieved, rel, k=10)
        if r is not None:
            recalls.append(r)
    return sum(recalls) / len(recalls) if recalls else 0.0


def test_beats_trivial():
    import os, json

    with open("/output/eval_report.json") as f:
        report = json.load(f)
    improved = report.get("improved_recall_at_10") or report.get("recall_at_10") or 0.0

    base = trivial_baseline()
    os.makedirs("/logs/verifier", exist_ok=True)
    with open("/logs/verifier/baseline_score.txt", "w") as bf:
        bf.write(f"agent={improved:.3f} baseline_popularity={base:.3f}\n")

    assert improved > base + 0.05, (
        f"Agent {improved:.3f} should beat trivial popularity baseline {base:.3f} by 0.05"
    )

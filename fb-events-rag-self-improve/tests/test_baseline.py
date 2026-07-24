"""
Baseline floor: beats trivial popularity heuristic that ignores facet.
Per TBR feedback: recompute improved recall from retrieved.jsonl (not trusting eval_report), and load ground truth from hidden /opt/eval to avoid agent tampering.
Baseline filters by city + category + time window + geo radius + dedup + not spam, then sorts by popularity only (no facet/semantic).
"""

import json
import math
import os
from datetime import datetime
from collections import defaultdict


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


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


def load_ground_truth():
    # Hidden ground truth first to prevent cheating via /app/data tampering
    events_paths = ["/opt/eval/events.jsonl", "/app/data/events.jsonl"]
    queries_paths = ["/opt/eval/queries.jsonl", "/app/data/queries.jsonl"]
    rel_paths = [
        "/opt/eval/relevance.jsonl",
        "/app/data/relevance.jsonl",
        "/tests/data/relevance.jsonl",
    ]

    events_file = first_existing(events_paths) or events_paths[-1]
    queries_file = first_existing(queries_paths) or queries_paths[-1]

    events = [json.loads(l) for l in open(events_file)]
    queries = [json.loads(l) for l in open(queries_file)]

    relevance = {}
    for rp in rel_paths:
        if os.path.exists(rp):
            relevance = {r["query_id"]: r["relevant_event_ids"] for r in load_jsonl(rp)}
            break

    # If no baked relevance (to avoid leakage), recompute
    if not relevance:
        cutoff = datetime(2026, 2, 1)
        valid_events = [
            e
            for e in events
            if not e.get("is_spam")
            and datetime.fromisoformat(e["start_time"]) >= cutoff
        ]
        ev_map = {e["id"]: e for e in valid_events}
        by_city = defaultdict(list)
        for e in valid_events:
            by_city[e["city"]].append(e)
        for q in queries:
            # queries from /opt/eval have full fields, from /app/data minimal may lack city/facet - skip if lacking
            if "city" not in q or "facet" not in q:
                continue
            relevance[q["id"]] = compute_relevance_for_query(
                q, valid_events, ev_map, by_city
            )

    return events, queries, relevance


def trivial_baseline():
    events, queries, relevance = load_ground_truth()

    cutoff = datetime(2026, 2, 1)
    valid_events = [
        e
        for e in events
        if not e.get("is_spam") and datetime.fromisoformat(e["start_time"]) >= cutoff
    ]
    ev_map = {e["id"]: e for e in valid_events}
    by_city = defaultdict(list)
    for e in valid_events:
        by_city[e["city"]].append(e)

    # If relevance still empty (because agent-visible queries lack facet), try hidden queries for baseline calc
    if not relevance:
        # fallback: load hidden queries that have facet
        hidden_q_path = "/opt/eval/queries.jsonl"
        if os.path.exists(hidden_q_path):
            queries = [json.loads(l) for l in open(hidden_q_path)]
            for q in queries:
                relevance[q["id"]] = compute_relevance_for_query(
                    q, valid_events, ev_map, by_city
                )

    recalls = []
    for q in queries:
        qid = q.get("id")
        if not qid:
            continue
        rel = relevance.get(qid, [])
        if not rel:
            continue
        # Need full query fields for filtering, skip if minimal
        if "city" not in q or "time_window_start" not in q:
            continue
        ws = datetime.fromisoformat(q["time_window_start"])
        we = datetime.fromisoformat(q["time_window_end"])
        q_lat, q_lng = q["lat"], q["lng"]
        radius = q["radius_km"]
        city = q["city"]
        cat = q.get("category")

        cand = by_city.get(city, [])
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


def compute_improved_recall_from_output():
    """Recompute improved recall from retrieved.jsonl vs hidden relevance, not trusting eval_report"""
    # Load hidden relevance
    rel_paths = ["/opt/eval/relevance.jsonl", "/app/data/relevance.jsonl"]
    relevance = {}
    for rp in rel_paths:
        if os.path.exists(rp):
            relevance = {
                r["query_id"]: set(r["relevant_event_ids"]) for r in load_jsonl(rp)
            }
            break

    # If no baked relevance, recompute
    if not relevance:
        events, queries, _ = load_ground_truth()
        cutoff = datetime(2026, 2, 1)
        valid = [
            e
            for e in events
            if not e.get("is_spam")
            and datetime.fromisoformat(e["start_time"]) >= cutoff
        ]
        ev_map = {e["id"]: e for e in valid}
        by_city = defaultdict(list)
        for e in valid:
            by_city[e["city"]].append(e)
        for q in queries:
            if "city" not in q:
                continue
            relevance[q["id"]] = set(
                compute_relevance_for_query(q, valid, ev_map, by_city)
            )
    else:
        # already set as set
        pass

    # Ensure relevance values are sets for recall calc, but handle both list/set
    relevance_sets = {}
    for k, v in relevance.items():
        relevance_sets[k] = set(v) if not isinstance(v, set) else v

    # Load retrieved
    retrieved = {}
    if os.path.exists("/output/retrieved.jsonl"):
        with open("/output/retrieved.jsonl") as f:
            for line in f:
                obj = json.loads(line)
                retrieved[obj["query_id"]] = obj.get("retrieved_ids", [])

    recalls = []
    for qid, rel in relevance_sets.items():
        if not rel:
            continue
        ret = retrieved.get(qid, [])
        if not ret:
            recalls.append(0.0)
            continue
        hits = len(set(ret[:10]) & rel)
        recalls.append(hits / len(rel))
    return sum(recalls) / len(recalls) if recalls else 0.0


def test_beats_trivial():
    improved = compute_improved_recall_from_output()
    base = trivial_baseline()
    os.makedirs("/logs/verifier", exist_ok=True)
    with open("/logs/verifier/baseline_score.txt", "w") as bf:
        bf.write(f"agent={improved:.3f} baseline_popularity={base:.3f}\n")
    assert improved > base + 0.05, (
        f"Agent {improved:.3f} should beat trivial popularity baseline {base:.3f} by 0.05"
    )

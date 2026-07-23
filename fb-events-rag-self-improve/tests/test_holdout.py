"""
Holdout generalization: re-evaluate on hidden holdout queries (future time windows).
"""

import json
import os
import math
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


def test_holdout_generalization():
    holdout_rel_candidates = [
        "/opt/eval/holdout_relevance.jsonl",
        "/app/data/holdout_relevance.jsonl",
        "/tests/data/holdout_relevance.jsonl",
    ]
    holdout_rel_path = None
    for p in holdout_rel_candidates:
        if os.path.exists(p):
            holdout_rel_path = p
            break

    # Load report for holdout recall if present
    with open("/output/eval_report.json") as f:
        report = json.load(f)

    holdout_recall = report.get("holdout_recall_at_10") or report.get("holdout_recall")
    train_recall = (
        report.get("improved_recall_at_10") or report.get("recall_at_10") or 0.0
    )

    if holdout_recall is not None:
        ratio = holdout_recall / max(train_recall, 1e-6)
        os.makedirs("/logs/verifier", exist_ok=True)
        with open("/logs/verifier/holdout_score.txt", "w") as hf:
            hf.write(f"{holdout_recall}\n")
        with open("/logs/verifier/holdout_ratio.txt", "w") as rf:
            rf.write(f"{ratio:.3f}\n")
        assert ratio >= 0.8, (
            f"Holdout degraded too much: holdout {holdout_recall:.3f} / train {train_recall:.3f} = {ratio:.3f} < 0.8"
        )
        return

    if holdout_rel_path is None:
        # No holdout data, try to recompute if holdout queries exist
        holdout_q_path = "/app/data/holdout_queries.jsonl"
        events_path = "/app/data/events.jsonl"
        if not os.path.exists(holdout_q_path) or not os.path.exists(events_path):
            return
        # recompute relevance on-the-fly
        events = load_jsonl(events_path)
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
        h_queries = load_jsonl(holdout_q_path)
        rel_map = {
            q["id"]: set(compute_relevance_for_query(q, valid_events, ev_map, by_city))
            for q in h_queries
        }
        holdout_rel_path = None  # we have rel_map computed
    else:
        rel_map = {
            r["query_id"]: set(r["relevant_event_ids"])
            for r in load_jsonl(holdout_rel_path)
        }

    retrieved_path = "/output/holdout_retrieved.jsonl"
    if not os.path.exists(retrieved_path):
        return

    # If rel_map not yet defined because we had path, ensure defined
    if "rel_map" not in locals():
        rel_map = {
            r["query_id"]: set(r["relevant_event_ids"])
            for r in load_jsonl(holdout_rel_path)
        }

    retrieved = {}
    with open(retrieved_path) as f:
        for line in f:
            obj = json.loads(line)
            retrieved[obj["query_id"]] = obj.get("retrieved_ids", [])

    recalls = []
    for qid, rel_ids in rel_map.items():
        if not rel_ids:
            continue
        ret = retrieved.get(qid, [])
        hits = len(set(ret[:10]) & rel_ids)
        recalls.append(hits / len(rel_ids))

    if not recalls:
        return

    h_score = sum(recalls) / len(recalls)
    with open("/logs/verifier/holdout_score.txt", "w") as hf:
        hf.write(f"{h_score}\n")
    with open("/logs/verifier/holdout_ratio.txt", "w") as rf:
        rf.write(f"{h_score / max(train_recall, 1e-6):.3f}\n")

    assert h_score >= 0.45, f"Holdout recall {h_score:.3f} too low (<0.45)"

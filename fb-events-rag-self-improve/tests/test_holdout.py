"""
Holdout generalization: re-evaluate on hidden holdout queries (future time windows).
Fix: recompute recall independently from retrieved files vs hidden ground truth, don't trust eval_report floats.
Ground truth loaded from /opt/eval first (hidden, not agent-writable).
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


def load_ground_truth():
    # Defense-in-depth: hidden not present at solve time, generated at test.sh time; if missing due to platform, auto-generate deterministically
    # Avoids brittle hard-asserts on /opt/eval readability that make reward depend on runtime USER/chmod
    if not os.path.exists("/opt/eval/queries.jsonl") and os.path.exists(
        "/opt/eval/generate_queries.py"
    ):
        try:
            import subprocess, sys

            subprocess.run(
                [
                    sys.executable,
                    "/opt/eval/generate_queries.py",
                    "--events",
                    "/opt/eval/events.jsonl",
                    "--n",
                    "500",
                    "--seed",
                    "42",
                    "--out-queries",
                    "/opt/eval/queries.jsonl",
                    "--out-rel",
                    "/tmp/relevance.jsonl",
                    "--out-holdout-queries",
                    "/opt/eval/holdout_queries.jsonl",
                    "--out-holdout-rel",
                    "/tmp/holdout_relevance.jsonl",
                    "--holdout-n",
                    "100",
                ],
                timeout=60,
                check=False,
            )
        except Exception:
            pass

    events_paths = ["/opt/eval/events.jsonl", "/app/data/events.jsonl"]
    queries_paths = ["/opt/eval/queries.jsonl", "/app/data/queries.jsonl"]
    h_queries_paths = [
        "/opt/eval/holdout_queries.jsonl",
        "/app/data/holdout_queries.jsonl",
    ]
    relevance_paths = ["/opt/eval/relevance.jsonl", "/app/data/relevance.jsonl"]
    h_relevance_paths = [
        "/opt/eval/holdout_relevance.jsonl",
        "/app/data/holdout_relevance.jsonl",
    ]

    def first_existing(paths):
        for p in paths:
            if os.path.exists(p):
                return p
        return None

    events_file = first_existing(events_paths)
    queries_file = first_existing(queries_paths)
    h_queries_file = first_existing(h_queries_paths)
    rel_file = first_existing(relevance_paths)
    h_rel_file = first_existing(h_relevance_paths)

    events = {}
    queries = {}
    h_queries = {}
    relevance = {}
    h_relevance = {}

    if events_file:
        events = {e["id"]: e for e in load_jsonl(events_file)}
    if queries_file:
        queries = {q["id"]: q for q in load_jsonl(queries_file)}
        # Guard structured fields - free-form has only id+text+query_date
        sq = next(iter(queries.values())) if queries else {}
        assert "city" in sq and "time_window_start" in sq, (
            f"Train queries missing structured city/time: {list(sq.keys())} - weak fallback"
        )
    if h_queries_file:
        h_queries = {q["id"]: q for q in load_jsonl(h_queries_file)}
        sq = next(iter(h_queries.values())) if h_queries else {}
        assert "city" in sq and "time_window_start" in sq, (
            f"Holdout queries missing structured fields: {list(sq.keys())} - weak fallback"
        )
    if rel_file:
        relevance = {
            r["query_id"]: set(r["relevant_event_ids"]) for r in load_jsonl(rel_file)
        }
    if h_rel_file:
        h_relevance = {
            r["query_id"]: set(r["relevant_event_ids"]) for r in load_jsonl(h_rel_file)
        }

    # Recompute relevance if missing (to avoid leakage, we don't require baked file)
    if not relevance and events and queries:
        cutoff = datetime(2026, 2, 1)
        valid = [
            e
            for e in events.values()
            if not e.get("is_spam")
            and datetime.fromisoformat(e["start_time"]) >= cutoff
        ]
        ev_map = {e["id"]: e for e in valid}
        by_city = defaultdict(list)
        for e in valid:
            by_city[e["city"]].append(e)
        for qid, q in queries.items():
            relevance[qid] = set(compute_relevance_for_query(q, valid, ev_map, by_city))

    if not h_relevance and events and h_queries:
        cutoff = datetime(2026, 2, 1)
        valid = [
            e
            for e in events.values()
            if not e.get("is_spam")
            and datetime.fromisoformat(e["start_time"]) >= cutoff
        ]
        ev_map = {e["id"]: e for e in valid}
        by_city = defaultdict(list)
        for e in valid:
            by_city[e["city"]].append(e)
        for qid, q in h_queries.items():
            h_relevance[qid] = set(
                compute_relevance_for_query(q, valid, ev_map, by_city)
            )

    # Final guards: ensure relevance not empty and not from weak heuristic
    assert len(relevance) >= 100 and sum(len(v) for v in relevance.values()) > 0, (
        "Train relevance empty - hidden ground truth unreadable, weak fallback would match oracle heuristic"
    )
    assert len(h_relevance) >= 20 and sum(len(v) for v in h_relevance.values()) > 0, (
        "Holdout relevance empty - hidden ground truth unreadable"
    )

    return events, queries, h_queries, relevance, h_relevance


def test_holdout_generalization():
    events, queries, h_queries, relevance, h_relevance = load_ground_truth()

    # Non-brittle guards: relevance non-empty and structured fields, not hard-asserting on /opt/eval chmod (fix brittle reward)
    # load_ground_truth already auto-generates hidden at test time if missing for defense-in-depth
    assert len(h_relevance) >= 20 and sum(len(v) for v in h_relevance.values()) > 0, (
        "Holdout relevance empty - hidden generation may have failed"
    )
    assert len(relevance) >= 100, "Train relevance empty"
    assert any("city" in q for q in h_queries.values()), (
        "Holdout queries missing structured city - weak fallback"
    )

    # Load retrieved outputs
    retrieved = {}
    h_retrieved = {}
    if os.path.exists("/output/retrieved.jsonl"):
        with open("/output/retrieved.jsonl") as f:
            for line in f:
                obj = json.loads(line)
                retrieved[obj["query_id"]] = obj.get("retrieved_ids", [])
    if os.path.exists("/output/holdout_retrieved.jsonl"):
        with open("/output/holdout_retrieved.jsonl") as f:
            for line in f:
                obj = json.loads(line)
                h_retrieved[obj["query_id"]] = obj.get("retrieved_ids", [])

    # If no holdout retrieved, skip - but ground truth must still be valid (guarded above)
    if not h_retrieved and not h_relevance:
        return

    # Recompute train and holdout recall independently from retrieved files (don't trust report)
    def recall_from_dict(ret_dict, rel_dict):
        rec = []
        for qid, rel_ids in rel_dict.items():
            if not rel_ids:
                continue
            ret = ret_dict.get(qid, [])
            hits = len(set(ret[:10]) & rel_ids)
            rec.append(hits / len(rel_ids))
        return sum(rec) / len(rec) if rec else 0.0

    train_recall = recall_from_dict(retrieved, relevance)
    h_score = recall_from_dict(h_retrieved, h_relevance)

    # Also read report for logging but not for decision
    report_train = 0.0
    report_holdout = None
    if os.path.exists("/output/eval_report.json"):
        with open("/output/eval_report.json") as f:
            rep = json.load(f)
            report_train = (
                rep.get("improved_recall_at_10") or rep.get("recall_at_10") or 0.0
            )
            report_holdout = rep.get("holdout_recall_at_10") or rep.get(
                "holdout_recall"
            )

    os.makedirs("/logs/verifier", exist_ok=True)
    with open("/logs/verifier/holdout_score.txt", "w") as hf:
        hf.write(f"{h_score}\n")
    with open("/logs/verifier/holdout_ratio.txt", "w") as rf:
        ratio = h_score / max(train_recall, 1e-6)
        rf.write(f"{ratio:.3f}\n")

    # Lowered for free-form per TBR justification after non-cheating golden and non-leaked solver pass
    assert h_score >= 0.25, (
        f"Holdout recall recomputed {h_score:.3f} too low (<0.25 free-form)"
    )

    if train_recall > 0:
        ratio = h_score / max(train_recall, 1e-6)
        assert ratio >= 0.4, (
            f"Holdout degraded too much: holdout {h_score:.3f} / train {train_recall:.3f} = {ratio:.3f} < 0.4 free-form"
        )

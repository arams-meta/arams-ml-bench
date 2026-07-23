"""
Main scoring for self-improving RAG.
Fixes HIGH issues: recompute recall independently from retrieved.jsonl vs hidden ground truth, don't trust eval_report floats.
Ground truth loaded from hidden /opt/eval first (not agent-writable /app/data), fallback to /app/data for local runs.
"""

import json
import os
import math
from datetime import datetime
from collections import defaultdict


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_ground_truth():
    """Load events/queries/relevance from hidden /opt/eval first to avoid agent tampering of /app/data"""
    events_paths = ["/opt/eval/events.jsonl", "/app/data/events.jsonl"]
    queries_paths = ["/opt/eval/queries.jsonl", "/app/data/queries.jsonl"]
    relevance_paths = [
        "/opt/eval/relevance.jsonl",
        "/app/data/relevance.jsonl",
        "/tests/data/relevance.jsonl",
    ]

    events_file = next((p for p in events_paths if os.path.exists(p)), events_paths[-1])
    queries_file = next(
        (p for p in queries_paths if os.path.exists(p)), queries_paths[-1]
    )

    events = {e["id"]: e for e in load_jsonl(events_file)}
    queries = {q["id"]: q for q in load_jsonl(queries_file)}

    relevance = {}
    for rp in relevance_paths:
        if os.path.exists(rp):
            relevance = {r["query_id"]: r["relevant_event_ids"] for r in load_jsonl(rp)}
            break

    # If no baked relevance (to avoid leakage, we may not have it in /app), recompute using deterministic rule
    if not relevance:
        # recompute using same rule as generator
        def dedup_ids(ids, ev_map):
            cluster_best = {}
            single = []
            for eid in ids:
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
                deduped,
                key=lambda eid: ev_map.get(eid, {}).get("popularity", 0),
                reverse=True,
            )
            return deduped_sorted[:30]

        def haversine_local(lat1, lon1, lat2, lon2):
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
            ws = datetime.fromisoformat(q["time_window_start"])
            we = datetime.fromisoformat(q["time_window_end"])
            rel = []
            for ev in by_city.get(q["city"], []):
                if q.get("category") and ev["category"] != q.get("category"):
                    continue
                if q.get("facet") and ev.get("topic") != q.get("facet"):
                    continue
                ev_start = datetime.fromisoformat(ev["start_time"])
                if not (ws <= ev_start <= we):
                    continue
                if (
                    haversine_local(q["lat"], q["lng"], ev["lat"], ev["lng"])
                    > q["radius_km"]
                ):
                    continue
                rel.append(ev["id"])
            relevance[qid] = dedup_ids(rel, ev_map)

    return events, queries, relevance


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


def recall_at_k(retrieved_ids, relevant_ids, k=10):
    if not relevant_ids:
        return None
    hits = len(set(retrieved_ids[:k]) & set(relevant_ids))
    return hits / len(relevant_ids)


def load_agent_output():
    retrieved_path = "/output/retrieved.jsonl"
    eval_report_path = "/output/eval_report.json"
    retrieved = {}
    if os.path.exists(retrieved_path):
        with open(retrieved_path) as f:
            for line in f:
                obj = json.loads(line)
                if "query_id" in obj:
                    retrieved[obj["query_id"]] = (
                        obj.get("retrieved_ids") or obj.get("event_ids") or []
                    )
                else:
                    retrieved.update(obj)
    report = {}
    if os.path.exists(eval_report_path):
        with open(eval_report_path) as f:
            report = json.load(f)
    return retrieved, report


def compute_baseline_recall(events, queries, relevance):
    """Trivial popularity baseline: filter city+category+time+geo+dedup+spam, sort pop (no facet check? Actually baseline for lift should ignore facet to show lift)"""
    # For lift test, baseline should be facet-blind popularity to demonstrate improvement
    # But we compute baseline that IGNORES facet (as in original trivial baseline) vs improved that uses facet
    cutoff = datetime(2026, 2, 1)
    valid = [
        e
        for e in events.values()
        if not e.get("is_spam") and datetime.fromisoformat(e["start_time"]) >= cutoff
    ]
    by_city = defaultdict(list)
    for e in valid:
        by_city[e["city"]].append(e)

    recalls = []
    for qid, q in queries.items():
        rel = relevance.get(qid, [])
        if not rel:
            continue
        ws = datetime.fromisoformat(q["time_window_start"])
        we = datetime.fromisoformat(q["time_window_end"])
        filtered = []
        for ev in by_city.get(q["city"], []):
            if q.get("category") and ev["category"] != q.get("category"):
                continue
            # NOTE: baseline ignores facet to show low recall
            ev_start = datetime.fromisoformat(ev["start_time"])
            if not (ws <= ev_start <= we):
                continue
            if haversine(q["lat"], q["lng"], ev["lat"], ev["lng"]) > q["radius_km"]:
                continue
            filtered.append(ev)
        # dedup
        cluster_best = {}
        singles = []
        for ev in filtered:
            cid = ev.get("cluster_id")
            if cid:
                if (
                    cid not in cluster_best
                    or ev["popularity"] > cluster_best[cid]["popularity"]
                ):
                    cluster_best[cid] = ev
            else:
                singles.append(ev)
        deduped = list(cluster_best.values()) + singles
        sorted_pop = sorted(deduped, key=lambda x: x["popularity"], reverse=True)
        retrieved = [e["id"] for e in sorted_pop[:10]]
        r = recall_at_k(retrieved, rel, k=10)
        if r is not None:
            recalls.append(r)
    return sum(recalls) / len(recalls) if recalls else 0.0


def compute_improved_recall_from_output(events, queries, relevance):
    retrieved, _ = load_agent_output()
    recalls = []
    for qid, rel in relevance.items():
        if not rel:
            continue
        ret = retrieved.get(qid, [])
        if not ret:
            recalls.append(0.0)
            continue
        recalls.append(recall_at_k(ret, rel, k=10) or 0.0)
    return sum(recalls) / len(recalls) if recalls else 0.0, retrieved


def test_self_improvement_lift():
    """Directional: improved recall must beat baseline by margin - recomputed independently, not trusting report"""
    events, queries, relevance = load_ground_truth()
    baseline_recall = compute_baseline_recall(events, queries, relevance)
    improved_recall, _ = compute_improved_recall_from_output(events, queries, relevance)

    # Also read report for informational, but don't trust for scoring
    _, report = load_agent_output()
    report_improved = (
        report.get("improved_recall_at_10") or report.get("recall_at_10") or 0.0
    )
    # Ensure report not wildly fabricating vs recomputed (allow 0.15 diff for different baseline defs)
    # We don't fail on fabrication, we use recomputed for lift

    assert improved_recall > baseline_recall + 0.08, (
        f"Expected lift (recomputed): improved {improved_recall:.3f} > baseline {baseline_recall:.3f} + 0.08"
    )


def test_recall_absolute_threshold():
    events, queries, relevance = load_ground_truth()
    improved_recall, _ = compute_improved_recall_from_output(events, queries, relevance)
    assert improved_recall >= 0.60, (
        f"Improved recall@10 recomputed {improved_recall:.3f} below absolute threshold 0.60"
    )


def test_constraint_time_filter():
    _, report = load_agent_output()
    retrieved, _ = load_agent_output()
    events, queries, _ = load_ground_truth()

    if not retrieved:
        assert (
            report.get("time_filter_pass_rate", 1.0) >= 0.95
            or "constraint" in json.dumps(report).lower()
        )
        return

    violations = 0
    total = 0
    for qid, r_ids in retrieved.items():
        if qid not in queries:
            continue
        q = queries[qid]
        ws = datetime.fromisoformat(q["time_window_start"])
        we = datetime.fromisoformat(q["time_window_end"])
        for eid in r_ids[:10]:
            if eid not in events:
                continue
            ev = events[eid]
            ev_start = datetime.fromisoformat(ev["start_time"])
            total += 1
            if not (ws <= ev_start <= we):
                violations += 1

    if total == 0:
        return
    violation_rate = violations / total
    assert violation_rate <= 0.10, (
        f"Time window violation rate {violation_rate:.2%} > 10% ({violations}/{total})"
    )


def test_constraint_geo_filter():
    retrieved, _ = load_agent_output()
    if not retrieved:
        return
    events, queries, _ = load_ground_truth()
    violations = 0
    total = 0
    for qid, r_ids in retrieved.items():
        if qid not in queries:
            continue
        q = queries[qid]
        q_lat, q_lng = q["lat"], q["lng"]
        radius = q["radius_km"]
        for eid in r_ids[:10]:
            if eid not in events:
                continue
            ev = events[eid]
            d = haversine(q_lat, q_lng, ev["lat"], ev["lng"])
            total += 1
            if d > radius + 1e-6:
                violations += 1

    if total == 0:
        return
    rate = violations / total
    assert rate <= 0.15, f"Geo radius violation {rate:.2%} > 15%"


def test_dedup_handling():
    retrieved, _ = load_agent_output()
    if not retrieved:
        return
    events, _, _ = load_ground_truth()
    dup_queries = 0
    for qid, r_ids in retrieved.items():
        seen_clusters = {}
        for eid in r_ids[:10]:
            ev = events.get(eid)
            if not ev:
                continue
            cid = ev.get("cluster_id")
            if cid:
                if cid in seen_clusters:
                    dup_queries += 1
                    break
                seen_clusters[cid] = eid

    if len(retrieved) == 0:
        return
    dup_rate = dup_queries / len(retrieved)
    assert dup_rate <= 0.30, (
        f"Dedup failure: {dup_rate:.2%} queries contain duplicate cluster members in top10"
    )


def test_eval_system_exists():
    with open("/output/eval_report.json") as f:
        rep = json.load(f)
    required_concepts = ["baseline", "improved", "recall", "method"]
    blob = json.dumps(rep).lower()
    matches = sum(1 for c in required_concepts if c in blob)
    assert matches >= 3, (
        f"eval_report.json incomplete - should describe baseline vs improved and method, got {rep.keys()}"
    )

    with open("/output/rag_config.json") as f:
        cfg = json.load(f)
    blob_cfg = json.dumps(cfg).lower()
    assert any(
        k in blob_cfg
        for k in [
            "rewrite",
            "rerank",
            "hyde",
            "fine-tune",
            "finetune",
            "prompt",
            "filter",
            "dedup",
            "improve",
        ]
    ), "rag_config.json must describe improvement method"


def test_no_spam_retrieved():
    retrieved, _ = load_agent_output()
    if not retrieved:
        return
    events, _, _ = load_ground_truth()
    spam_hits = 0
    total = 0
    for r_ids in retrieved.values():
        for eid in r_ids[:10]:
            if eid not in events:
                continue
            total += 1
            if events[eid].get("is_spam"):
                spam_hits += 1
    if total == 0:
        return
    rate = spam_hits / total
    assert rate <= 0.05, f"Spam rate in top10 {rate:.2%} > 5%"


def test_reference_toy():
    retrieved, _ = load_agent_output()
    events, queries, relevance = load_ground_truth()
    improved_recall, _ = compute_improved_recall_from_output(events, queries, relevance)
    assert improved_recall >= 0.5, "Recall too low for non-trivial solution"

"""
Main scoring for self-improving RAG.
Fixes HIGH issues: recompute recall independently from retrieved.jsonl vs hidden ground truth, don't trust eval_report floats.
Ground truth loaded from hidden /opt/eval first (not agent-writable /app/data), fallback to /app/data for local runs.
Fixes Test Quality: no early return on empty input, holdout blocking, self-improvement loop explicitly tested.
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
    """Load events/queries/relevance from hidden /opt/eval first - robust to platform, auto-generates hidden at test time if missing for defense-in-depth"""
    # Defense-in-depth: hidden structured queries are NOT present at solve time (only events + generators), generated at test.sh time
    # If missing due to platform chmod behavior, try to generate deterministically (same seed 42) rather than hard-failing entire suite
    # This avoids brittle reward that depends on runtime USER enforcement, while still preventing weak free-form fallback
    if not os.path.exists("/opt/eval/queries.jsonl") and os.path.exists(
        "/opt/eval/generate_queries.py"
    ):
        # Try to generate hidden at test time (deterministic, seed 42)
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
    # Hidden queries should have structured fields (city, time_window) - if we get free-form, we are in weak fallback mode
    # Instead of hard-asserting on /opt/eval readability (brittle), check structured field presence and relevance non-empty
    sample_q = next(iter(queries.values())) if queries else {}
    # If structured fields missing, we are in agent-visible free-form fallback - recompute would be same heuristic as oracle (weak), so we treat as pipeline failure not weak label scoring
    # We keep this check but as informative, not as hard dependency on /opt/eval chmod
    if "city" not in sample_q or "time_window_start" not in sample_q:
        # Try one more time to generate hidden if generator exists
        if os.path.exists("/opt/eval/generate_queries.py"):
            try:
                import subprocess, sys

                subprocess.run(
                    [
                        sys.executable,
                        "/opt/eval/generate_queries.py",
                        "--events",
                        events_file,
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
                if os.path.exists("/opt/eval/queries.jsonl"):
                    queries = {
                        q["id"]: q for q in load_jsonl("/opt/eval/queries.jsonl")
                    }
                    sample_q = next(iter(queries.values())) if queries else {}
            except Exception:
                pass
    # Final check: must have structured fields to avoid weak heuristic scoring
    assert "city" in sample_q and "time_window_start" in sample_q, (
        f"Queries missing structured fields city/time_window: got {list(sample_q.keys())} - need hidden /opt/eval generated at test time"
    )

    relevance = {}
    for rp in relevance_paths:
        if os.path.exists(rp):
            relevance = {r["query_id"]: r["relevant_event_ids"] for r in load_jsonl(rp)}
            break

    if not relevance:

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

    # Guard relevance non-empty and from hidden structured queries
    assert len(relevance) >= 100 and sum(len(v) for v in relevance.values()) > 0, (
        "Relevance recomputed empty - hidden ground truth unreadable, weak fallback would be same heuristic as oracle"
    )

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
            ev_start = datetime.fromisoformat(ev["start_time"])
            if not (ws <= ev_start <= we):
                continue
            if haversine(q["lat"], q["lng"], ev["lat"], ev["lng"]) > q["radius_km"]:
                continue
            filtered.append(ev)
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
    """Directional: improved recall must beat baseline by margin - recomputed independently from hidden ground truth"""
    events, queries, relevance = load_ground_truth()
    # Relevance and baseline checks are done inside load_ground_truth via structured field presence, not hard-asserting on /opt/eval chmod (brittle)
    # Keep only non-brittle checks: relevance non-empty and baseline>0
    assert len(relevance) >= 100 and sum(len(v) for v in relevance.values()) > 0, (
        "Relevance empty - cannot establish lift floor, hidden generation may have failed"
    )
    baseline_recall = compute_baseline_recall(events, queries, relevance)
    assert baseline_recall > 0.0, "Baseline recomputed as 0.0 - relevance empty"
    improved_recall, _ = compute_improved_recall_from_output(events, queries, relevance)
    assert improved_recall > baseline_recall + 0.08, (
        f"Expected lift (recomputed): improved {improved_recall:.3f} > baseline {baseline_recall:.3f} + 0.08"
    )


def test_recall_absolute_threshold():
    events, queries, relevance = load_ground_truth()
    assert len(relevance) >= 100, "Relevance empty - hidden generation may have failed"
    improved_recall, _ = compute_improved_recall_from_output(events, queries, relevance)
    assert improved_recall >= 0.40, (
        f"Improved recall@10 recomputed {improved_recall:.3f} below absolute threshold 0.40 (free-form, lowered per TBR justification)"
    )


def test_self_improvement_loop_executed():
    """Explicitly test that self-improvement loop artifacts exist - does not trust self-reported floats per spec, only checks existence"""
    retrieved, report = load_agent_output()
    assert retrieved, (
        "Missing /output/retrieved.jsonl - self-improvement loop must produce retrieval output"
    )
    # Check eval_report has lift and method indicating improvement - existence only, not trusting float per spec
    # Spec says grader RECOMPUTES recall from retrieved.jsonl, self-reported numbers are schema only
    assert "lift" in report or "baseline_recall_at_10" in report, (
        "eval_report must contain lift / baseline metrics from self-eval loop (existence, not float trust)"
    )
    assert "improved_recall_at_10" in report or "recall_at_10" in report, (
        "eval_report must contain improved recall field"
    )
    # Check rag_config declares self-improvement
    with open("/output/rag_config.json") as f:
        cfg = json.load(f)
    assert cfg.get("self_improvement") or "improvement_method" in cfg, (
        "rag_config must declare self-improvement"
    )
    blob = json.dumps(cfg).lower()
    # Must mention at least two improvement techniques
    techniques = ["filter", "rewrite", "rerank", "dedup", "facet", "semantic"]
    matches = sum(1 for t in techniques if t in blob)
    assert matches >= 2, (
        f"rag_config should describe at least 2 improvement techniques, got {matches}"
    )


def test_constraint_time_filter():
    retrieved, report = load_agent_output()
    events, queries, _ = load_ground_truth()
    assert retrieved, "Missing retrieved.jsonl - cannot test time filter"
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
    assert total > 0, "No retrieved IDs to check time filter"
    violation_rate = violations / total
    # Allow up to 40% for free-form extraction with inventory anchor (time phrase ambiguous per TBR)
    assert violation_rate <= 0.40, (
        f"Time window violation rate {violation_rate:.2%} > 40% ({violations}/{total})"
    )


def test_constraint_geo_filter():
    retrieved, _ = load_agent_output()
    events, queries, _ = load_ground_truth()
    assert retrieved, "Missing retrieved.jsonl - cannot test geo filter"
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
    assert total > 0, "No retrieved IDs to check geo filter"
    rate = violations / total
    assert rate <= 0.20, f"Geo radius violation {rate:.2%} > 20%"


def test_dedup_handling():
    retrieved, _ = load_agent_output()
    events, _, _ = load_ground_truth()
    assert retrieved, "Missing retrieved.jsonl - cannot test dedup"
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
    assert len(retrieved) > 0, "No retrieved queries"
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
    events, _, _ = load_ground_truth()
    assert retrieved, "Missing retrieved.jsonl - cannot test spam filter"
    spam_hits = 0
    total = 0
    for r_ids in retrieved.values():
        for eid in r_ids[:10]:
            if eid not in events:
                continue
            total += 1
            if events[eid].get("is_spam"):
                spam_hits += 1
    assert total > 0, "No retrieved IDs to check spam"
    rate = spam_hits / total
    assert rate <= 0.05, f"Spam rate in top10 {rate:.2%} > 5%"


def test_reference_toy():
    retrieved, _ = load_agent_output()
    events, queries, relevance = load_ground_truth()
    improved_recall, _ = compute_improved_recall_from_output(events, queries, relevance)
    assert retrieved, "Missing retrieved.jsonl"
    assert improved_recall >= 0.40, (
        f"Recall too low for non-trivial solution: {improved_recall:.3f} (need >=0.40 free-form, aligned with test_recall_absolute_threshold)"
    )

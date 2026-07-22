"""
Main scoring for self-improving RAG.
Research extension: naive baseline must fail, improved must pass directional + constraint checks.
"""

import json
import os
import math
from datetime import datetime
from collections import defaultdict


# Helpers
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


def recall_at_k(retrieved_ids, relevant_ids, k=10):
    if not relevant_ids:
        return None
    retrieved_k = retrieved_ids[:k]
    hits = len(set(retrieved_k) & set(relevant_ids))
    return hits / len(relevant_ids)


# Load agent artifacts
def load_agent_output():
    # Agent must write retrieval results for queries: /output/retrieved.jsonl = {query_id: [event_ids]}
    # Or eval_report contains metrics computed by agent's own eval loop
    retrieved_path = "/output/retrieved.jsonl"
    eval_report_path = "/output/eval_report.json"

    retrieved = {}
    if os.path.exists(retrieved_path):
        with open(retrieved_path) as f:
            for line in f:
                obj = json.loads(line)
                # support two formats: {"query_id":..., "retrieved_ids": [...]} or dict mapping
                if "query_id" in obj:
                    retrieved[obj["query_id"]] = (
                        obj.get("retrieved_ids") or obj.get("event_ids") or []
                    )
                else:
                    # maybe full dict mapping
                    retrieved.update(obj)

    with open(eval_report_path) as f:
        report = json.load(f)

    return retrieved, report


def test_self_improvement_lift():
    """Directional: improved recall must beat baseline by margin."""
    _, report = load_agent_output()
    # Support multiple key naming
    base = (
        report.get("baseline_recall_at_10")
        or report.get("baseline_recall")
        or report.get("recall_baseline_at_10")
        or 0.0
    )
    improved = (
        report.get("improved_recall_at_10")
        or report.get("recall_at_10")
        or report.get("recall_improved")
        or report.get("final_recall_at_10")
        or 0.0
    )

    # If report contains only single recall (agent claims it's improved), compare against embedded baseline file if present
    if base == 0.0 and "baseline_recall_at_10" not in report:
        # Try to load baseline computed by tests/data if exists, else require improved > 0.55 absolute
        assert improved > 0.55, (
            f"Improved recall {improved} must be >0.55 and report should contain baseline_recall_at_10"
        )
        return

    assert improved > base + 0.08, (
        f"Expected lift: improved {improved:.3f} > baseline {base:.3f} + 0.08 (self-improvement loop not effective)"
    )


def test_recall_absolute_threshold():
    _, report = load_agent_output()
    improved = (
        report.get("improved_recall_at_10")
        or report.get("recall_at_10")
        or report.get("final_recall_at_10")
        or 0.0
    )
    assert improved >= 0.60, (
        f"Improved recall@10 {improved:.3f} below absolute threshold 0.60"
    )


def test_constraint_time_filter():
    """Adherence to time window - no past/outside-window events."""
    retrieved, _ = load_agent_output()
    if not retrieved:
        # If agent didn't write retrieved.jsonl, skip but require eval_report to declare constraint handling
        with open("/output/eval_report.json") as f:
            rep = json.load(f)
        # Must have explicit check
        assert (
            rep.get("time_filter_pass_rate", 1.0) >= 0.95
            or "constraint" in json.dumps(rep).lower()
        )
        return

    # Load queries
    queries = {q["id"]: q for q in load_jsonl("/app/data/queries.jsonl")}
    events = {e["id"]: e for e in load_jsonl("/app/data/events.jsonl")}

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
    """Geo radius must be respected."""
    retrieved, _ = load_agent_output()
    if not retrieved:
        return

    queries = {q["id"]: q for q in load_jsonl("/app/data/queries.jsonl")}
    events = {e["id"]: e for e in load_jsonl("/app/data/events.jsonl")}

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
    """If agent returns near-duplicate cluster events in top-k, it should dedup to 1 per cluster."""
    retrieved, _ = load_agent_output()
    if not retrieved:
        return

    events = {e["id"]: e for e in load_jsonl("/app/data/events.jsonl")}

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

    # Allow some duplication but penalize heavy duplication
    # Fail if >30% of queries have duplicate clusters in top10
    if len(retrieved) == 0:
        return
    dup_rate = dup_queries / len(retrieved)
    assert dup_rate <= 0.30, (
        f"Dedup failure: {dup_rate:.2%} queries contain duplicate cluster members in top10"
    )


def test_eval_system_exists():
    """Agent must implement eval system, not just RAG."""
    with open("/output/eval_report.json") as f:
        rep = json.load(f)
    # Must contain methodology fields showing eval loop was run
    required_concepts = ["baseline", "improved", "recall", "method"]
    blob = json.dumps(rep).lower()
    matches = sum(1 for c in required_concepts if c in blob)
    assert matches >= 3, (
        f"eval_report.json seems incomplete - should describe baseline vs improved and method, got {rep.keys()}"
    )

    # Must declare improvement method details in rag_config
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
    ), (
        "rag_config.json must describe improvement method (rewrite, rerank, HyDE, fine-tune, etc.)"
    )


def test_no_spam_retrieved():
    """Spam events should not be in retrieved results."""
    retrieved, _ = load_agent_output()
    if not retrieved:
        return

    events = {e["id"]: e for e in load_jsonl("/app/data/events.jsonl")}
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
    """Naive approach discriminator: if retrieved == simple popularity sort, fail."""
    # This test ensures solution is not trivial baseline (popularity only)
    retrieved, report = load_agent_output()
    if not retrieved:
        # rely on recall lift already checked
        return
    improved = report.get("improved_recall_at_10") or report.get("recall_at_10") or 0
    # If recall is high but no self-improvement described, it's suspicious - but main lift test catches
    assert improved >= 0.5, "Recall too low for non-trivial solution"

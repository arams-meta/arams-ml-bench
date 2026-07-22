"""
Holdout generalization: re-evaluate on hidden holdout queries (future time windows).
"""

import json
import os
import math
from datetime import datetime


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def test_holdout_generalization():
    # Agent must produce retrieved for holdout as well OR eval_report contains holdout scores
    # We support both: if /output/holdout_retrieved.jsonl exists, score it; else check report fields
    holdout_rel_path = "/app/data/holdout_relevance.jsonl"
    if not os.path.exists(holdout_rel_path):
        return  # no holdout data baked

    # Try to get holdout recall from report
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

    # Fallback: try to score /output/holdout_retrieved.jsonl
    retrieved_path = "/output/holdout_retrieved.jsonl"
    if not os.path.exists(retrieved_path):
        # No holdout eval provided, skip as informational (per template, holdout non-blocking if implemented this way)
        # But we log that it was missing
        return

    # Score holdout retrieved vs relevance
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

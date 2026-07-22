import json
import os
from datetime import datetime


def test_events_schema():
    path = "/app/data/events.jsonl"
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            ev = json.loads(line)
            assert "id" in ev
            assert "title" in ev
            assert "description" in ev
            assert "category" in ev
            assert "city" in ev
            assert "lat" in ev and "lng" in ev
            assert "start_time" in ev
            # parse time
            datetime.fromisoformat(ev["start_time"])


def test_queries_schema():
    with open("/app/data/queries.jsonl") as f:
        q = json.loads(next(f))
    assert "id" in q
    assert "text" in q
    assert "city" in q
    assert "time_window_start" in q
    assert "radius_km" in q


def test_output_schema():
    with open("/output/rag_config.json") as f:
        cfg = json.load(f)
    # must declare improvement method
    assert "improvement_method" in cfg or "self_improvement" in cfg or "method" in cfg

    with open("/output/eval_report.json") as f:
        rep = json.load(f)
    # must have at least recall metrics
    assert any(
        k in rep
        for k in [
            "recall_at_10",
            "recall@10",
            "baseline_recall_at_10",
            "improved_recall_at_10",
            "recall_improved",
        ]
    )

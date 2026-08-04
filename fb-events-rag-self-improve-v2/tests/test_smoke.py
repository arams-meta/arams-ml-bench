import os
import json


def test_output_exists():
    # Agent must produce rag_config and eval_report
    assert os.path.exists("/output/rag_config.json"), "Missing /output/rag_config.json"
    assert os.path.exists("/output/eval_report.json"), (
        "Missing /output/eval_report.json"
    )


def test_output_loadable():
    with open("/output/rag_config.json") as f:
        cfg = json.load(f)
    assert isinstance(cfg, dict)
    assert "retriever" in cfg or "embedding_model" in cfg

    with open("/output/eval_report.json") as f:
        report = json.load(f)
    assert "baseline_recall_at_10" in report or "recall_at_10" in report


def test_data_baked():
    assert os.path.exists("/app/data/events.jsonl")
    assert os.path.exists("/app/data/queries.jsonl")

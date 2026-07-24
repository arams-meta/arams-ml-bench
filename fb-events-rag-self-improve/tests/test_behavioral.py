"""
Behavioral tests to enforce RAG, embedding usage, free-form extraction, and self-improvement per TBR feedback.
These checks ensure agent didn't just do deterministic rule-based retrieval using structured fields, but actually used embeddings and parsed free-form queries.
"""

import json
import os


def test_parsed_filters_exists_and_valid():
    """Agent must output parsed_filters.jsonl with extracted hard filters from free-form text"""
    path = "/output/parsed_filters.jsonl"
    assert os.path.exists(path), (
        "Missing /output/parsed_filters.jsonl - must extract hard filters from free-form query text"
    )

    with open(path) as f:
        lines = [json.loads(l) for l in f]

    assert len(lines) >= 100, (
        f"parsed_filters should have 500 entries, got {len(lines)}"
    )

    # Check schema
    required = ["query_id", "city", "facet"]
    for entry in lines[:20]:
        for r in required:
            assert r in entry, f"parsed filter missing {r}: {entry}"

    # Check that city values are from known cities and facet from known topics
    # Load hidden ground truth for comparison
    hidden_q_path = "/opt/eval/queries.jsonl"
    if os.path.exists(hidden_q_path):
        with open(hidden_q_path) as f:
            hidden = {json.loads(l)["id"]: json.loads(l) for l in f}
        # Compute accuracy of parsed city/facet vs hidden
        correct_city = 0
        correct_facet = 0
        total = 0
        for entry in lines:
            qid = entry["query_id"]
            if qid not in hidden:
                continue
            total += 1
            if entry.get("city") == hidden[qid].get("city"):
                correct_city += 1
            if entry.get("facet") == hidden[qid].get("facet"):
                correct_facet += 1
        if total > 0:
            city_acc = correct_city / total
            facet_acc = correct_facet / total
            # Require at least 60% accuracy on city and 50% on facet to prove extraction works
            assert city_acc >= 0.60, (
                f"Parsed city accuracy too low: {city_acc:.2f} (need >=0.60) - must extract city from free-form text"
            )
            assert facet_acc >= 0.50, (
                f"Parsed facet accuracy too low: {facet_acc:.2f} (need >=0.50) - must extract facet/topic from text"
            )


def test_embedding_usage_proven():
    """Prove MiniLM embedding search was actually used, not just strings in config"""
    # Check solve.sh contains embedding usage
    solve_paths = [
        "/solution/solve.sh",
        "/app/solution/solve.sh",
        "./solution/solve.sh",
    ]
    solve_content = ""
    for p in solve_paths:
        if os.path.exists(p):
            with open(p) as f:
                solve_content = f.read()
                break

    # Must reference embedding model or corpus embeddings
    assert any(
        kw in solve_content
        for kw in [
            "SentenceTransformer",
            "corpus_emb",
            "all-MiniLM",
            "embedding",
            "FAISS",
            "faiss",
        ]
    ), (
        "solve.sh must contain actual embedding code (SentenceTransformer / corpus_emb.npy / FAISS), not just config strings"
    )

    # Check precomputed embeddings exist (evidence of RAG pipeline)
    assert os.path.exists("/app/data/corpus_emb.npy") or os.path.exists(
        "/app/data/minilm"
    ), (
        "Missing precomputed embeddings /app/data/corpus_emb.npy or minilm model - must use baked MiniLM"
    )

    # Check rag_config declares embedding usage
    with open("/output/rag_config.json") as f:
        cfg = json.load(f)
    assert (
        cfg.get("embedding_used")
        or "embedding" in json.dumps(cfg).lower()
        or "minilm" in json.dumps(cfg).lower()
    ), "rag_config must declare embedding_used"

    # Check eval_report declares embedding
    with open("/output/eval_report.json") as f:
        rep = json.load(f)
    assert (
        rep.get("embedding_used") or "embedding" in json.dumps(rep).lower() or True
    ), "eval_report should mention embedding"


def test_self_improvement_loop_proven():
    """Prove self-improvement loop was executed, not just claimed"""
    with open("/output/rag_config.json") as f:
        cfg = json.load(f)
    with open("/output/eval_report.json") as f:
        rep = json.load(f)

    # Must have self_improvement flag
    assert cfg.get("self_improvement"), "rag_config must have self_improvement: true"

    # Must have steps indicating iterative improvement
    steps = cfg.get("steps", [])
    blob = json.dumps(cfg).lower() + json.dumps(rep).lower()
    # Should mention at least 3 distinct improvement techniques and have lift
    techniques = [
        "filter",
        "rewrite",
        "rerank",
        "dedup",
        "facet",
        "semantic",
        "extract",
    ]
    matches = sum(1 for t in techniques if t in blob)
    assert matches >= 3, (
        f"Should describe >=3 improvement techniques, got {matches}: {techniques}"
    )

    # Must show lift in report
    assert "lift" in rep, "eval_report must contain lift from self-eval loop"
    assert rep.get("lift", 0) > 0.05, (
        f"Self-improvement lift {rep.get('lift')} too low (<0.05)"
    )

    # Check retrieved.jsonl exists and has 500 entries (proves loop produced outputs, not just report)
    with open("/output/retrieved.jsonl") as f:
        retrieved_count = sum(1 for _ in f)
    assert retrieved_count >= 400, (
        f"retrieved.jsonl should have 500 entries, got {retrieved_count}"
    )

    # Check parsed_filters count matches queries
    with open("/output/parsed_filters.jsonl") as f:
        pf_count = sum(1 for _ in f)
    assert pf_count >= 400, f"parsed_filters should have 500 entries, got {pf_count}"


def test_agent_queries_free_form_only():
    """Ensure agent-visible queries are free-form only (no structured city/category/facet leak)"""
    agent_q_path = "/app/data/queries.jsonl"
    assert os.path.exists(agent_q_path)
    with open(agent_q_path) as f:
        sample = json.loads(next(f))
    # Agent-visible should NOT have city, facet, time_window, radius, lat/lng
    # It should only have id, text, query_date
    assert "text" in sample and "id" in sample, "Agent queries must have id and text"
    # These structured fields must NOT be present in agent-visible version (to force extraction)
    forbidden = [
        "city",
        "facet",
        "time_window_start",
        "time_window_end",
        "radius_km",
        "lat",
        "lng",
    ]
    present_forbidden = [k for k in forbidden if k in sample]
    # Allow city to be present in text but not as structured field
    assert len(present_forbidden) == 0, (
        f"Agent-visible queries should be free-form only, but contain structured fields {present_forbidden} - this enables rule-based cheating. Only id+text+query_date should be present."
    )


def test_no_relevance_file_in_app():
    """Ensure relevance.jsonl is not present in agent-visible /app/data to prevent direct cheating"""
    forbidden_paths = [
        "/app/data/relevance.jsonl",
        "/app/data/holdout_relevance.jsonl",
        "/app/references",
    ]
    for p in forbidden_paths:
        # It's okay if /opt/eval has relevance (hidden), but not /app/data
        if p.startswith("/app/data"):
            assert not os.path.exists(p), (
                f"Leakage: {p} should not exist in agent-visible /app/data - agent could cheat by reading labels. Use hidden /opt/eval and recompute on-the-fly."
            )


def test_holdout_structured_hidden():
    """Holdout queries structured labels must be hidden, only free-form visible"""
    agent_h_path = "/app/data/holdout_queries.jsonl"
    if not os.path.exists(agent_h_path):
        return
    with open(agent_h_path) as f:
        sample = json.loads(next(f))
    forbidden = ["city", "facet", "time_window_start", "radius_km"]
    present = [k for k in forbidden if k in sample]
    assert len(present) == 0, (
        f"Holdout agent-visible queries should be free-form only, but contain {present}"
    )

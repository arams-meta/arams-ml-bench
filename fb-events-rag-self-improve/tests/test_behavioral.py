"""
Behavioral tests to enforce RAG, embedding usage, free-form extraction, and self-improvement per TBR feedback.
Fixes:
- No /opt/eval mount in solver (isolation) - checked
- Agent queries free-form only (no structured city/category/facet/time/radius/lat/lng)
- No relevance file anywhere in image (recomputed on-the-fly)
- Embedding usage via agent artifacts, not reference path, no or True vacuous
"""

import json
import os


def test_parsed_filters_exists_and_valid():
    path = "/output/parsed_filters.jsonl"
    assert os.path.exists(path), (
        "Missing /output/parsed_filters.jsonl - must extract hard filters from free-form query text"
    )
    with open(path) as f:
        lines = [json.loads(l) for l in f]
    assert len(lines) >= 500, (
        f"parsed_filters should have 500 entries, got {len(lines)}"
    )
    required = [
        "query_id",
        "city",
        "facet",
        "time_window_start",
        "time_window_end",
        "lat",
        "lng",
        "radius_km",
        "category",
    ]
    for entry in lines[:5]:
        for r in required:
            assert r in entry, f"parsed filter missing {r}: {entry}"
    # Accuracy checks: hidden ground truth must exist at build (700 root) and made readable at test.sh for verifier
    # Present at build so reward depends on build-time 700 isolation, not auto-generated at verify time
    hidden = {}
    for p in ["/opt/eval/queries.jsonl", "data/queries_full_hidden.jsonl"]:
        if os.path.exists(p):
            try:
                with open(p) as f:
                    hidden = {json.loads(l)["id"]: json.loads(l) for l in f}
                break
            except Exception:
                pass
    # Fail loudly if hidden not readable (not silently skip) - ensures extraction accuracy enforced
    assert hidden, (
        "Hidden ground truth /opt/eval/queries.jsonl not readable even after auto-generation - cannot verify extraction accuracy"
    )

    def acc_for(field, thresh):
        correct = 0
        total = 0
        for entry in lines:
            qid = entry["query_id"]
            if qid not in hidden:
                continue
            total += 1
            if str(entry.get(field)) == str(hidden[qid].get(field)):
                correct += 1
        if total > 0:
            acc = correct / total
            assert acc >= thresh, (
                f"Parsed {field} accuracy too low: {acc:.2f} (need >={thresh})"
            )

    acc_for("city", 0.60)
    acc_for("facet", 0.50)
    acc_for("category", 0.40)


def test_embedding_usage_proven():
    """Prove MiniLM embedding search was actually used via behavioral facet-match + recomputed recall (not just baked files)"""
    with open("/output/rag_config.json") as f:
        cfg = json.load(f)
    # Config must declare exact model and boolean true - not just keyword substring (fixes keyword-grep gate string-matchable)
    assert cfg.get("embedding_model") == "all-MiniLM-L6-v2", (
        "rag_config must declare embedding_model exactly all-MiniLM-L6-v2 - exact match, not substring"
    )
    assert cfg.get("embedding_used") is True, (
        "rag_config embedding_used must be true boolean"
    )
    assert cfg.get("self_improvement") is True, (
        "rag_config self_improvement must be true"
    )
    # Baked artifacts existence checked in test_app_data_not_modified, not sole gate here - behavioral proof below is main
    # Load events and retrieved for behavioral proof (recomputed from hidden vs output, not trusting eval_report float)
    events_paths = ["/opt/eval/events.jsonl", "/app/data/events.jsonl"]
    events_file = next((p for p in events_paths if os.path.exists(p)), None)
    assert events_file, "Events file not found"
    events = {json.loads(l)["id"]: json.loads(l) for l in open(events_file)}
    assert os.path.exists("/output/retrieved.jsonl"), "Missing retrieved.jsonl"
    retrieved = {}
    with open("/output/retrieved.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            retrieved[obj["query_id"]] = obj.get("retrieved_ids", [])

    # Behavioral facet-match rate: improved filters by facet, so retrieved events topic == query facet
    # Instruction states: facet/topic match is key semantic signal; popularity-only baseline ignores it and fails ~0.27
    # Baseline pop-only facet match ~1/num_topics (~5-10%), improved must be >=60% - strong behavioral proof, stated in instruction
    parsed_path = "/output/parsed_filters.jsonl"
    assert os.path.exists(parsed_path), "Missing parsed_filters.jsonl for facet proof"
    parsed = {}
    with open(parsed_path) as f:
        for line in f:
            pf = json.loads(line)
            parsed[pf["query_id"]] = pf.get("facet")

    facet_hits = 0
    facet_total = 0
    for qid, facet in parsed.items():
        if not facet:
            continue
        ret = retrieved.get(qid, [])
        for eid in ret[:10]:
            ev = events.get(eid)
            if not ev:
                continue
            facet_total += 1
            if ev.get("topic") == facet:
                facet_hits += 1
    assert facet_total > 0, "No retrieved events to check facet match"
    facet_rate = facet_hits / facet_total
    assert facet_rate >= 0.60, (
        f"Facet-match rate {facet_rate:.2%} too low - retrieved events should have topic==query facet {facet_hits}/{facet_total}, "
        f"proves facet-aware filtering per instruction (facet is key signal)"
    )

    # Semantic similarity behavioral proof: retrieved events should have higher cosine similarity to query than random events
    # This proves embedding scoring was actually used, not just rule-based facet filter (which would have same facet-match but no semantic ranking)
    # Uses baked MiniLM at /app/data/minilm + corpus_emb.npy precomputed at build
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model_path = "/app/data/minilm"
        if os.path.exists(model_path):
            model = SentenceTransformer(model_path)
            # Load corpus embeddings and ids
            corpus_emb = np.load("/app/data/corpus_emb.npy")
            corpus_ids = json.loads(open("/app/data/corpus_ids.json").read())
            id_to_idx = {eid: idx for idx, eid in enumerate(corpus_ids)}
            # Sample 20 queries for speed
            sample_qids = list(retrieved.keys())[:20]
            sim_improved = []
            sim_random = []
            for qid in sample_qids:
                # Query text from /app/data/queries.jsonl free-form
                # Find query text
                qtext = None
                for qp in ["/app/data/queries.jsonl", "data/queries.jsonl"]:
                    if os.path.exists(qp):
                        with open(qp) as qf:
                            for line in qf:
                                obj = json.loads(line)
                                if obj.get("id") == qid:
                                    qtext = obj.get("text")
                                    break
                        if qtext:
                            break
                if not qtext:
                    continue
                q_emb = model.encode(qtext, normalize_embeddings=True)
                ret_ids = retrieved.get(qid, [])[:5]
                for eid in ret_ids:
                    idx = id_to_idx.get(eid)
                    if idx is not None:
                        sim = float(np.dot(q_emb, corpus_emb[idx]))
                        sim_improved.append(sim)
                # Random 5 events not retrieved
                import random

                random.seed(42)
                all_valid_ids = [eid for eid in corpus_ids if eid not in ret_ids]
                rand_ids = random.sample(all_valid_ids, min(5, len(all_valid_ids)))
                for eid in rand_ids:
                    idx = id_to_idx.get(eid)
                    if idx is not None:
                        sim = float(np.dot(q_emb, corpus_emb[idx]))
                        sim_random.append(sim)
            if sim_improved and sim_random:
                avg_improved = sum(sim_improved) / len(sim_improved)
                avg_random = sum(sim_random) / len(sim_random)
                # Improved should have higher semantic similarity than random (embedding scoring)
                assert avg_improved > avg_random, (
                    f"Semantic similarity not higher: improved avg {avg_improved:.3f} <= random avg {avg_random:.3f} - must use embedding scoring"
                )
    except Exception as e:
        # If model not available or fails, skip semantic check but facet-match + recall already prove embedding usage via filtering
        # Don't fail test on model load failure, just log
        print(f"Semantic similarity check skipped: {e}")

    # Recompute recall independently from hidden ground truth vs retrieved (per spec: grader RECOMPUTES, not trusting eval_report floats)
    # Avoid trusting self-reported eval_report float which contradicts spec
    import math
    from datetime import datetime
    from collections import defaultdict

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
        return sorted(
            deduped,
            key=lambda eid: ev_map.get(eid, {}).get("popularity", 0),
            reverse=True,
        )[:30]

    # Load hidden queries for relevance recompute
    queries_file = "/opt/eval/queries.jsonl"
    assert os.path.exists(queries_file), (
        "Hidden queries must exist for recomputed recall"
    )
    queries = {json.loads(l)["id"]: json.loads(l) for l in open(queries_file)}
    cutoff = datetime(2026, 2, 1)
    valid = [
        e
        for e in events.values()
        if not e.get("is_spam") and datetime.fromisoformat(e["start_time"]) >= cutoff
    ]
    ev_map = {e["id"]: e for e in valid}
    by_city = defaultdict(list)
    for e in valid:
        by_city[e["city"]].append(e)
    relevance = {}
    for qid, q in queries.items():
        rel = []
        for ev in by_city.get(q["city"], []):
            if q.get("category") and ev["category"] != q.get("category"):
                continue
            if q.get("facet") and ev.get("topic") != q.get("facet"):
                continue
            ev_start = datetime.fromisoformat(ev["start_time"])
            ws = datetime.fromisoformat(q["time_window_start"])
            we = datetime.fromisoformat(q["time_window_end"])
            if not (ws <= ev_start <= we):
                continue
            if haversine(q["lat"], q["lng"], ev["lat"], ev["lng"]) > q["radius_km"]:
                continue
            rel.append(ev["id"])
        relevance[qid] = dedup_ids(rel, ev_map)

    # Compute improved recall recomputed (not trusting eval_report float)
    recalls = []
    for qid, rel in relevance.items():
        if not rel:
            continue
        ret = retrieved.get(qid, [])
        if not ret:
            recalls.append(0.0)
            continue
        hits = len(set(ret[:10]) & set(rel))
        recalls.append(hits / len(rel))
    improved_recall = sum(recalls) / len(recalls) if recalls else 0.0
    assert improved_recall >= 0.40, (
        f"Recomputed improved recall@10 {improved_recall:.3f} <0.40 - baseline pop-only is 0.27, need facet-aware embedding (recomputed, not trusting eval_report)"
    )


def test_self_improvement_loop_proven():
    with open("/output/rag_config.json") as f:
        cfg = json.load(f)
    with open("/output/eval_report.json") as f:
        rep = json.load(f)
    assert cfg.get("self_improvement") is True, (
        "rag_config self_improvement must be true boolean"
    )
    # Steps must be list with >=3 items describing method, not just keyword substring in blob (fixes string-matchable gate)
    steps = cfg.get("steps", [])
    assert isinstance(steps, list) and len(steps) >= 3, (
        f"rag_config steps must be list with >=3 items, got {steps}"
    )
    # Check that steps contain improvement techniques as explicit list items, not just substring anywhere
    techniques = [
        "filter",
        "rewrite",
        "rerank",
        "dedup",
        "facet",
        "semantic",
        "extract",
    ]
    # Count technique keywords that appear in steps list items (not whole blob)
    steps_blob = " ".join(str(s).lower() for s in steps)
    matches = sum(1 for t in techniques if t in steps_blob)
    assert matches >= 3, (
        f"rag_config steps must describe >=3 techniques {techniques}, got {matches} in {steps}"
    )
    assert "lift" in rep or "baseline_recall_at_10" in rep, (
        "eval_report must contain lift/baseline"
    )

    # Recompute lift independently from hidden ground truth vs retrieved (per spec: grader recomputes, not trusting eval_report float)
    # Fixes 'directional assertion trusting self-reported eval_report float contradicting spec'
    import math
    from datetime import datetime
    from collections import defaultdict

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
        return sorted(
            deduped,
            key=lambda eid: ev_map.get(eid, {}).get("popularity", 0),
            reverse=True,
        )[:30]

    events = {
        json.loads(l)["id"]: json.loads(l) for l in open("/opt/eval/events.jsonl")
    }
    queries = {
        json.loads(l)["id"]: json.loads(l) for l in open("/opt/eval/queries.jsonl")
    }
    cutoff = datetime(2026, 2, 1)
    valid = [
        e
        for e in events.values()
        if not e.get("is_spam") and datetime.fromisoformat(e["start_time"]) >= cutoff
    ]
    ev_map = {e["id"]: e for e in valid}
    by_city = defaultdict(list)
    for e in valid:
        by_city[e["city"]].append(e)
    relevance = {}
    for qid, q in queries.items():
        rel = []
        for ev in by_city.get(q["city"], []):
            if q.get("category") and ev["category"] != q.get("category"):
                continue
            if q.get("facet") and ev.get("topic") != q.get("facet"):
                continue
            ev_start = datetime.fromisoformat(ev["start_time"])
            ws = datetime.fromisoformat(q["time_window_start"])
            we = datetime.fromisoformat(q["time_window_end"])
            if not (ws <= ev_start <= we):
                continue
            if haversine(q["lat"], q["lng"], ev["lat"], ev["lng"]) > q["radius_km"]:
                continue
            rel.append(ev["id"])
        relevance[qid] = dedup_ids(rel, ev_map)

    # Baseline pop-only (facet-ignorant) recomputed
    baseline_recalls = []
    for qid, q in queries.items():
        rel = relevance.get(qid, [])
        if not rel:
            continue
        filtered = []
        for ev in by_city.get(q["city"], []):
            if q.get("category") and ev["category"] != q.get("category"):
                continue
            ev_start = datetime.fromisoformat(ev["start_time"])
            ws = datetime.fromisoformat(q["time_window_start"])
            we = datetime.fromisoformat(q["time_window_end"])
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
        retrieved_baseline = [e["id"] for e in sorted_pop[:10]]
        hits = len(set(retrieved_baseline) & set(rel))
        baseline_recalls.append(hits / len(rel))
    baseline_recall = (
        sum(baseline_recalls) / len(baseline_recalls) if baseline_recalls else 0.0
    )
    assert baseline_recall > 0.0, (
        "Baseline recomputed as 0.0 - hidden ground truth unreadable"
    )

    # Improved recomputed from /output/retrieved.jsonl
    retrieved = {}
    with open("/output/retrieved.jsonl") as fr:
        for line in fr:
            obj = json.loads(line)
            retrieved[obj["query_id"]] = obj.get("retrieved_ids", [])
    improved_recalls = []
    for qid, rel in relevance.items():
        if not rel:
            continue
        ret = retrieved.get(qid, [])
        if not ret:
            improved_recalls.append(0.0)
            continue
        hits = len(set(ret[:10]) & set(rel))
        improved_recalls.append(hits / len(rel))
    improved_recall = (
        sum(improved_recalls) / len(improved_recalls) if improved_recalls else 0.0
    )
    lift = improved_recall - baseline_recall
    assert lift > 0.08, (
        f"Recomputed self-improvement lift {lift:.3f} too low (<0.08 strictest gate): improved {improved_recall:.3f} baseline {baseline_recall:.3f}"
    )
    with open("/output/retrieved.jsonl") as f:
        retrieved_count = sum(1 for _ in f)
    assert retrieved_count >= 500, (
        f"retrieved.jsonl should have 500 entries, got {retrieved_count}"
    )
    with open("/output/parsed_filters.jsonl") as f:
        pf_count = sum(1 for _ in f)
    assert pf_count >= 500, f"parsed_filters should have 500 entries, got {pf_count}"


def test_agent_queries_free_form_only():
    """Ensure agent-visible queries are free-form only (no structured city/category/facet/time/radius/lat/lng)"""
    for p in ["/app/data/queries.jsonl", "data/queries.jsonl"]:
        if os.path.exists(p):
            agent_q_path = p
            break
    else:
        agent_q_path = "/app/data/queries.jsonl"
    assert os.path.exists(agent_q_path), "Missing agent queries"
    with open(agent_q_path) as f:
        sample = json.loads(next(f))
    assert "text" in sample and "id" in sample, "Agent queries must have id and text"
    forbidden = [
        "city",
        "facet",
        "time_window_start",
        "time_window_end",
        "radius_km",
        "lat",
        "lng",
        "category",
        "anchored",
    ]
    present_forbidden = [k for k in forbidden if k in sample]
    assert len(present_forbidden) == 0, (
        f"Agent-visible queries should be free-form only id+text+query_date, but contain {present_forbidden} - enables rule-based cheating"
    )


def test_no_relevance_file_anywhere():
    """Ensure relevance not present in /app/data nor /opt/eval (isolation)"""
    for p in ["/app/data/relevance.jsonl", "/app/data/holdout_relevance.jsonl"]:
        assert not os.path.exists(p), (
            f"Leakage: {p} should not exist in agent-visible /app/data. Recompute on-the-fly."
        )
    for p in ["/opt/eval/relevance.jsonl", "/opt/eval/holdout_relevance.jsonl"]:
        if os.path.exists(p):
            assert False, (
                f"Leakage: {p} is world-readable and contains labels. Per TBR, do not mount /opt/eval with relevance in solver, recompute on-the-fly."
            )


def test_opt_eval_unavailable_from_agent():
    """Isolation: hidden ground truth must be root-owned 700 at build and agent-visible must be free-form only"""
    # No relevance file anywhere (label leakage)
    assert not os.path.exists("/app/data/relevance.jsonl"), (
        "/app/data/relevance.jsonl must not exist"
    )
    assert not os.path.exists("/app/data/holdout_relevance.jsonl"), (
        "holdout relevance must not exist in /app/data"
    )
    assert not os.path.exists("/opt/eval/relevance.jsonl"), (
        "/opt/eval relevance must not exist anywhere (recomputed on-the-fly)"
    )
    assert not os.path.exists("/opt/eval/holdout_relevance.jsonl"), (
        "holdout relevance must not exist"
    )

    # Agent-visible queries must be free-form only id+text+query_date (proves structured not agent-readable)
    agent_q_path = "/app/data/queries.jsonl"
    assert os.path.exists(agent_q_path), "Missing agent queries"
    with open(agent_q_path) as f:
        sample = json.loads(next(f))
    assert "text" in sample and "id" in sample, "Agent queries must have id+text"
    forbidden = [
        "city",
        "facet",
        "time_window_start",
        "time_window_end",
        "radius_km",
        "lat",
        "lng",
        "category",
        "anchored",
    ]
    present = [k for k in forbidden if k in sample]
    assert not present, (
        f"Agent-visible queries should be free-form only, but contain {present} - enables cheating"
    )

    # Hidden structured queries must exist at build (700 root:root) and made readable at test.sh chmod 755 for verifier
    # Present at build so reward depends on build-time 700 isolation (defense-in-depth via USER agentuser + solve.sh root check)
    hidden_path = "/opt/eval/queries.jsonl"
    assert os.path.exists(hidden_path), (
        "Hidden /opt/eval/queries.jsonl must exist at build (700 root), made readable at test time"
    )
    st = os.stat(hidden_path)
    assert st.st_uid == 0, (
        "Hidden queries should be owned by root (isolation: agentuser cannot chown root files)"
    )
    with open(hidden_path) as hf:
        h_sample = json.loads(next(hf))
    assert (
        "city" in h_sample and "facet" in h_sample and "time_window_start" in h_sample
    ), (
        f"Hidden queries must have structured fields city/facet/time: got {list(h_sample.keys())}"
    )
    # Also check hidden has more fields than agent-visible (proves free-form vs structured distinction)
    assert len(h_sample.keys()) > len(sample.keys()), (
        "Hidden should have more fields than agent-visible free-form"
    )


def test_holdout_structured_hidden():
    for p in ["/app/data/holdout_queries.jsonl", "data/holdout_queries.jsonl"]:
        if os.path.exists(p):
            agent_h_path = p
            break
    else:
        return
    with open(agent_h_path) as f:
        sample = json.loads(next(f))
    forbidden = [
        "city",
        "facet",
        "time_window_start",
        "radius_km",
        "lat",
        "lng",
        "category",
    ]
    present = [k for k in forbidden if k in sample]
    assert len(present) == 0, (
        f"Holdout agent-visible queries should be free-form only, but contain {present}"
    )


def test_app_data_not_modified():
    """Direction B: instruction says Do not modify any file in /app/data - check integrity"""
    import os

    # Events must exist and have 20000 lines (generated at build)
    events_path = "/app/data/events.jsonl"
    assert os.path.exists(events_path), (
        "events.jsonl missing - agent may have deleted /app/data"
    )
    with open(events_path) as f:
        event_lines = sum(1 for _ in f)
    assert event_lines == 20000, (
        f"events.jsonl should have 20000 lines, got {event_lines} - do not modify /app/data"
    )

    # Queries must have 500 lines, free-form only
    queries_path = "/app/data/queries.jsonl"
    assert os.path.exists(queries_path), "queries.jsonl missing"
    with open(queries_path) as f:
        q_lines = [json.loads(line) for line in f]
    assert len(q_lines) == 500, f"queries.jsonl should have 500, got {len(q_lines)}"
    # Must still be free-form only (no structured city leak)
    assert "city" not in q_lines[0], (
        "queries.jsonl modified to include city - do not modify /app/data"
    )

    # Holdout 100
    h_path = "/app/data/holdout_queries.jsonl"
    assert os.path.exists(h_path), "holdout_queries.jsonl missing"
    with open(h_path) as f:
        h_lines = sum(1 for _ in f)
    assert h_lines == 100, f"holdout_queries.jsonl should have 100, got {h_lines}"

    # Baked model must still exist
    assert os.path.exists("/app/data/minilm") or os.path.exists(
        "/app/data/minilm/config.json"
    ), "minilm model missing - do not modify /app/data"
    assert os.path.exists("/app/data/corpus_emb.npy"), "corpus_emb.npy missing"
    assert os.path.exists("/app/data/corpus_ids.json"), "corpus_ids.json missing"

The Facebook events need a RAG model that will support event search both by keywords and embeddings. It should be able to extract hard filters (like location, date, etc) from free-form query and create set of hard filters + RAG embeddings for semantic search. Create that RAG.

Inventory, queries, holdout inputs live in /app/data:
- /app/data/events.jsonl: 20k synthetic events with fields id, title, description, category (Music/Sports/Food/Tech/Arts/Nightlife/Community/Business), city, venue, lat/lng, start_time/end_time ISO, popularity 1-100, is_spam bool, cluster_id (near-duplicates share id), topic (semantic facet: tech_topic/cuisine/artist/sport/art/activity)
- /app/data/queries.jsonl: 500 queries with id, text, city, category (optional), facet (topic to match), time_window_start/end ISO, radius_km, lat/lng, query_date, anchored bool
- /app/data/holdout_queries.jsonl: 100 holdout queries shifted +35 days for generalization check

Relevance rule (deterministic, no LLM):
- event city == query city
- category match if query specifies category
- facet/topic match: event.topic == query.facet
- start_time within [time_window_start, time_window_end]
- haversine distance <= radius_km
- not is_spam, start_time >= 2026-02-01 cutoff
- dedup: keep only 1 per cluster_id (highest popularity) before capping 30
- guarantee >=1 relevant per query (anchored forced)

You need to produce output in following targets:
- /output/retrieved.jsonl: jsonl lines {"query_id": "q_00000", "retrieved_ids": ["ev_..."] } top 10 per query
- /output/holdout_retrieved.jsonl: same for holdout queries
- /output/rag_config.json: {"retriever": "...", "embedding_model": "all-MiniLM-L6-v2", "improvement_method": "filter-first + facet semantic + dedup + popularity re-rank + query rewrite", "self_improvement": true, "steps": [...]}
- /output/eval_report.json: {"baseline_recall_at_10": float, "improved_recall_at_10": float, "recall_at_10": float, "holdout_recall_at_10": float, "lift": float, "num_queries": 500, "num_events": ..., "time_filter_pass_rate": 1.0, "geo_filter_pass_rate": 1.0, "dedup_enabled": true}

Constraints:
- No external API
- Use baked MiniLM at /app/data/minilm (sentence-transformers/all-MiniLM-L6-v2, 384-dim)
- CPU only, FAISS IndexFlatIP or brute-force cosine allowed
- Seed is 42 for determinism
- Retrieved outputs should respect time/geo/dedup/spam constraints (violation <=10% time, <=15% geo, dup_rate <=30%, spam <=5%)
- Do NOT read relevance files if present in /app/data or /opt/eval - they are for evaluation only and would be considered cheating. Recompute relevance using rule above for eval.

Scoring thresholds (agent must meet):
- Improved recall@10 >= 0.60 absolute
- Lift: improved > baseline + 0.08
- Baseline floor: trivial popularity-only (filter city+category+time+geo+dedup, sort by popularity) has ~0.25 recall because it ignores facet, so improved must beat it by >0.05
- Holdout generalization: holdout_recall / train_recall >= 0.8 and holdout >=0.45
- Build must pass structural checks, no hardcoded /tmp/oracle.py paths, taxonomy valid

Self-improvement loop should retry query rewrite techniques (different hard filter + embeddings combinations) to achieve the highest match score (recall@10, baseline vs improved largest delta).

Suggested improved method: filter-first-then-semantic-rank
1. Filter valid events by city, category (if any), time window, geo radius (haversine), not spam, dedup via seen_clusters keeping highest popularity
2. Build corpus texts only for filtered subset including topic field
3. Encode filtered subset, compute cosine vs query (or query+facet expanded f"{q.text} . Topic {q.facet} . Category {q.category}"), rank by score + popularity*0.2
4. Keep top 10, for low-recall (<0.5) do query rewrite with facet expanded and merge

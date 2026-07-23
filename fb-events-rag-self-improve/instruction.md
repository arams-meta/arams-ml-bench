The Facebook events need a RAG model that will support event search both by keywords and embeddings. It should be able to extract hard filters (like location, date, etc) from free-form query and create set of hard filters + RAG embeddings for semantic search. Create that RAG.

Inventory, queries, holdout inputs live in /app/data:
- /app/data/events.jsonl: 20k synthetic events with fields id, title, description, category (Music/Sports/Food/Tech/Arts/Nightlife/Community/Business), city, venue, lat/lng, start_time/end_time ISO, popularity 1-100, is_spam bool, cluster_id (near-duplicates share id), topic (semantic facet)
- /app/data/queries.jsonl: 500 queries with id, text, city, category (optional), facet (topic to match), time_window_start/end ISO, radius_km, lat/lng, query_date
- /app/data/holdout_queries.jsonl: 100 holdout queries shifted +35 days for generalization check

Relevance rule (deterministic, no LLM):
- event city == query city
- category match if query specifies category
- facet/topic match: event.topic == query.facet (this is the key semantic signal; popularity-only baseline ignores it and fails)
- start_time within [time_window_start, time_window_end]
- haversine distance <= radius_km
- not is_spam, start_time >= 2026-02-01 cutoff
- dedup: keep only 1 per cluster_id (highest popularity) before capping 30
- guarantee >=1 relevant per query

You need to produce output in following targets:
- /output/retrieved.jsonl: jsonl lines {"query_id": "q_00000", "retrieved_ids": ["ev_..."] } top 10 per query
- /output/holdout_retrieved.jsonl: same for holdout queries
- /output/rag_config.json: {"retriever": "...", "embedding_model": "all-MiniLM-L6-v2", "improvement_method": "...", "self_improvement": true}
- /output/eval_report.json: {"baseline_recall_at_10": float, "improved_recall_at_10": float, "recall_at_10": float, "holdout_recall_at_10": float, "lift": float}

Constraints:
- No external API
- Use baked MiniLM at /app/data/minilm (sentence-transformers/all-MiniLM-L6-v2, 384-dim)
- CPU only, FAISS allowed
- Seed is 42 for determinism
- Retrieved outputs should respect time/geo/dedup/spam constraints (violation <=10% time, <=15% geo, dup_rate <=30%, spam <=5%)
- Do NOT read relevance files if present - they are for evaluation only. Recompute relevance using rule above for eval. Do not modify any file in /app/data.
- The grader RECOMPUTES recall@10 from your retrieved.jsonl against the relevance rule - self-reported numbers in eval_report are checked for schema only, not trusted as scores.

Scoring thresholds (grader recomputes independently):
- Improved recall@10 >= 0.60 absolute
- Lift: improved > baseline + 0.08 where baseline is trivial popularity-only filtered by city/category/time/geo/dedup
- Baseline floor: trivial popularity-only has ~0.25 recall because it ignores facet, so improved must beat it by >0.05
- Holdout generalization: holdout_recall / train_recall >= 0.8 and holdout >=0.45

Self-improvement loop should analyze low-recall queries and retry techniques like query rewriting, re-ranking, or different filter combinations to achieve the highest recall@10 and largest delta over baseline. You need to figure out how to use the semantic facet and other signals to beat the baseline.

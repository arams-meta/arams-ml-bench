The Facebook events need a RAG model that will support event search both by keywords and embeddings. It should be able to extract hard filters (like location, date, etc) from free-form query and create set of hard filters + RAG embeddings for semantic search. Create that RAG.

Inventory, queries, holdout inputs live in /app/data:
- /app/data/events.jsonl: 20k synthetic events with fields id, title, description, category (Music/Sports/Food/Tech/Arts/Nightlife/Community/Business), city, venue, lat/lng, start_time/end_time ISO, popularity 1-100, is_spam bool, cluster_id (near-duplicates share id), topic (semantic facet: AI/Web3/Thai/Vegan/etc for dedup)
- /app/data/queries.jsonl: 500 queries FREE-FORM ONLY with fields id, text, query_date (text contains city, category, facet like "best Thai Food events near Austin this weekend", you must extract hard filters from text)
- /app/data/holdout_queries.jsonl: 100 holdout queries shifted +35 days, also free-form only id+text+query_date
- Hidden ground truth for evaluation lives in /opt/eval/ (events.jsonl, queries.jsonl with full city/category/facet/time_window/radius/lat/lng, holdout_queries.jsonl full) - NOT agent-visible, used by tests to recompute relevance. Do not rely on structured fields in /app/data/queries.

Note: Full structured queries with city/category/facet/time_window/radius/lat/lng are in /opt/eval/ for grading only. Agent must parse free-form text to extract city, category, facet, time phrase, etc.

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
- /output/parsed_filters.jsonl: jsonl lines {"query_id": "...", "city": "...", "category": "...", "facet": "...", "time_window_start": "...", "time_window_end": "...", "lat": ..., "lng": ..., "radius_km": ... } - your extracted hard filters from free-form text
- /output/rag_config.json: {"retriever": "...", "embedding_model": "all-MiniLM-L6-v2", "improvement_method": "...", "self_improvement": true, "steps": [...], "embedding_used": true}
- /output/eval_report.json: {"baseline_recall_at_10": float, "improved_recall_at_10": float, "recall_at_10": float, "holdout_recall_at_10": float, "lift": float, "parsed_filters_count": 500}

Free-form extraction requirement: queries only have text like "best Thai Food events near Austin this weekend". You must extract city (Austin), category (Food), facet (Thai), time phrase (this weekend -> window). Use keyword matching, regex, or embedding, not structured fields.
Embedding requirement: you must use baked MiniLM embeddings for semantic ranking, prove usage via rag_config embedding_used and code containing SentenceTransformer / corpus_emb.npy usage.

Constraints:
- No external API
- Use baked MiniLM at /app/data/minilm (sentence-transformers/all-MiniLM-L6-v2, 384-dim)
- CPU only, FAISS allowed
- Seed is 42 for determinism
- Retrieved outputs should respect time/geo/dedup/spam constraints (violation <=40% time, <=20% geo, dup_rate <=30%, spam <=5%)
- Do NOT read relevance files if present - they are for evaluation only. Recompute relevance using rule above for eval. Do not modify any file in /app/data.
- The grader RECOMPUTES recall@10 from your retrieved.jsonl against the relevance rule - self-reported numbers in eval_report are checked for schema only, not trusted as scores.

Temporal semantics (deterministic):
- query_date is ISO date of query issuance (e.g., 2026-02-20)
- Time phrases map to offsets from query_date:
  tomorrow = query_date+1 day, this weekend = query_date+3 days, next week = +10 days,
  this month = +15 days, next month = +45 days
- time_window_start/end are query_date + phrase_offset ±20 days fixed (deterministic)
- Geo: lat/lng = city center, radius_km 50 fixed

Scoring thresholds (grader recomputes from hidden ground truth in /opt/eval):
- Improved recall@10 >= 0.40 absolute
- Lift: improved > baseline + 0.08 (baseline is trivial popularity-only, ~0.25-0.30 because it ignores facet)
- Holdout: holdout recall >=0.25 and ratio >=0.4
- Time/geo/dedup/spam: time <=40%, geo <=20%, dup_rate <=30%, spam <=5%
- Free-form extraction: city >=0.60, facet >=0.50, category >=0.40
- Embedding: must use baked MiniLM at /app/data/minilm, rag_config embedding_model all-MiniLM-L6-v2 and embedding_used true, plus facet-match rate >=60% on retrieved (topic==facet) proving facet-aware filtering
- Self-improvement: rag_config self_improvement true, 500 parsed, 500 retrieved, 100 holdout

Constraints: Do NOT read /opt/eval if present - grading only. Do NOT modify /app/data.

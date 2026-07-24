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

Temporal semantics (deterministic, standard rule from query_date per TBR feedback):
- query_date is ISO date of query issuance (e.g., 2026-02-20)
- Time phrases map to offsets from query_date (deterministic):
  tomorrow = query_date+1 day, this weekend = query_date+3 days, next week = +10 days,
  this month = +15 days, next month = +45 days, this Friday = +4 days
- time_window_start/end are query_date + phrase_offset ±20 days fixed (no random jitter, deterministic)
  So a parser that extracts phrase and adds ±20 days around query_date+offset will exactly match hidden windows.
  This follows practice guide to make hidden time windows follow standard deterministic rules from query_date (fixes R01 Spec sufficiency and R08 Accepts alternatives).
- Geo: lat/lng = city center (average lat/lng of events in city, no jitter) or anchor lat/lng, radius_km 50 fixed. Agent must extract city and use city center for geo.

Scoring thresholds (grader recomputes independently from hidden ground truth in /opt/eval, not self-reported):
- Improved recall@10 >= 0.40 absolute (free-form, recomputed from retrieved.jsonl vs hidden relevance)
- Lift: improved > baseline + 0.08 where baseline is trivial popularity-only filtered by city/category/time/geo/dedup and recomputed (baseline ignores facet, so ~0.25-0.30)
- Baseline floor: trivial pop-only has ~0.25-0.30 recall because it ignores facet, improved must beat by >0.05 (recomputed from retrieved)
- Holdout: holdout recall >=0.25 and holdout/train ratio >=0.4 (free-form, recomputed)
- Time/geo/dedup/spam constraints: time violation <=40%, geo <=20%, dup_rate <=30%, spam <=5%
- Free-form extraction accuracy (parsed_filters.jsonl vs hidden /opt/eval):
  city >=0.60, category >=0.40, facet >=0.50, time_window present, radius/lat/lng present
- Embedding usage: rag_config must contain embedding_model all-MiniLM-L6-v2 and embedding_used true, plus steps describing filter-first, semantic, facet
  Behavioral proof: retrieved events must have facet-match rate >=60% (retrieved event topic == query facet). Baseline pop-only ignores facet and gets ~8% facet-match (30 topics random), so 60% proves facet-aware semantic filtering via embeddings. Baked artifacts /app/data/minilm + corpus_emb.npy must exist (offline model, no API).
- Self-improvement loop: rag_config self_improvement true + steps covering filter/rewrite/rerank, eval_report lift>0.05, parsed_filters_count 500, retrieved_count 500
- Isolation: hidden ground truth in /opt/eval chmod 700 root:root is NOT agent-readable; agent-visible queries are free-form only id+text+query_date. Solver must run as non-root agentuser (enforced via Dockerfile USER agentuser and solve.sh root check) - no sudo granted to agent to bypass. Do NOT attempt to read /opt/eval.

Self-improvement loop should analyze low-recall queries and retry techniques like query rewriting, re-ranking, or different filter combinations to achieve the highest recall@10 and largest delta over baseline. You need to figure out how to use the semantic facet and other signals to beat the baseline.

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
Embedding requirement: must use baked MiniLM at /app/data/minilm (384-dim) + precomputed corpus_emb.npy + corpus_ids.json (baked at build to fix timeout). Use FAISS or cosine. Prove via rag_config embedding_model all-MiniLM-L6-v2 and embedding_used true, plus behavioral: retrieved events must have topic==facet facet-match >=60% and higher cosine similarity to query than random events (avg improved > random) using MiniLM embeddings.

Constraints:
- No external API, CPU only, FAISS allowed, seed 42, deterministic
- Baked artifacts must exist: /app/data/minilm, /app/data/corpus_emb.npy, /app/data/corpus_ids.json (generated at build)
- Do NOT read /opt/eval if present at solve time - grading only, hidden present at build 700 root:root and deleted at solve for defense-in-depth, regenerated at test time if missing. Do NOT modify /app/data
- Grader recomputes recall/lift/facet-match/semantic from retrieved.jsonl vs hidden /opt/eval (events+queries full), not trusting eval_report floats - eval_report is schema only

Temporal semantics (deterministic):
- query_date: query issuance date
- Time phrases: tomorrow +1d, weekend +3d, next week +10d, this month +15d, next month +45d
- Train: time_window = query_date + offset ±20d fixed, lat/lng = city center, radius 50 fixed
- Holdout: queries shifted +35d future, windows shift +35d. Reference may use multi-anchor union ±30d around closest inventory anchors for robustness (train ±20d, holdout ±30d because +35d shift).

Scoring thresholds (recomputed from hidden /opt/eval, not self-reported):
- Improved recall@10 >=0.40, lift > baseline+0.08 (baseline ~0.27 pop-only ignoring facet), holdout >=0.25 ratio >=0.4
- Time/geo/dedup/spam/freshness: time <=40%, geo <=20%, dup <=30%, spam <=5%, freshness start_time>=2026-02-01 must be 0%
- Free-form extraction: city >=0.60, facet >=0.50, category >=0.40 (parsed_filters.jsonl vs hidden)
- Embedding: as above, facet-match >=60% and semantic cosine improved > random
- Self-improvement: loop must show iteration: rag_config self_improvement true + steps list >=3 with rewrite/rerank, eval_report must have baseline and improved (existence), lift>0.08 recomputed from hidden vs retrieved (not trusting float), plus 500 parsed, 500 retrieved, 100 holdout
- Isolation: hidden in /opt/eval 700 root:root at build, USER agentuser enforced, solve.sh drops root to agentuser and deletes hidden if readable at solve (defense-in-depth, no sudo), no relevance file anywhere, agent-visible queries free-form only id+text+query_date

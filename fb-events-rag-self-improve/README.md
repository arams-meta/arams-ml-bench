# FB Events RAG Self-Improve - LLM Research Task

## Description
Self-improving RAG system with auto-eval loop for synthetic Facebook Events inventory.

Agent gets 20k synthetic events (music, food, tech, etc.) with temporal, geo, category, semantic facet metadata, plus failure modes:
- 10% past events (freshness filter required)
- 31% near-duplicate clusters (dedup required, 1 per cluster highest popularity)
- 5% spam (filtering)
- Geo scatter around city centers (radius filtering, haversine)
- Semantic facet (topic) matching required for relevance (pop-only baseline fails)

Relevance rule (deterministic):
city==, category if given, topic==facet, start_time in window, haversine<=radius, not spam, start>=2026-02-01, dedup 1 per cluster best popularity, cap 30, guarantee >=1 per query.

Agent must implement:
1. Base RAG: MiniLM embeddings (baked at /app/data/minilm) + FAISS + filter-first
2. Auto-eval: deterministic relevance via city/category/facet/time/geo/dedup/spam, recall@10
3. Self-improvement loop: analyze low-recall queries -> query rewriting with facet expansion f"{q.text} . Topic {q.facet} . Category {q.category}" + popularity re-rank -> re-evaluate showing lift

Improved method: filter-first + facet semantic + dedup + popularity re-rank + query rewrite
- Filter by city, category, time window, geo radius (city-indexed), spam, dedup
- Build corpus only for filtered subset including topic
- Cosine vs query (or facet-expanded), rank score + popularity*0.2, top10
- Low-recall rewrite (recall<0.5) with facet expansion, merge

## Completion Rates
- Oracle: 3/3 passed local docker (baseline 0.25 -> improved 0.707, holdout 0.656, lift +0.45). Cloud oracle previously 0/2 timeout (8.7M haversine calls), fixed with city-index to 725k calls, now expected 3/3.
- Sonnet 4.6: 0/5 (failed to implement facet-aware filtering, returned popularity baseline)
- Opus 4.8: 0/5 (similar, no facet handling)
- Avocado (meta/avocado_dvsc_tester): 0/5 initially, 0/7 metacode on cloud, too hard without RAG scaffold
- GPT 5.5: 0/1 (BAD_LEAKAGE fixed, still hard)

All models fail the lift check without proper facet semantic handling, demonstrating task difficulty is in correct filter-first + facet logic, not setup.

## Model Analysis
- **Oracle 3/3**: Uses filter-first (city index), encodes filtered subset with MiniLM+topic, cosine + pop*0.2, query rewrite for 120 low-recall queries, recomputes relevance on-the-fly from hidden /opt/eval to avoid leakage. Passes all 15 pytest (smoke 3, schema 3, baseline 1, main 8, holdout 1).
- **Sonnet/Opus/GPT 0/5**: Common failure - implements only global vector search without facet/topic matching, no dedup, no time/geo filtering, writes fabricated eval_report.json with fake recall numbers. Our new tests recompute recall independently from retrieved.jsonl vs hidden /opt/eval ground truth, catching fabrication.
- **Dominant failure modes across models**:
  - 60%: Missing facet filter (topic != facet) -> recall ~0.25 baseline
  - 25%: No dedup handling -> dup_rate >30% fails
  - 15%: Time/geo violation >10%/15% or spam >5%
- These reflect reasoning gaps (understanding facet-aware relevance and constraint handling), not setup issues. Build is pinned (sentence-transformers==2.2.2, huggingface_hub==0.16.4, faiss-cpu==1.7.4) and model baked offline.

## Anti-Cheating Analysis
- Dataset is synthetic, deterministic seed 42, generated at build time in /tmp/build_refs then deleted, not public
- No external API calls (offline, baked MiniLM weights at /app/data/minilm, TRANSFORMERS_OFFLINE=1)
- Ground truth hidden: relevance and also events/queries copied to /opt/eval (not in /app), tests load from /opt/eval first (not agent-writable /app/data), generators deleted at build time to avoid leakage of relevance rule
- Baseline floor beats trivial popularity: filter city+category+time+geo+dedup pop-only has ~0.27 recall because it ignores facet, improved must beat by >0.05 and >0.08 lift + >=0.60 absolute, recomputed independently from retrieved.jsonl (not trusting report)
- Holdout queries shifted +35 days future - tests generalization (ratio >=0.8 and absolute >=0.45), recomputed independently
- Time/geo/spam/dedup constraint tests prevent hardcoding event IDs (must respect windows)
- Naive approach discriminator: simple global vector search without filters fails time/geo/dedup and recall lift
- Canary GUID present in tests/test.sh, solution uses /app/oracle_impl.py not /tmp/oracle.py to avoid structural WARN

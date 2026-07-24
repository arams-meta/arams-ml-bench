# FB Events RAG Self-Improve - LLM Research Task

## Description
Self-improving RAG system with auto-eval loop for synthetic Facebook Events inventory.

Agent gets 20k synthetic events (music, food, tech, etc.) with temporal, geo, category metadata, plus failure modes:
- 10% past events (freshness filter required)
- 31% near-duplicate clusters (dedup required)
- 5% spam (filtering)
- Geo scatter around city centers (radius filtering)

Agent must implement:
1. Base RAG: MiniLM embeddings + FAISS + generator (or retrieval-only for CPU)
2. Auto-eval: deterministic relevance via time window + geo + categorypus rules, recall@10
3. Self-improvement loop: analyze low-recall queries -> query rewriting / re-ranking / dedup -> re-evaluate showing lift

## Completion Ratesm
TODO - fill after codimango trials

## Model Analysis
- Naive baseline (popularity sort, no filters): expected recall ~0.3-0.4
- Filtered + dedup + rerank: recall >=0.6
- + query rewriting: recall >=0.65-0.70
- Lift required: +0.08 over baseline (directional test)

## Anti-Cheating Analysis
- Dataset is synthetic, deterministic seed 42, not public
- No external API calls (offline, baked MiniLM weights)
- Baseline floor beats trivial popularity
- Holdout queries shifted 35 days future - tests generalization (holdout recall >=0.25 and holdout/train ratio >=0.4)
- Time/geo/spam/dedup constraint tests prevent hardcoding event IDs
- Naive approach discriminator: simple vector search without filters fails time/geo checks

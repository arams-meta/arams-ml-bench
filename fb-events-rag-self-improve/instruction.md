The Facebook events need a RAG model that will support event search both by keywords and embeddings. It should be able to extract hard filters (like location, date, etc) from free-form query and create set of hard filters + RAG embeddings for semantic search. Create that RAG.

- Inventory, queries, relevance, holdout inputs live in /app/data.

- You need to produce output in following targets - /output/rag_config.json, /output/eval_report.json, /output/retrieved.jsonl, /output/holdout_retrieved.jsonl.

Constraints:
- No external API
- Use baked MiniLM at /app/data/minilm
- CPU only
- Seed is 42
- Retrieved outputs should respect time/geo/dedup/spam constraints

Output schema:
- Beyond score, include baseline, improved, method, and any other info you think is useful for debugging.

Self-improvement loop should retry query rewrite techniques (different hard filter + embeddings combinations) to achieve the highest match score (recall@10, baseline vs improved largest delta).

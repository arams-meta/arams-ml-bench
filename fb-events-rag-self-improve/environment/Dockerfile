# Self-improving RAG for FB Events inventory - ML Bench
FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl && rm -rf /var/lib/apt/lists/*

# Python deps - baked at build time for determinism
# huggingface_hub pinned: sentence-transformers==2.2.2 imports cached_download,
# which was removed in newer huggingface_hub releases. 0.16.4 keeps it available.
RUN pip install --no-cache-dir \
    sentence-transformers==2.2.2 \
    huggingface_hub==0.16.4 \
    faiss-cpu==1.7.4 \
    numpy==1.24.3 \
    scikit-learn==1.3.0 \
    pandas==2.0.3 \
    pytest==7.4.0 \
    pytest-json-report==1.5.0

# Work dirs
WORKDIR /app
RUN mkdir -p /app/data /app/references /output /tests /solution /logs/verifier

# Copy generator references (deterministic, no 3P tokens in data)
COPY environment/references/ /app/references/
COPY environment/references/ /references/

# Generate synthetic inventory at build time (deterministic seed=42)
# 20k events, 500 queries + 100 holdout for eval - CPU friendly ~ few MB
RUN python3 /references/generate_events.py --n 20000 --seed 42 --out /app/data/events.jsonl
RUN python3 /references/generate_queries.py \
    --events /app/data/events.jsonl \
    --n 500 --seed 42 \
    --out-queries /app/data/queries.jsonl \
    --out-rel /app/data/relevance.jsonl \
    --out-holdout-queries /app/data/holdout_queries.jsonl \
    --out-holdout-rel /app/data/holdout_relevance.jsonl \
    --holdout-n 100

# Pre-download embedding model to avoid runtime download (offline eval)
RUN python3 -c "from sentence_transformers import SentenceTransformer; m=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); m.save('/app/data/minilm')"

# Copy tests and solution placeholders (evaluator mounts them at /tests, /solution)
COPY tests/ /tests/
COPY solution/ /solution/

ENV PYTHONHASHSEED=42
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

# Default cmd for debugging
CMD ["bash"]

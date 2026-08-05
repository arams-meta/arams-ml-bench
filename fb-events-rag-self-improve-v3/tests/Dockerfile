# V3 verifier - simple base + CPU torch pin + pre-baked
FROM public.ecr.aws/docker/library/python:3.10.12-slim-bullseye

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.1.2 && \
    pip install --no-cache-dir \
    sentence-transformers==2.2.2 \
    transformers==4.35.2 \
    tokenizers==0.15.2 \
    safetensors==0.4.1 \
    huggingface_hub==0.16.4 \
    faiss-cpu==1.7.4 \
    numpy==1.24.3 \
    scikit-learn==1.3.0 \
    pandas==2.0.3 \
    pytest==7.4.0 \
    pytest-json-report==1.5.0

WORKDIR /app
RUN mkdir -p /app/data /opt/eval /output /logs/verifier /tests

# Pre-baked: agent-visible + hidden
COPY data/ /app/data/
COPY eval/ /opt/eval/

COPY test.sh categorize_failure test_smoke.py test_schema.py test_baseline.py test_main.py test_behavioral.py test_holdout.py /tests/
RUN chmod +x /tests/test.sh /tests/categorize_failure

RUN ls -lh /app/data/ /opt/eval/ && \
    echo "Verifier must have hidden:" && test -f /opt/eval/queries.jsonl && test -f /opt/eval/events.jsonl && echo "hidden exists good" && \
    if [ -d /app/data/eval ]; then echo "LEAK: eval in agent data!"; exit 1; else echo "no eval in agent good"; fi && \
    chmod 700 /opt/eval && chown -R root:root /opt/eval && \
    chown -R root:root /app/data && chmod -R a+rX /app/data && chmod 700 /opt/eval && \
    echo "verifier ready v3 clean"

ENV PYTHONHASHSEED=42
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

WORKDIR /app
CMD ["bash"]

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY config.yaml ./

# Pre-bake the reranker so first request isn't slow with a cold model download.
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"

EXPOSE 8000
CMD ["uvicorn", "agentic_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]

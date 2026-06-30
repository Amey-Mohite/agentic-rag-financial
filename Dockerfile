FROM python:3.12-slim

# build-essential + libpq-dev: needed to build psycopg (pgvector backend) wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# Install the package + the table-extraction extra (pdfplumber) so PDF uploads keep tables readable.
# (python-multipart for uploads is a core dependency.) Add ".[hybrid]" too for SPLADE/BM42 in-cluster.
RUN pip install --no-cache-dir -e ".[tables]"

# App assets: both configs, the web UI, and an uploads dir for documents sent via /upload.
COPY config.yaml config.k8s.yaml ./
COPY web ./web
RUN mkdir -p data/uploads

# Pre-bake the reranker so the first question isn't slow with a cold model download.
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"

EXPOSE 8000
# Serve the API + UI. Shell form so $PORT is expanded at runtime: PaaS platforms inject a port
# (Render sets $PORT; Hugging Face expects 7860 — set a PORT=7860 variable there). Defaults to 8000
# locally / in docker-compose / in k8s where PORT is unset. RAG_CONFIG selects the config file.
CMD uvicorn agentic_rag.api:app --host 0.0.0.0 --port ${PORT:-8000}

# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

WORKDIR /app

# CPU-only PyTorch keeps the image small
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml README.md LICENSE ./
COPY memoryforge ./memoryforge
RUN pip install ".[postgres,embeddings]" \
 && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

RUN useradd --create-home --uid 1000 forge && mkdir -p /app/runs && chown -R forge /app /opt/hf
USER forge

EXPOSE 8765
CMD ["memoryforge", "serve", "--host", "0.0.0.0", "--port", "8765"]

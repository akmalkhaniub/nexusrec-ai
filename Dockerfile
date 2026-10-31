# ==============================================================================
# NexusRec AI — Multi-Stage Production Dockerfile
# Supports Google Cloud Run (Scale-to-Zero CPU / NVIDIA L4 GPU) & AWS ECS Fargate
# ==============================================================================
FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install --prefix=/install -r requirements.txt

# ==============================================================================
# Stage 2: Distroless-style non-root production runtime
# ==============================================================================
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

RUN groupadd -r nexusrec && useradd -r -g nexusrec -d /app -s /sbin/nologin nexusrec

WORKDIR /app

COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY pyproject.toml README.md ./

RUN chown -R nexusrec:nexusrec /app
USER nexusrec

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

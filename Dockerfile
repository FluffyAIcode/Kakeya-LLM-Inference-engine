# Kakeya inference engine — CPU-only Docker image.
#
# Builds a runtime container that serves the OpenAI-compatible API
# defined in `inference_engine.server`. CPU backend is the default —
# Apple Silicon (MLX) does not run in Linux Docker; for CUDA, a
# separate Dockerfile.cuda would carry the NVIDIA base image and AWQ
# loader (ADR 0002 §3.5 territory; deferred).
#
# Usage:
#   docker build -t kakeya:latest .
#
#   # Mount the HuggingFace cache so weights persist across container
#   # restarts (the first start downloads ~5 GB):
#   docker run --rm -p 8000:8000 \
#     -v "$HOME/.cache/huggingface:/home/kakeya/.cache/huggingface" \
#     kakeya:latest
#
#   # Pass extra serve.py flags after the image name:
#   docker run --rm -p 8000:8000 kakeya:latest \
#     --max-concurrent 4 --admission-policy queue --queue-max-wait-s 30
#
# The image runs as a non-root user (uid 10001) and listens on port
# 8000. /healthz is wired into Docker's HEALTHCHECK so orchestrators
# can detect a hung process without parsing logs.

FROM python:3.12-slim AS runtime

# OCI labels for image registries / GHCR.
LABEL org.opencontainers.image.title="kakeya"
LABEL org.opencontainers.image.description="DLM-proposer + AR-verifier speculative decoding engine"
LABEL org.opencontainers.image.source="https://github.com/FluffyAIcode/Kakeya-LLM-Inference-engine"
LABEL org.opencontainers.image.licenses="see LICENSE in repo root"

# Avoid surprises from pip's cache and Python's user-site interaction.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install runtime dependencies in a separate layer so iteration on
# source code does not invalidate the deps cache. requirements.txt
# pins all the runtime packages we ship.
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Application source. `.dockerignore` prunes tests/, results/, .git/,
# and other non-runtime artifacts so the image stays small.
COPY inference_engine ./inference_engine
COPY kv_cache_proposer ./kv_cache_proposer
COPY scripts ./scripts
COPY training ./training
COPY README.md LICENSE ./

# Run as non-root. UID 10001 avoids collision with common host UIDs.
RUN groupadd --system --gid 10001 kakeya \
    && useradd --system --uid 10001 --gid 10001 --create-home --shell /bin/bash kakeya \
    && chown -R kakeya:kakeya /app
USER kakeya

# `PYTHONPATH` is set so `python scripts/serve.py` can import
# `inference_engine.*` without an editable install.
ENV PYTHONPATH=/app

EXPOSE 8000

# Liveness probe via the same /healthz route Prometheus and external
# load balancers consume. urllib is stdlib — no curl install required.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request, sys; \
r = urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3); \
sys.exit(0 if r.status == 200 else 1)"

# Default to the CPU backend bound to all interfaces inside the
# container. Override on the command line for different policies.
ENTRYPOINT ["python", "scripts/serve.py"]
CMD ["--backend", "cpu", "--host", "0.0.0.0", "--port", "8000"]

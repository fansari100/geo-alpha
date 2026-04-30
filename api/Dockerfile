# Multi-stage build for the geo-alpha API gateway.
#
# This Dockerfile expects the build context to be the repository root
# (not the api/ subdirectory) so that the sibling geoalpha-quant package
# can be vendored alongside the api source tree. Build invocation:
#
#   docker build -t geoalpha/api -f api/Dockerfile .
#
ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /build
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY quant /build/quant
COPY api/pyproject.toml /build/api/pyproject.toml
COPY api/README.md      /build/api/README.md
COPY api/src            /build/api/src

RUN pip install --upgrade pip hatchling \
    && pip install --target /install \
        numpy scipy \
        "fastapi>=0.110" "uvicorn[standard]>=0.27" "pydantic>=2.5" \
    && pip install --target /install --no-deps /build/quant \
    && pip install --target /install --no-deps /build/api

FROM python:${PYTHON_VERSION}-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    LOG_LEVEL=INFO
WORKDIR /app
RUN groupadd -g 10001 ga && useradd -m -u 10001 -g ga ga
COPY --from=builder /install /app
USER ga
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1
ENTRYPOINT ["python", "-m", "geoalpha_api"]

# Multi-stage build for the geo-alpha API gateway.
ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY ../quant /quant
COPY pyproject.toml ./
COPY src ./src
RUN pip install --upgrade pip hatchling \
    && pip install --target /install numpy scipy "fastapi>=0.110" "uvicorn[standard]>=0.27" "pydantic>=2.5" \
    && pip install --target /install --no-deps /quant \
    && pip install --target /install --no-deps .

FROM python:${PYTHON_VERSION}-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app LOG_LEVEL=INFO
WORKDIR /app
RUN groupadd -g 10001 ga && useradd -m -u 10001 -g ga ga
COPY --from=builder /install /app
USER ga
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1
ENTRYPOINT ["python", "-m", "geoalpha_api"]

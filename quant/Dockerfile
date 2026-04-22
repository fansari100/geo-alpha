# Multi-stage build for the geoalpha-quant Python package.
ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY src ./src
RUN pip install --upgrade pip hatchling \
    && pip install --target /install numpy scipy \
    && pip install --target /install --no-deps .

FROM python:${PYTHON_VERSION}-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
WORKDIR /app
RUN groupadd -g 10001 ga && useradd -m -u 10001 -g ga ga
COPY --from=builder /install /app
USER ga
CMD ["python", "-c", "import geoalpha_quant; print(geoalpha_quant.__version__)"]

# syntax=docker/dockerfile:1.7
#
# One Dockerfile, three targets:
#   core  — data versioning, snapshots, calibration, evaluation, gate, registry (multi-arch)
#   test  — core + dev deps; runs the test suite at build time
#   train — core + PaddleOCR/Paddle + PyTorch/timm (linux/amd64 only: no paddle arm64 wheels)

ARG PYTHON_IMAGE=python:3.12-slim-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.19

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS base
COPY --from=uv /uv /uvx /bin/
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
# git: DVC needs it. libgl1/libglib2.0-0: OpenCV (paddlex pulls the non-headless build).
RUN apt-get update \
 && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/* \
 && git config --system --add safe.directory '*'
WORKDIR /app

# ---------------------------------------------------------------- dependencies (cached layer)
FROM base AS deps
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# ---------------------------------------------------------------- core
FROM deps AS core
COPY README.md ./
COPY src ./src
COPY configs ./configs
# Editable install: when docker-compose mounts the repo at /app, the mounted src/ is used.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev
# WordNet for METEOR (ThaiOCRBench metric), baked in so evaluation runs offline.
ENV NLTK_DATA=/usr/local/share/nltk_data
RUN python -m nltk.downloader -q -d ${NLTK_DATA} wordnet omw-1.4
CMD ["parcel-ocr", "--help"]

# ---------------------------------------------------------------- test
FROM core AS test
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen
COPY tests ./tests
COPY params.yaml ./
RUN pytest

# ---------------------------------------------------------------- train
FROM core AS train
ARG TRAIN_VARIANT=cpu
ARG PADDLEOCR_REF=v3.7.0
COPY docker/train/requirements-${TRAIN_VARIANT}.txt /tmp/requirements-train.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python -r /tmp/requirements-train.txt
# PaddleOCR's training entry points (tools/train.py, tools/export_model.py) live in the repo,
# not in the pip package. Pinned to the same release as the paddleocr wheel.
RUN git clone --depth 1 --branch ${PADDLEOCR_REF} https://github.com/PaddlePaddle/PaddleOCR /opt/PaddleOCR
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python -r /opt/PaddleOCR/requirements.txt
ENV PARCEL_OCR_PADDLEOCR_HOME=/opt/PaddleOCR \
    PADDLE_PDX_CACHE_HOME=/app/artifacts/.paddlex
CMD ["parcel-ocr", "--help"]

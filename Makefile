# Everything runs in containers — nothing is installed on the host.
COMPOSE      := docker compose
RUN          := $(COMPOSE) --profile tools run --rm pipeline
RUN_TRAIN    := $(COMPOSE) --profile train run --rm train
GPU_COMPOSE  := $(COMPOSE) -f docker-compose.yml -f docker-compose.gpu.yml

# Train/notebook images build for the host architecture: Apple Silicon → linux/arm64 (native,
# no x86 emulation), x86 servers → linux/amd64. Override: make notebook TRAIN_PLATFORM=linux/amd64
HOST_ARCH      := $(shell uname -m | sed -e 's/x86_64/amd64/' -e 's/aarch64/arm64/')
TRAIN_PLATFORM ?= linux/$(HOST_ARCH)
export TRAIN_PLATFORM

.PHONY: help env build build-train up down test lint init base-models retrain status shell benchmark-thai notebook

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

env:             ## create .env from the template (edit secrets afterwards)
	@test -f .env || cp .env.example .env

build: env       ## build core + test + mlflow images
	$(COMPOSE) build init mlflow
	$(COMPOSE) --profile test build test

build-train: env ## build the training image (host architecture, see TRAIN_PLATFORM)
	$(COMPOSE) --profile train build train

up: env          ## start infra: postgres, seaweedfs, mlflow, label-studio
	$(COMPOSE) up -d

down:            ## stop infra (volumes kept)
	$(COMPOSE) down

test: env        ## run the test suite in the test image
	$(COMPOSE) --profile test run --rm test

lint: env        ## ruff in the test image
	$(COMPOSE) --profile test run --rm test ruff check src tests

init: up         ## create data dirs (DVC deps)
	$(RUN) parcel-ocr infra init-dirs

base-models:     ## one-off: download pretrained PaddleOCR models into artifacts/base, then dvc add
	$(RUN_TRAIN) sh -c "parcel-ocr fetch-base-models && dvc add artifacts/base"

retrain:         ## one supervised cycle (trigger → repro → gate → register challenger)
	$(RUN_TRAIN) parcel-ocr retrain

benchmark-thai: env ## ThaiOCRBench, 4 OCR tasks (LIMIT=20 smoke run; ARGS="--bundle artifacts/bundle")
	$(COMPOSE) --profile train run --rm --no-deps train \
		parcel-ocr benchmark-thai $(if $(LIMIT),--limit $(LIMIT)) $(ARGS)

notebook: env    ## JupyterLab (train image) on http://127.0.0.1:8888 — token from NOTEBOOK_TOKEN
	$(COMPOSE) --profile notebook run --rm --no-deps --service-ports notebook

status:          ## champion / challenger / previous
	$(RUN) parcel-ocr status

shell:           ## shell in the core image
	$(RUN) bash

.PHONY: help build up down test lint bench

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-8s %s\n", $$1, $$2}'

build: ## build the service image (PADDLE=gpu for CUDA)
	docker compose build

up:    ## run the API on http://localhost:8000 (docs at /docs)
	docker compose up -d

down:  ## stop it
	docker compose down

test:  ## unit tests in the image
	docker compose run --rm --no-deps ocr pytest

lint:  ## ruff in the image
	docker compose run --rm --no-deps ocr ruff check src tests

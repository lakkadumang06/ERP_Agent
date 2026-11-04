PYTHON ?= python
VENV   ?= .venv
BIN    := $(VENV)/bin
DOC    ?= data/sample_docs/po_001.txt

.PHONY: setup seed run cli test eval lint pdfs docker-build docker-up clean

setup:            ## create venv and install pinned dependencies
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	@test -f .env || cp .env.example .env
	@echo "Done. Edit .env and add your API keys."

seed:             ## (re)create data/erp.db with sample customers and products
	$(BIN)/python -m app.seed

run:              ## start the FastAPI server with auto-reload on :8000
	$(BIN)/uvicorn app.api:app --reload --port 8000

cli:              ## process one document: make cli DOC=data/sample_docs/err_over_stock.txt PROVIDER=anthropic
	$(BIN)/python scripts/run_cli.py $(DOC) $(if $(PROVIDER),--provider $(PROVIDER),)

test:             ## run the test-suite (no API keys needed)
	$(BIN)/python -m pytest -q

eval:             ## compare providers on the sample documents (needs API keys)
	$(BIN)/python evals/run_evals.py

lint:             ## ruff check + format check
	$(BIN)/ruff check app tests evals scripts
	$(BIN)/ruff format --check app tests evals scripts

pdfs:             ## regenerate the sample PDFs from the .txt sources
	$(BIN)/python scripts/make_pdfs.py

docker-build:
	docker compose build

docker-up:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

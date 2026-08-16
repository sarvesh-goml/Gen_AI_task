.PHONY: install docker-up docker-down setup-db seed-db ingest setup run-api run-ui run-cli-rag run-cli-agent test

install:
	pip install -r requirements.txt

docker-up:
	docker compose up -d postgres

docker-down:
	docker compose down

setup-db:
	python -m scripts.setup_db

seed-db:
	python -m scripts.seed_db

ingest:
	python -m scripts.ingest

# Full first-time setup: installs deps, starts Postgres, builds both the fleet DB and the
# vector knowledge base. Run `make run-api` and `make run-ui` (separate terminals) after.
setup: install docker-up
	@echo "Waiting for Postgres to become healthy..."
	@until docker inspect -f '{{.State.Health.Status}}' jarvis_postgres 2>/dev/null | grep -q healthy; do sleep 1; done
	$(MAKE) setup-db
	$(MAKE) seed-db
	$(MAKE) ingest
	@echo "Setup complete. Run 'make run-api' then 'make run-ui' in separate terminals."

run-api:
	uvicorn src.api.main:app --reload --port 8000

run-ui:
	streamlit run src/ui/streamlit_app.py

run-cli-rag:
	python -m src.ui.cli --mode rag

run-cli-agent:
	python -m src.ui.cli --mode agent

test:
	pytest tests/ -v

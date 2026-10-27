PY ?= .venv/bin/python

.PHONY: install test bench serve validate docker-up docker-bench clean

install:
	python3 -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[postgres,embeddings,dev]"

test:
	MF_OFFLINE=1 $(PY) -m pytest -q

validate:
	$(PY) -m memoryforge validate

bench:
	$(PY) -m memoryforge bench --seeds 10

serve:
	$(PY) -m memoryforge serve

docker-up:
	docker compose up --build app

docker-bench:
	docker compose run --rm --build bench

clean:
	rm -rf runs/*/ runs/*.sqlite3* .pytest_cache

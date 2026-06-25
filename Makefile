.PHONY: install test lint format docker-build docker-up docker-down clean k8s-deploy

# ── Setup ─────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"

# ── Testing ───────────────────────────────────────────────────
test:
	pytest tests/unit/ -v --cov=src --cov-report=term-missing

test-all:
	pytest tests/ -v --cov=src

test-fast:
	pytest tests/unit/ -v -x --timeout=60

# ── Code Quality ──────────────────────────────────────────────
lint:
	flake8 src/ tests/ --max-line-length=120
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/
	isort src/ tests/

format-check:
	black --check src/ tests/
	isort --check-only src/ tests/

# ── Docker ────────────────────────────────────────────────────
docker-build:
	docker build -t enterprise-hybrid-rag:latest .

docker-up:
	docker-compose up -d --build

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f rag-api

docker-restart:
	docker-compose restart rag-api

# ── Kubernetes ────────────────────────────────────────────────
k8s-deploy:
	kubectl apply -f deployment/kubernetes/namespace.yaml
	kubectl apply -f deployment/kubernetes/configmap.yaml
	kubectl apply -f deployment/kubernetes/deployment.yaml
	kubectl apply -f deployment/kubernetes/service.yaml
	kubectl apply -f deployment/kubernetes/ingress.yaml

k8s-status:
	kubectl get all -n rag-system

k8s-logs:
	kubectl logs -f deployment/rag-api -n rag-system

k8s-delete:
	kubectl delete namespace rag-system

# ── Development ───────────────────────────────────────────────
dev:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

ingest-sample:
	python scripts/ingest_sample.py

benchmark:
	python scripts/run_benchmark.py

# ── Cleanup ───────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov/ dist/ build/ *.egg-info/

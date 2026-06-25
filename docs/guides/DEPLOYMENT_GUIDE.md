# Enterprise RAG Deployment Guide

## Local Development

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/enterprise-hybrid-rag.git
cd enterprise-hybrid-rag
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # Fill in API keys

# 2. Start dependencies only
docker-compose up -d chromadb postgres redis

# 3. Run API in dev mode (hot reload)
make dev
# OR
uvicorn src.api.main:app --reload --port 8000

# 4. Test it
curl http://localhost:8000/health
curl -X POST http://localhost:8000/upload -F "file=@documents/transformer_paper.txt"
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is multi-head attention?"}'
```

## Docker Deployment

```bash
# Full stack (API + all services)
docker-compose up -d --build

# Check all containers
docker-compose ps

# View API logs
docker-compose logs -f rag-api

# Scale API horizontally
docker-compose up -d --scale rag-api=3

# Stop all
docker-compose down

# Stop and remove volumes (DESTRUCTIVE)
docker-compose down -v
```

## Kubernetes Deployment

```bash
# 1. Create namespace and configs
kubectl apply -f deployment/kubernetes/namespace.yaml
kubectl apply -f deployment/kubernetes/configmap.yaml

# 2. Create secrets (fill in real values first)
kubectl create secret generic rag-secrets \
  --from-literal=OPENAI_API_KEY="sk-your-key" \
  --from-literal=POSTGRES_USER="raguser" \
  --from-literal=POSTGRES_PASSWORD="strong-password" \
  --from-literal=SECRET_KEY="$(openssl rand -hex 32)" \
  -n rag-system

# 3. Deploy all services
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/service.yaml
kubectl apply -f deployment/kubernetes/ingress.yaml

# 4. Monitor deployment
kubectl get pods -n rag-system -w
kubectl rollout status deployment/rag-api -n rag-system

# 5. Check logs
kubectl logs -f deployment/rag-api -n rag-system

# 6. Scale
kubectl scale deployment rag-api --replicas=5 -n rag-system

# 7. Rolling update (zero-downtime)
kubectl set image deployment/rag-api \
  rag-api=your-registry/enterprise-hybrid-rag:v1.1.0 \
  -n rag-system

# 8. Rollback
kubectl rollout undo deployment/rag-api -n rag-system
```

## Monitoring Setup

```bash
# Access monitoring UIs
open http://localhost:9090    # Prometheus
open http://localhost:3000    # Grafana (admin/admin)
open http://localhost:8000/docs   # API Swagger

# Import Grafana dashboard
python -c "from src.monitoring.grafana_dashboard import save_dashboard; save_dashboard()"
# Then import deployment/grafana/dashboard.json in Grafana UI

# Query metrics in Prometheus
# Rate of requests: rate(enterprise_rag_requests_total[5m])
# P95 latency:     histogram_quantile(0.95, rate(enterprise_rag_request_duration_seconds_bucket[5m]))
# Error rate:      rate(enterprise_rag_errors_total[5m])
```

## Scaling Guide

### Vertical Scaling (single node)
- Increase CPU/memory limits in deployment.yaml
- Enable GPU for faster embeddings: `EMBEDDING_DEVICE=cuda`
- Increase `API_WORKERS` for CPU-bound tasks

### Horizontal Scaling (multiple nodes)
- Use HPA (already configured) for auto-scaling
- Switch to Qdrant cluster mode for distributed vector search
- Use PostgreSQL connection pooling (PgBouncer)
- Add Redis cluster for distributed caching

### Performance Tuning
```python
# Embedding optimization
EMBEDDING_BATCH_SIZE=128   # Larger batches = faster throughput
EMBEDDING_DEVICE=cuda       # GPU = 10x faster

# Retrieval tuning (accuracy vs speed tradeoff)
DENSE_TOP_K=50             # More candidates = better recall, slower
SPARSE_TOP_K=50
RERANK_TOP_K=10

# FAISS optimization for large collections
FAISS_INDEX_TYPE=IVFFlat   # Use IVF for >100K chunks
FAISS_NLIST=1000           # Clusters (sqrt(N) is good rule)
FAISS_NPROBE=50            # Clusters to search (higher = more accurate)
```

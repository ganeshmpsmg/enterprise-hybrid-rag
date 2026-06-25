# Production Best Practices

## Security
- Never commit .env files or secrets to git
- Use Kubernetes Secrets / AWS Secrets Manager / HashiCorp Vault
- Run containers as non-root (uid 1000)
- Enable HTTPS with cert-manager + Let's Encrypt
- Validate all file uploads (type, size, content)
- Rate limit all API endpoints (100 req/min per IP default)
- Rotate API keys regularly

## Reliability
- Use health checks at every layer (container, K8s, LB)
- Implement circuit breakers for LLM calls (tenacity retry)
- Set request timeouts for all external calls
- Use readiness probes to prevent traffic to unready pods
- Implement graceful shutdown for in-flight requests

## Performance
- Pre-load embedding model at startup (not per-request)
- Use batch embedding for ingestion (64-128 at a time)
- Cache popular queries with semantic cache (Redis)
- Use connection pooling for PostgreSQL (asyncpg pool)
- Profile and identify bottlenecks: embedding > reranking > LLM

## Observability  
- Structured JSON logging (correlate with request_id)
- Prometheus metrics for all pipeline stages
- Distributed tracing (OpenTelemetry) for request flows
- Grafana dashboards for operational visibility
- Alert on: high latency, high error rate, low disk space

## Data Quality
- Validate documents before ingestion (size, type, encoding)
- Monitor embedding quality (cosine similarity distributions)
- Detect and reject near-duplicate documents
- Version your indexes for rollback capability
- Regularly evaluate retrieval quality with RAGAS

## Cost Optimization
- Use gpt-4o-mini for low-stakes queries, gpt-4o for complex ones
- Cache LLM responses for identical queries
- Batch LLM API calls where possible
- Use FAISS/Qdrant on-disk storage for cost savings
- Monitor token usage and set hard limits per user

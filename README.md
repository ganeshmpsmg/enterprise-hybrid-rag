# Enterprise Hybrid RAG Search System for Machine Learning Documents

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![LangChain](https://img.shields.io/badge/LangChain-0.3-orange)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

A production-grade **Enterprise Hybrid Retrieval-Augmented Generation (RAG)** system designed for Machine Learning document search and question answering. This system combines **dense retrieval**, **sparse retrieval**, **Reciprocal Rank Fusion (RRF)**, and **Cross-Encoder re-ranking** to deliver highly accurate, context-aware answers from large ML document corpora.

## Architecture

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                   CLIENT / USER                          │
                    └──────────────────────────┬──────────────────────────────┘
                                               │
                                               ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │              FastAPI REST API Layer                      │
                    │   /upload  /search  /retrieve  /ask  /health            │
                    └──────┬────────────────────────────────────┬─────────────┘
                           │                                    │
               ┌───────────▼──────────┐            ┌───────────▼──────────┐
               │   Document Ingestion  │            │    Query Pipeline     │
               │  PDF / DOCX / TXT    │            │                      │
               └───────────┬──────────┘            └───────────┬──────────┘
                           │                                    │
               ┌───────────▼──────────┐            ┌───────────▼──────────┐
               │   Preprocessing      │            │   Query Expansion     │
               │  Clean/Normalize     │            │   + Rewriting         │
               └───────────┬──────────┘            └───────────┬──────────┘
                           │                                    │
               ┌───────────▼──────────┐            ┌───────────▼──────────┐
               │   Chunking           │            │   Hybrid Retrieval    │
               │  Semantic/Recursive  │            │  Dense + Sparse + RRF │
               └───────────┬──────────┘            └───────────┬──────────┘
                           │                                    │
               ┌───────────▼──────────┐            ┌───────────▼──────────┐
               │   Embedding          │            │   Cross-Encoder       │
               │  all-MiniLM-L6-v2   │            │   Re-Ranking          │
               └───────────┬──────────┘            └───────────┬──────────┘
                           │                                    │
               ┌───────────▼──────────┐            ┌───────────▼──────────┐
               │   Vector Stores      │            │   LLM Generation      │
               │  FAISS/Chroma/Qdrant │            │  GPT/Llama/Mistral   │
               └──────────────────────┘            └──────────────────────┘
                           │
               ┌───────────▼──────────┐
               │   PostgreSQL DB      │
               │  Metadata Storage    │
               └──────────────────────┘
```

## Key Features

- **Multi-format Ingestion**: PDF, DOCX, TXT, Markdown
- **Hybrid Retrieval**: BM25 sparse + Dense semantic + RRF fusion
- **Cross-Encoder Re-ranking**: ms-marco-MiniLM-L-6-v2
- **Query Expansion**: Synonym expansion + LLM-based rewriting
- **Multi-Vector Store**: FAISS, ChromaDB, Qdrant
- **Production API**: FastAPI with async support
- **Monitoring**: Prometheus + Grafana dashboards
- **Evaluation**: RAGAS + custom metrics (Precision@K, Recall@K, MRR, nDCG)
- **Docker & Kubernetes**: Full containerized deployment
- **CI/CD**: GitHub Actions pipeline

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/yourusername/enterprise-hybrid-rag.git
cd enterprise-hybrid-rag

# 2. Set up environment
cp .env.example .env
# Edit .env with your API keys

# 3. Docker deployment
docker-compose up -d

# 4. Upload documents
curl -X POST http://localhost:8000/upload \
  -F "file=@your_ml_paper.pdf"

# 5. Ask questions
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is transformer attention mechanism?"}'
```

## Project Structure

```
enterprise-hybrid-rag/
├── data/                    # Data storage
│   ├── raw/                 # Raw uploaded documents
│   ├── processed/           # Cleaned, chunked text
│   └── embeddings/          # Cached embeddings
├── documents/               # Sample ML documents
├── notebooks/               # Jupyter exploration notebooks
├── configs/                 # YAML configuration files
├── scripts/                 # Utility scripts
├── docs/                    # Documentation (MkDocs)
│   ├── api/                 # API reference
│   ├── architecture/        # Architecture diagrams
│   └── guides/              # User guides
├── deployment/              # Deployment manifests
│   ├── kubernetes/          # K8s YAML files
│   └── helm/                # Helm charts
├── docker/                  # Docker files
├── tests/                   # Test suite
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── e2e/                 # End-to-end tests
├── logs/                    # Application logs
├── src/                     # Source code
│   ├── ingestion/           # Document loaders
│   ├── preprocessing/       # Text cleaning
│   ├── chunking/            # Text chunking strategies
│   ├── embeddings/          # Embedding models
│   ├── vectorstore/         # Vector DB managers
│   ├── sparse_retrieval/    # BM25, TF-IDF
│   ├── dense_retrieval/     # Semantic search
│   ├── hybrid_retrieval/    # Hybrid + RRF
│   ├── reranker/            # Cross-encoder re-ranking
│   ├── llm/                 # LLM connectors
│   ├── api/                 # FastAPI routes
│   ├── evaluation/          # RAG evaluation
│   ├── monitoring/          # Metrics & logging
│   └── utils/               # Shared utilities
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── setup.py
└── .env.example
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| API Framework | FastAPI 0.115 |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Vector DB | FAISS + ChromaDB + Qdrant |
| Sparse Retrieval | BM25 (rank_bm25) + TF-IDF |
| Database | PostgreSQL 16 |
| LLM | OpenAI GPT-4 / Llama 3 / Mistral |
| RAG Framework | LangChain + LlamaIndex |
| Monitoring | Prometheus + Grafana |
| Testing | Pytest |
| Deployment | Docker + Kubernetes |
| CI/CD | GitHub Actions |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /upload | Upload and ingest documents |
| POST | /search | Hybrid search (no LLM) |
| POST | /retrieve | Pure retrieval with metadata |
| POST | /ask | Full RAG Q&A pipeline |
| GET | /health | System health check |
| GET | /metrics | Prometheus metrics |

## Evaluation Metrics

- **Precision@K**: Fraction of top-K retrieved docs that are relevant
- **Recall@K**: Fraction of relevant docs retrieved in top-K
- **MRR**: Mean Reciprocal Rank
- **nDCG**: Normalized Discounted Cumulative Gain
- **Faithfulness**: RAGAS metric for answer grounding
- **Answer Relevancy**: RAGAS metric for answer quality

## Docker Commands

```bash
# Build and start all services
docker-compose up -d --build

# View logs
docker-compose logs -f rag-api

# Stop services
docker-compose down

# Rebuild specific service
docker-compose up -d --build rag-api
```

## Kubernetes Commands

```bash
# Deploy to Kubernetes
kubectl apply -f deployment/kubernetes/

# Check pod status
kubectl get pods -n rag-system

# Scale API deployment
kubectl scale deployment rag-api --replicas=3 -n rag-system

# View logs
kubectl logs -f deployment/rag-api -n rag-system
```

## Monitoring

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **API Docs**: http://localhost:8000/docs
- **Metrics**: http://localhost:8000/metrics

## License

MIT License - see LICENSE file for details.

## Author

Built as a production-grade portfolio project demonstrating enterprise ML engineering, RAG systems, and MLOps best practices.

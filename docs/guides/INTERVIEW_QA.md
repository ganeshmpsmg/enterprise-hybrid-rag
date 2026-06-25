# Interview Questions & Answers: Enterprise RAG System

## RAG Fundamentals

**Q1: What is RAG and why is it better than fine-tuning for factual QA?**

RAG (Retrieval-Augmented Generation) retrieves relevant documents at query time and injects them 
into the LLM context. Advantages over fine-tuning:
- No retraining needed when knowledge changes (just re-index)
- Prevents hallucination by grounding answers in retrieved documents
- Can cite sources for auditability
- Cheaper: no GPU training required

**Q2: Explain the difference between dense and sparse retrieval.**

- **Sparse (BM25/TF-IDF)**: Term frequency-based. Exact keyword matching. Fast, interpretable, 
  excellent for rare terms and acronyms. Fails on vocabulary mismatch ("heart attack" vs "myocardial infarction").
- **Dense (Embeddings)**: Semantic similarity. Works on meaning, not exact terms. Handles synonyms 
  and paraphrases. Slower, requires GPU for large scale. Fails on rare/domain-specific terms.
- **Hybrid**: Combines both. State-of-the-art on BEIR benchmark (3-8% improvement over either alone).

**Q3: What is Reciprocal Rank Fusion (RRF) and why use it?**

RRF = sum[ 1/(k + rank_i) ] across all rankers.

Key properties:
- Parameter-free (no score normalization needed — only ranks matter)
- Robust to different score scales from different rankers
- k=60 is empirically shown to work well across tasks
- Produces superior results to linear score combination
- Simple to implement but highly effective

**Q4: Why use a two-stage retrieval pipeline (retrieval + reranking)?**

- Stage 1 (fast bi-encoder): Retrieves N candidates efficiently (O(1) with vector index)
- Stage 2 (slow cross-encoder): Precisely scores N candidates using full query-doc attention
- Cross-encoders see BOTH query and document simultaneously → much more accurate relevance score
- Tradeoff: Cross-encoder is O(N) with N candidates — only feasible on small set (10-50 docs)
- This two-stage approach achieves near-oracle accuracy at acceptable latency

**Q5: What is the "Lost in the Middle" problem in RAG?**

LLMs struggle to use information placed in the middle of long contexts. Performance is best for:
- Information at the BEGINNING of context
- Information at the END of context
- Worst for: information in the MIDDLE

Mitigation: Our `ContextBuilder` with `lost_in_middle` strategy places most relevant chunk first, 
second most relevant last, less relevant chunks in the middle.

## ML Concepts

**Q6: How does BM25 differ from TF-IDF?**

BM25 score(t,d) = IDF(t) × [TF(t,d) × (k1+1)] / [TF(t,d) + k1×(1-b+b×|d|/avgdl)]

Key differences:
- **Term saturation**: BM25 uses k1 to control diminishing returns for repeated terms (TF-IDF doesn't)
- **Length normalization**: b parameter normalizes for document length
- Empirically outperforms TF-IDF by 5-15% on standard IR benchmarks

**Q7: Explain sentence-transformers/all-MiniLM-L6-v2**

- BERT-like architecture, 6-layer, distilled from larger models
- 384-dimensional embeddings (vs BERT's 768)
- Trained with contrastive learning (semantic textual similarity tasks)
- 22MB model size, ~14k sentences/second on CPU
- Optimized for semantic similarity tasks — perfect for RAG retrieval

**Q8: What is nDCG and why is it better than accuracy for IR evaluation?**

nDCG@K = DCG@K / IDCG@K

- DCG gives more credit for relevant results appearing earlier in the ranking
- Normalized by ideal DCG (if all relevant docs were retrieved first)
- Unlike accuracy, it's graded (rewards partial credit for near-misses)
- Standard metric in TREC, BEIR, and MS MARCO evaluations

## MLOps Concepts

**Q9: How would you monitor a RAG system in production?**

Four pillars:
1. **Infrastructure**: CPU, memory, GPU utilization, pod health
2. **Service**: Latency (P50/P95/P99), throughput, error rate
3. **Data**: Documents indexed, chunks per document, embedding quality
4. **Model**: Faithfulness drift, answer quality degradation, retrieval precision

Key alerts:
- P95 latency > 5s → scale up or optimize
- Error rate > 1% → investigate
- Faithfulness score drops → check for data quality issues

**Q10: How do you handle document updates and deletions in a RAG system?**

- **ChromaDB/Qdrant**: Native deletion by doc_id → re-index new version
- **FAISS**: No deletion support → must rebuild entire index (use HNSW variant)
- **BM25**: Must rebuild from scratch with new corpus

Production pattern:
1. Mark document as deleted in PostgreSQL
2. Delete from vector store by doc_id
3. Rebuild sparse index (nightly job for large corpora)
4. Use versioned indexes for zero-downtime updates

## System Design

**Q11: How would you scale this system to 100M documents?**

1. **Vector store**: Qdrant distributed cluster with sharding
2. **Embedding**: Pre-compute embeddings offline with GPU workers
3. **BM25**: Distributed Elasticsearch or Solr instead of in-memory BM25
4. **Retrieval**: ANN (Approximate Nearest Neighbor) instead of exact search
5. **Caching**: Redis for popular queries (semantic cache using embedding similarity)
6. **API**: Horizontal pod autoscaling with 10-50 replicas
7. **Async**: Queue-based ingestion with Celery + RabbitMQ

**Q12: What is semantic caching and how does it improve RAG latency?**

Instead of caching by exact query string, embed the query and check if a semantically similar 
query was answered recently:

```python
# Check cache
cached = redis.get_by_embedding_similarity(query_emb, threshold=0.95)
if cached:
    return cached  # Sub-millisecond response

# Generate answer (expensive)
answer = rag_pipeline.run(query)
redis.store_with_embedding(query_emb, answer, ttl=3600)
```

This handles paraphrases: "What is attention?" and "Explain the attention mechanism" 
would return the same cached answer.

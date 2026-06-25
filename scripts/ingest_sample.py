#!/usr/bin/env python3
"""Script to ingest sample ML documents for testing."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_DOCS = [
    {
        "filename": "transformer_paper.txt",
        "content": """Attention Is All You Need

Abstract
The dominant sequence transduction models are based on complex recurrent or convolutional neural 
networks that include an encoder and a decoder. The best performing models also connect the encoder 
and decoder through an attention mechanism. We propose a new simple network architecture, the 
Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions 
entirely.

1. Introduction
Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, 
have been firmly established as state of the art approaches in sequence modeling and transduction 
problems such as language modeling and machine translation.

2. Model Architecture
The Transformer follows an encoder-decoder structure using stacked self-attention and point-wise, 
fully connected layers for both the encoder and decoder. The encoder maps an input sequence of 
symbol representations to a sequence of continuous representations. Given z, the decoder then 
generates an output sequence of symbols one element at a time.

3. Attention Mechanism
An attention function can be described as mapping a query and a set of key-value pairs to an output, 
where the query, keys, values, and output are all vectors. The output is computed as a weighted sum 
of the values, where the weight assigned to each value is computed by a compatibility function of 
the query with the corresponding key.

Scaled Dot-Product Attention
The input consists of queries and keys of dimension dk, and values of dimension dv. We compute the 
dot products of the query with all keys, divide each by sqrt(dk), and apply a softmax function to 
obtain the weights on the values.

Multi-Head Attention
Instead of performing a single attention function with dmodel-dimensional keys, values and queries, 
we found it beneficial to linearly project the queries, keys and values h times with different, 
learned linear projections to dk, dk and dv dimensions, respectively.

4. Results
On the WMT 2014 English-to-German translation task, the big transformer model outperforms the 
best previously reported models including ensembles by more than 2.0 BLEU, establishing a new 
state-of-the-art BLEU score of 28.4.
""",
    },
    {
        "filename": "rag_overview.txt",
        "content": """Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

Abstract
Large pre-trained language models store factual knowledge in their parameters, and achieve 
state-of-the-art results when fine-tuned on downstream NLP tasks. However, their ability to 
access and precisely manipulate knowledge is still limited, and hence on knowledge-intensive tasks, 
their performance lags behind task-specific architectures. 

Introduction
Retrieval-Augmented Generation (RAG) combines parametric memory (the LLM) with non-parametric 
memory (a retrieval system). The retrieval system provides relevant context documents that the 
LLM uses to generate accurate, grounded answers.

Key Components of RAG Systems

1. Document Ingestion Pipeline
   - PDF, DOCX, TXT document loading
   - Text extraction and cleaning
   - Metadata extraction (author, date, topic)
   - Text chunking with configurable overlap

2. Embedding and Indexing
   - Dense embeddings using sentence transformers
   - Vector database storage (FAISS, ChromaDB, Qdrant)
   - Sparse indexing using BM25

3. Hybrid Retrieval
   - Dense retrieval: semantic similarity search
   - Sparse retrieval: keyword-based BM25 search
   - Reciprocal Rank Fusion (RRF) for combining results

4. Re-ranking
   - Cross-encoder models provide precise relevance scoring
   - More accurate than bi-encoder but slower
   - Applied to smaller candidate sets after retrieval

5. Answer Generation
   - Context formatting with source citations
   - LLM prompt engineering for faithfulness
   - Streaming and synchronous generation

Evaluation Metrics
- Faithfulness: Does the answer stay grounded in context?
- Answer Relevancy: Is the answer relevant to the question?
- Context Precision: Are the retrieved documents relevant?
- Context Recall: Does the context contain the needed information?
""",
    },
]


def create_sample_documents():
    docs_dir = Path("documents")
    docs_dir.mkdir(exist_ok=True)
    for doc in SAMPLE_DOCS:
        path = docs_dir / doc["filename"]
        path.write_text(doc["content"])
        logger.info(f"Created sample document: {path}")


def main():
    logger.info("Creating sample documents...")
    create_sample_documents()

    logger.info("\nTo ingest documents via API:")
    for doc in SAMPLE_DOCS:
        logger.info(f"  curl -X POST http://localhost:8000/upload -F 'file=@documents/{doc['filename']}'")
    logger.info("\nTo ask questions:")
    logger.info("  curl -X POST http://localhost:8000/ask \\")
    logger.info("    -H 'Content-Type: application/json' \\")
    logger.info("    -d '{\"query\": \"How does the attention mechanism work in transformers?\"}'")


if __name__ == "__main__":
    main()

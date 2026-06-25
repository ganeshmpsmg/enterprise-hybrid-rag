-- Enterprise Hybrid RAG System - Database Initialization
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    doc_id VARCHAR(64) UNIQUE NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_type VARCHAR(10) NOT NULL,
    file_size_bytes BIGINT,
    title TEXT,
    authors TEXT[],
    topics TEXT[],
    total_pages INT DEFAULT 1,
    word_count INT DEFAULT 0,
    content_quality VARCHAR(20) DEFAULT 'unknown',
    publication_year INT,
    source_path TEXT,
    indexed_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Chunks table
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id VARCHAR(64) UNIQUE NOT NULL,
    doc_id VARCHAR(64) REFERENCES documents(doc_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INT NOT NULL,
    page_number INT,
    word_count INT,
    chunk_strategy VARCHAR(30) DEFAULT 'recursive',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Query logs table
CREATE TABLE IF NOT EXISTS query_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query TEXT NOT NULL,
    answer TEXT,
    retrieval_type VARCHAR(30),
    chunks_used INT,
    latency_ms FLOAT,
    model VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_documents_doc_id ON documents(doc_id);
CREATE INDEX IF NOT EXISTS idx_documents_file_type ON documents(file_type);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_query_logs_created ON query_logs(created_at);

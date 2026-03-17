-- Initialize PostgreSQL database schema for Noble RAG
-- This script is automatically executed on container startup

-- ==================== Create Extensions ====================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS age CASCADE;

-- ==================== Create Schemas ====================
CREATE SCHEMA IF NOT EXISTS rag;
SET search_path TO rag, public;

-- ==================== Document Metadata Table ====================
-- Tracks the status of each ingested document
CREATE TABLE IF NOT EXISTS rag.documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL UNIQUE,
    source_url TEXT,
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    file_size BIGINT,
    md5_hash VARCHAR(32),
    status VARCHAR(50) DEFAULT 'active' CHECK (status IN ('active', 'archived', 'processing', 'failed')),
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_by VARCHAR(255),
    updated_by VARCHAR(255)
);

CREATE INDEX idx_documents_filename ON rag.documents(filename);
CREATE INDEX idx_documents_status ON rag.documents(status);
CREATE INDEX idx_documents_ingested_at ON rag.documents(ingested_at DESC);
CREATE INDEX idx_documents_metadata ON rag.documents USING GIN(metadata);

-- ==================== Chunks Table ====================
-- Stores text chunks extracted from documents
CREATE TABLE IF NOT EXISTS rag.chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES rag.documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    content_length INT,
    embedding_model VARCHAR(255) DEFAULT 'text-embedding-3-small',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'active' CHECK (status IN ('active', 'archived')),
    UNIQUE(document_id, chunk_index)
);

CREATE INDEX idx_chunks_document_id ON rag.chunks(document_id);
CREATE INDEX idx_chunks_status ON rag.chunks(status);
CREATE INDEX idx_chunks_created_at ON rag.chunks(created_at DESC);
CREATE INDEX idx_chunks_content_trgm ON rag.chunks USING GIN(content gin_trgm_ops);
CREATE INDEX idx_chunks_metadata ON rag.chunks USING GIN(metadata);

-- ==================== Entities Table ====================
-- Stores extracted entities (people, organizations, locations, etc.)
CREATE TABLE IF NOT EXISTS rag.entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES rag.documents(id) ON DELETE CASCADE,
    entity_text VARCHAR(255) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    occurrences INT DEFAULT 1,
    first_occurrence INT,
    last_occurrence INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    UNIQUE(document_id, entity_text, entity_type)
);

CREATE INDEX idx_entities_document_id ON rag.entities(document_id);
CREATE INDEX idx_entities_entity_type ON rag.entities(entity_type);
CREATE INDEX idx_entities_entity_text ON rag.entities(entity_text);
CREATE INDEX idx_entities_metadata ON rag.entities USING GIN(metadata);

-- ==================== Relations Table ====================
-- Stores relationships between entities
CREATE TABLE IF NOT EXISTS rag.relations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES rag.documents(id) ON DELETE CASCADE,
    source_entity_id UUID REFERENCES rag.entities(id) ON DELETE CASCADE,
    target_entity_id UUID REFERENCES rag.entities(id) ON DELETE CASCADE,
    source_entity_text VARCHAR(255),
    target_entity_text VARCHAR(255),
    relation_type VARCHAR(100) NOT NULL,
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_relations_document_id ON rag.relations(document_id);
CREATE INDEX idx_relations_source_entity_id ON rag.relations(source_entity_id);
CREATE INDEX idx_relations_target_entity_id ON rag.relations(target_entity_id);
CREATE INDEX idx_relations_relation_type ON rag.relations(relation_type);
CREATE INDEX idx_relations_metadata ON rag.relations USING GIN(metadata);

-- ==================== Query History Table ====================
-- Stores user queries for analytics and improvement
CREATE TABLE IF NOT EXISTS rag.query_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255),
    user_query TEXT NOT NULL,
    query_embedding_model VARCHAR(255) DEFAULT 'text-embedding-3-small',
    results_count INT,
    response_time_ms INT,
    relevance_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    feedback VARCHAR(50) CHECK (feedback IN ('positive', 'negative', NULL))
);

CREATE INDEX idx_query_history_session_id ON rag.query_history(session_id);
CREATE INDEX idx_query_history_created_at ON rag.query_history(created_at DESC);
CREATE INDEX idx_query_history_feedback ON rag.query_history(feedback);

-- ==================== Create Views ====================
-- View for active documents only
CREATE OR REPLACE VIEW rag.active_documents AS
SELECT * FROM rag.documents WHERE status = 'active';

-- View for document statistics
CREATE OR REPLACE VIEW rag.document_stats AS
SELECT 
    d.id,
    d.filename,
    COUNT(c.id) as chunk_count,
    SUM(c.content_length) as total_content_length,
    COUNT(DISTINCT e.id) as entity_count,
    d.ingested_at,
    d.updated_at
FROM rag.documents d
LEFT JOIN rag.chunks c ON d.id = c.document_id
LEFT JOIN rag.entities e ON d.id = e.document_id
GROUP BY d.id, d.filename, d.ingested_at, d.updated_at;

-- ==================== Create Functions ====================
-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION rag.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at column
CREATE TRIGGER trigger_documents_updated_at BEFORE UPDATE ON rag.documents
FOR EACH ROW EXECUTE FUNCTION rag.update_updated_at_column();

CREATE TRIGGER trigger_chunks_updated_at BEFORE UPDATE ON rag.chunks
FOR EACH ROW EXECUTE FUNCTION rag.update_updated_at_column();

-- ==================== Sample Data (Optional - for testing) ====================
-- Uncomment the following lines to insert sample data for testing
/*
INSERT INTO rag.documents (filename, source_url, status)
VALUES ('sample_doc.txt', 'http://example.com/sample', 'active');

INSERT INTO rag.chunks (document_id, chunk_index, content, content_length)
SELECT id, 1, 'This is a sample chunk content for testing purposes.', 50
FROM rag.documents
WHERE filename = 'sample_doc.txt'
LIMIT 1;
*/

-- ==================== Grant Permissions ====================
-- Grant appropriate permissions to the rag_user
GRANT USAGE ON SCHEMA rag TO rag_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA rag TO rag_user;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA rag TO rag_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA rag TO rag_user;

-- Allow creating temporary tables
ALTER DEFAULT PRIVILEGES IN SCHEMA rag GRANT SELECT ON TABLES TO rag_user;

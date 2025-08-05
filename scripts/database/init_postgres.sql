-- =====================================================================================
-- PostgreSQL Database Initialization for GraphRAG Implementation
-- LightRAG + n8n integration with pgvector extension
-- =====================================================================================

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- =====================================================================================
-- LIGHTRAG SCHEMA
-- =====================================================================================

-- Create lightrag schema
CREATE SCHEMA IF NOT EXISTS lightrag;

-- Grant permissions
GRANT USAGE ON SCHEMA lightrag TO lightrag;
GRANT CREATE ON SCHEMA lightrag TO lightrag;

-- Key-Value Store for LightRAG metadata
CREATE TABLE IF NOT EXISTS lightrag.kv_store (
    id BIGSERIAL PRIMARY KEY,
    key VARCHAR(512) UNIQUE NOT NULL,
    value JSONB,
    value_type VARCHAR(50) DEFAULT 'json',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    version INTEGER DEFAULT 1,
    metadata JSONB DEFAULT '{}',
    
    CONSTRAINT kv_store_key_not_empty CHECK (length(key) > 0)
);

-- Indexes for kv_store
CREATE INDEX IF NOT EXISTS idx_kv_store_key ON lightrag.kv_store USING btree(key);
CREATE INDEX IF NOT EXISTS idx_kv_store_created_at ON lightrag.kv_store USING btree(created_at);
CREATE INDEX IF NOT EXISTS idx_kv_store_value_type ON lightrag.kv_store USING btree(value_type);
CREATE INDEX IF NOT EXISTS idx_kv_store_metadata ON lightrag.kv_store USING gin(metadata);

-- Vector embeddings storage
CREATE TABLE IF NOT EXISTS lightrag.vectors (
    id BIGSERIAL PRIMARY KEY,
    key VARCHAR(512) UNIQUE NOT NULL,
    vector vector(1536), -- OpenAI text-embedding-3-large dimension
    content TEXT,
    content_hash VARCHAR(64),
    source_type VARCHAR(50) DEFAULT 'document',
    source_id VARCHAR(255),
    chunk_index INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT vectors_key_not_empty CHECK (length(key) > 0),
    CONSTRAINT vectors_chunk_index_non_negative CHECK (chunk_index >= 0)
);

-- Indexes for vectors
CREATE INDEX IF NOT EXISTS idx_vectors_key ON lightrag.vectors USING btree(key);
CREATE INDEX IF NOT EXISTS idx_vectors_vector ON lightrag.vectors USING ivfflat (vector vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_vectors_content_hash ON lightrag.vectors USING btree(content_hash);
CREATE INDEX IF NOT EXISTS idx_vectors_source ON lightrag.vectors USING btree(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_vectors_metadata ON lightrag.vectors USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_vectors_content_trgm ON lightrag.vectors USING gin(content gin_trgm_ops);

-- Document status tracking
CREATE TABLE IF NOT EXISTS lightrag.doc_status (
    id BIGSERIAL PRIMARY KEY,
    source_id VARCHAR(255) UNIQUE NOT NULL,
    document_path TEXT,
    content_hash VARCHAR(64) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    file_size BIGINT,
    mime_type VARCHAR(100),
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT doc_status_source_id_not_empty CHECK (length(source_id) > 0),
    CONSTRAINT doc_status_valid_status CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'skipped')),
    CONSTRAINT doc_status_retry_count_non_negative CHECK (retry_count >= 0)
);

-- Indexes for doc_status
CREATE INDEX IF NOT EXISTS idx_doc_status_source_id ON lightrag.doc_status USING btree(source_id);
CREATE INDEX IF NOT EXISTS idx_doc_status_status ON lightrag.doc_status USING btree(status);
CREATE INDEX IF NOT EXISTS idx_doc_status_content_hash ON lightrag.doc_status USING btree(content_hash);
CREATE INDEX IF NOT EXISTS idx_doc_status_created_at ON lightrag.doc_status USING btree(created_at);
CREATE INDEX IF NOT EXISTS idx_doc_status_metadata ON lightrag.doc_status USING gin(metadata);

-- Knowledge graph entities
CREATE TABLE IF NOT EXISTS lightrag.entities (
    id BIGSERIAL PRIMARY KEY,
    entity_id UUID DEFAULT uuid_generate_v4() UNIQUE,
    name VARCHAR(500) NOT NULL,
    type VARCHAR(100),
    description TEXT,
    properties JSONB DEFAULT '{}',
    confidence_score REAL DEFAULT 0.0,
    mention_count INTEGER DEFAULT 1,
    first_seen_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT entities_name_not_empty CHECK (length(name) > 0),
    CONSTRAINT entities_confidence_valid CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    CONSTRAINT entities_mention_count_positive CHECK (mention_count > 0)
);

-- Indexes for entities
CREATE INDEX IF NOT EXISTS idx_entities_name ON lightrag.entities USING btree(name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON lightrag.entities USING btree(type);
CREATE INDEX IF NOT EXISTS idx_entities_confidence ON lightrag.entities USING btree(confidence_score);
CREATE INDEX IF NOT EXISTS idx_entities_properties ON lightrag.entities USING gin(properties);
CREATE INDEX IF NOT EXISTS idx_entities_name_trgm ON lightrag.entities USING gin(name gin_trgm_ops);

-- Knowledge graph relationships
CREATE TABLE IF NOT EXISTS lightrag.relationships (
    id BIGSERIAL PRIMARY KEY,
    relationship_id UUID DEFAULT uuid_generate_v4() UNIQUE,
    source_entity_id UUID NOT NULL REFERENCES lightrag.entities(entity_id) ON DELETE CASCADE,
    target_entity_id UUID NOT NULL REFERENCES lightrag.entities(entity_id) ON DELETE CASCADE,
    relationship_type VARCHAR(100) NOT NULL,
    description TEXT,
    properties JSONB DEFAULT '{}',
    weight REAL DEFAULT 1.0,
    confidence_score REAL DEFAULT 0.0,
    evidence_count INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT relationships_different_entities CHECK (source_entity_id != target_entity_id),
    CONSTRAINT relationships_weight_positive CHECK (weight > 0.0),
    CONSTRAINT relationships_confidence_valid CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    CONSTRAINT relationships_evidence_count_positive CHECK (evidence_count > 0)
);

-- Indexes for relationships
CREATE INDEX IF NOT EXISTS idx_relationships_source ON lightrag.relationships USING btree(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON lightrag.relationships USING btree(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON lightrag.relationships USING btree(relationship_type);
CREATE INDEX IF NOT EXISTS idx_relationships_weight ON lightrag.relationships USING btree(weight);
CREATE INDEX IF NOT EXISTS idx_relationships_confidence ON lightrag.relationships USING btree(confidence_score);
CREATE INDEX IF NOT EXISTS idx_relationships_properties ON lightrag.relationships USING gin(properties);

-- =====================================================================================
-- N8N SCHEMA
-- =====================================================================================

-- Create n8n schema
CREATE SCHEMA IF NOT EXISTS n8n;

-- Grant permissions
GRANT USAGE ON SCHEMA n8n TO n8n;
GRANT CREATE ON SCHEMA n8n TO n8n;

-- Enhanced documents table for n8n workflow
CREATE TABLE IF NOT EXISTS n8n.documents_v2 (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    embedding vector(1536),
    title VARCHAR(1000),
    file_id VARCHAR(255),
    file_path TEXT,
    file_type VARCHAR(50),
    file_size BIGINT,
    chunk_index INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 1,
    metadata JSONB DEFAULT '{}',
    processing_metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT documents_v2_content_not_empty CHECK (length(content) > 0),
    CONSTRAINT documents_v2_chunk_index_non_negative CHECK (chunk_index >= 0),
    CONSTRAINT documents_v2_total_chunks_positive CHECK (total_chunks > 0),
    CONSTRAINT documents_v2_chunk_index_valid CHECK (chunk_index < total_chunks)
);

-- Indexes for documents_v2
CREATE INDEX IF NOT EXISTS idx_documents_v2_embedding ON n8n.documents_v2 USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_documents_v2_content_hash ON n8n.documents_v2 USING btree(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_v2_file_id ON n8n.documents_v2 USING btree(file_id);
CREATE INDEX IF NOT EXISTS idx_documents_v2_file_type ON n8n.documents_v2 USING btree(file_type);
CREATE INDEX IF NOT EXISTS idx_documents_v2_tags ON n8n.documents_v2 USING gin(tags);
CREATE INDEX IF NOT EXISTS idx_documents_v2_metadata ON n8n.documents_v2 USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_documents_v2_created_at ON n8n.documents_v2 USING btree(created_at);
CREATE INDEX IF NOT EXISTS idx_documents_v2_content_trgm ON n8n.documents_v2 USING gin(content gin_trgm_ops);

-- Record manager for tracking document processing
CREATE TABLE IF NOT EXISTS n8n.record_manager (
    id BIGSERIAL PRIMARY KEY,
    namespace VARCHAR(100) DEFAULT 'default',
    source_id VARCHAR(255) NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    group_id VARCHAR(255),
    cleanup_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT record_manager_source_id_not_empty CHECK (length(source_id) > 0),
    CONSTRAINT record_manager_status_valid CHECK (status IN ('active', 'deleted', 'archived', 'processing')),
    UNIQUE(namespace, source_id)
);

-- Indexes for record_manager
CREATE INDEX IF NOT EXISTS idx_record_manager_source ON n8n.record_manager USING btree(namespace, source_id);
CREATE INDEX IF NOT EXISTS idx_record_manager_status ON n8n.record_manager USING btree(status);
CREATE INDEX IF NOT EXISTS idx_record_manager_group_id ON n8n.record_manager USING btree(group_id);
CREATE INDEX IF NOT EXISTS idx_record_manager_expires_at ON n8n.record_manager USING btree(expires_at);

-- Conversation history for chat functionality
CREATE TABLE IF NOT EXISTS n8n.conversation_history (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    conversation_id UUID DEFAULT uuid_generate_v4(),
    user_query TEXT NOT NULL,
    ai_response TEXT,
    response_metadata JSONB DEFAULT '{}',
    query_embedding vector(1536),
    response_embedding vector(1536),
    feedback_score INTEGER,
    feedback_comment TEXT,
    processing_time_ms INTEGER,
    token_usage JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT conversation_history_user_query_not_empty CHECK (length(user_query) > 0),
    CONSTRAINT conversation_history_feedback_score_valid CHECK (feedback_score IS NULL OR (feedback_score >= 1 AND feedback_score <= 5)),
    CONSTRAINT conversation_history_processing_time_non_negative CHECK (processing_time_ms IS NULL OR processing_time_ms >= 0)
);

-- Indexes for conversation_history
CREATE INDEX IF NOT EXISTS idx_conversation_history_session ON n8n.conversation_history USING btree(session_id);
CREATE INDEX IF NOT EXISTS idx_conversation_history_conversation_id ON n8n.conversation_history USING btree(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversation_history_created_at ON n8n.conversation_history USING btree(created_at);
CREATE INDEX IF NOT EXISTS idx_conversation_history_query_embedding ON n8n.conversation_history USING ivfflat (query_embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_conversation_history_response_embedding ON n8n.conversation_history USING ivfflat (response_embedding vector_cosine_ops) WITH (lists = 100);

-- =====================================================================================
-- SHARED SCHEMA (for cross-system functionality)
-- =====================================================================================

-- Create shared schema
CREATE SCHEMA IF NOT EXISTS shared;

-- System metrics and monitoring
CREATE TABLE IF NOT EXISTS shared.system_metrics (
    id BIGSERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC NOT NULL,
    metric_type VARCHAR(50) DEFAULT 'gauge',
    tags JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    source_system VARCHAR(50) NOT NULL,
    
    CONSTRAINT system_metrics_metric_name_not_empty CHECK (length(metric_name) > 0),
    CONSTRAINT system_metrics_source_system_not_empty CHECK (length(source_system) > 0)
);

-- Indexes for system_metrics
CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON shared.system_metrics USING btree(metric_name);
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON shared.system_metrics USING btree(timestamp);
CREATE INDEX IF NOT EXISTS idx_system_metrics_source ON shared.system_metrics USING btree(source_system);
CREATE INDEX IF NOT EXISTS idx_system_metrics_tags ON shared.system_metrics USING gin(tags);

-- API usage tracking
CREATE TABLE IF NOT EXISTS shared.api_usage (
    id BIGSERIAL PRIMARY KEY,
    endpoint VARCHAR(200) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms INTEGER,
    request_size BIGINT,
    response_size BIGINT,
    user_agent TEXT,
    ip_address INET,
    api_key_hash VARCHAR(64),
    error_message TEXT,
    request_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT api_usage_endpoint_not_empty CHECK (length(endpoint) > 0),
    CONSTRAINT api_usage_method_valid CHECK (method IN ('GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS', 'HEAD')),
    CONSTRAINT api_usage_response_time_non_negative CHECK (response_time_ms IS NULL OR response_time_ms >= 0)
);

-- Indexes for api_usage
CREATE INDEX IF NOT EXISTS idx_api_usage_endpoint ON shared.api_usage USING btree(endpoint);
CREATE INDEX IF NOT EXISTS idx_api_usage_method ON shared.api_usage USING btree(method);
CREATE INDEX IF NOT EXISTS idx_api_usage_status_code ON shared.api_usage USING btree(status_code);
CREATE INDEX IF NOT EXISTS idx_api_usage_created_at ON shared.api_usage USING btree(created_at);
CREATE INDEX IF NOT EXISTS idx_api_usage_api_key_hash ON shared.api_usage USING btree(api_key_hash);
CREATE INDEX IF NOT EXISTS idx_api_usage_ip_address ON shared.api_usage USING btree(ip_address);

-- =====================================================================================
-- FUNCTIONS AND PROCEDURES
-- =====================================================================================

-- Hybrid search function for documents
CREATE OR REPLACE FUNCTION n8n.hybrid_search(
    query_text TEXT,
    query_embedding vector(1536) DEFAULT NULL,
    match_count INTEGER DEFAULT 10,
    similarity_threshold REAL DEFAULT 0.3,
    filter_metadata JSONB DEFAULT '{}'::jsonb,
    boost_recent BOOLEAN DEFAULT true
)
RETURNS TABLE (
    id BIGINT,
    content TEXT,
    similarity REAL,
    metadata JSONB,
    file_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE
)
LANGUAGE plpgsql
AS $$
DECLARE
    use_vector_search BOOLEAN := query_embedding IS NOT NULL;
    recency_boost_factor REAL := CASE WHEN boost_recent THEN 0.1 ELSE 0.0 END;
BEGIN
    IF use_vector_search THEN
        -- Vector similarity search with optional text search boost
        RETURN QUERY
        SELECT
            d.id,
            d.content,
            (1 - (d.embedding <=> query_embedding)) + 
            CASE 
                WHEN boost_recent THEN 
                    LEAST(recency_boost_factor, EXTRACT(EPOCH FROM (NOW() - d.created_at)) / 86400.0 / 30.0)
                ELSE 0.0 
            END AS similarity,
            d.metadata,
            d.file_id,
            d.created_at
        FROM n8n.documents_v2 d
        WHERE 
            (filter_metadata = '{}'::jsonb OR d.metadata @> filter_metadata)
            AND d.embedding IS NOT NULL
            AND (1 - (d.embedding <=> query_embedding)) >= similarity_threshold
        ORDER BY similarity DESC
        LIMIT match_count;
    ELSE
        -- Text-based search using trigrams
        RETURN QUERY
        SELECT 
            d.id,
            d.content,
            similarity(d.content, query_text) AS similarity,
            d.metadata,
            d.file_id,
            d.created_at
        FROM n8n.documents_v2 d
        WHERE 
            (filter_metadata = '{}'::jsonb OR d.metadata @> filter_metadata)
            AND d.content % query_text
            AND similarity(d.content, query_text) >= similarity_threshold
        ORDER BY similarity DESC
        LIMIT match_count;
    END IF;
END;
$$;

-- Update timestamp function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- =====================================================================================
-- TRIGGERS
-- =====================================================================================

-- Create update triggers for all tables with updated_at columns
DO $$
DECLARE
    table_names TEXT[] := ARRAY[
        'lightrag.kv_store',
        'lightrag.vectors', 
        'lightrag.doc_status',
        'lightrag.entities',
        'lightrag.relationships',
        'n8n.documents_v2',
        'n8n.record_manager',
        'n8n.conversation_history'
    ];
    table_name TEXT;
BEGIN
    FOREACH table_name IN ARRAY table_names
    LOOP
        EXECUTE format('
            CREATE TRIGGER update_%s_updated_at
            BEFORE UPDATE ON %s
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        ', replace(table_name, '.', '_'), table_name);
    END LOOP;
END $$;

-- =====================================================================================
-- VIEWS
-- =====================================================================================

-- View for entity relationship network
CREATE OR REPLACE VIEW lightrag.entity_network AS
SELECT 
    e1.name as source_entity,
    e1.type as source_type,
    r.relationship_type,
    e2.name as target_entity,
    e2.type as target_type,
    r.weight,
    r.confidence_score,
    r.created_at
FROM lightrag.relationships r
JOIN lightrag.entities e1 ON r.source_entity_id = e1.entity_id
JOIN lightrag.entities e2 ON r.target_entity_id = e2.entity_id;

-- View for document processing status
CREATE OR REPLACE VIEW shared.document_processing_status AS
SELECT 
    ds.source_id,
    ds.status as lightrag_status,
    ds.processing_completed_at as lightrag_completed_at,
    rm.status as n8n_status,
    rm.updated_at as n8n_updated_at,
    COUNT(dv.id) as n8n_document_count,
    ds.metadata as lightrag_metadata,
    rm.cleanup_metadata as n8n_metadata
FROM lightrag.doc_status ds
FULL OUTER JOIN n8n.record_manager rm ON ds.source_id = rm.source_id
LEFT JOIN n8n.documents_v2 dv ON rm.source_id = dv.file_id
GROUP BY ds.source_id, ds.status, ds.processing_completed_at, rm.status, rm.updated_at, ds.metadata, rm.cleanup_metadata;

-- =====================================================================================
-- PERMISSIONS
-- =====================================================================================

-- Grant permissions to lightrag user
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA lightrag TO lightrag;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA shared TO lightrag;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA lightrag TO lightrag;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA shared TO lightrag;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA lightrag TO lightrag;

-- Grant permissions to n8n user
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA n8n TO n8n;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA shared TO n8n;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA n8n TO n8n;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA shared TO n8n;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA n8n TO n8n;

-- Grant read access to views
GRANT SELECT ON ALL TABLES IN SCHEMA lightrag TO n8n;
GRANT SELECT ON ALL TABLES IN SCHEMA n8n TO lightrag;

-- =====================================================================================
-- SAMPLE DATA (for testing)
-- =====================================================================================

-- Insert sample data for testing (only if tables are empty)
DO $$
BEGIN
    -- Only insert if tables are empty
    IF NOT EXISTS (SELECT 1 FROM lightrag.entities LIMIT 1) THEN
        INSERT INTO lightrag.entities (name, type, description, confidence_score) VALUES
        ('Artificial Intelligence', 'concept', 'The simulation of human intelligence in machines', 0.95),
        ('Machine Learning', 'concept', 'A subset of AI that enables computers to learn without explicit programming', 0.90),
        ('Deep Learning', 'concept', 'A subset of machine learning based on artificial neural networks', 0.88),
        ('Neural Networks', 'concept', 'Computing systems inspired by biological neural networks', 0.85);
        
        -- Create sample relationships
        INSERT INTO lightrag.relationships (source_entity_id, target_entity_id, relationship_type, weight, confidence_score)
        SELECT 
            e1.entity_id, e2.entity_id, 'part_of', 0.8, 0.85
        FROM lightrag.entities e1, lightrag.entities e2 
        WHERE e1.name = 'Machine Learning' AND e2.name = 'Artificial Intelligence'
        
        UNION ALL
        
        SELECT 
            e1.entity_id, e2.entity_id, 'part_of', 0.9, 0.88
        FROM lightrag.entities e1, lightrag.entities e2 
        WHERE e1.name = 'Deep Learning' AND e2.name = 'Machine Learning'
        
        UNION ALL
        
        SELECT 
            e1.entity_id, e2.entity_id, 'uses', 0.7, 0.80
        FROM lightrag.entities e1, lightrag.entities e2 
        WHERE e1.name = 'Deep Learning' AND e2.name = 'Neural Networks';
    END IF;
END $$;

-- =====================================================================================
-- MAINTENANCE PROCEDURES
-- =====================================================================================

-- Cleanup old conversation history (run periodically)
CREATE OR REPLACE FUNCTION shared.cleanup_old_conversations(days_to_keep INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM n8n.conversation_history 
    WHERE created_at < NOW() - INTERVAL '1 day' * days_to_keep;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    INSERT INTO shared.system_metrics (metric_name, metric_value, source_system, tags)
    VALUES ('conversations_cleaned', deleted_count, 'maintenance', '{"operation": "cleanup"}');
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Vacuum and analyze all tables (run periodically)
CREATE OR REPLACE FUNCTION shared.maintenance_vacuum()
RETURNS TEXT AS $$
DECLARE
    result TEXT := 'Maintenance completed: ';
BEGIN
    -- Vacuum and analyze key tables
    VACUUM ANALYZE lightrag.vectors;
    VACUUM ANALYZE n8n.documents_v2;
    VACUUM ANALYZE n8n.conversation_history;
    VACUUM ANALYZE shared.api_usage;
    
    result := result || 'VACUUM ANALYZE completed';
    
    INSERT INTO shared.system_metrics (metric_name, metric_value, source_system, tags)
    VALUES ('maintenance_vacuum', 1, 'maintenance', '{"operation": "vacuum"}');
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- =====================================================================================
-- COMPLETION MESSAGE
-- =====================================================================================

DO $$
BEGIN
    RAISE NOTICE 'GraphRAG PostgreSQL database initialization completed successfully!';
    RAISE NOTICE 'Schemas created: lightrag, n8n, shared';
    RAISE NOTICE 'Extensions enabled: uuid-ossp, vector, pg_trgm, btree_gin';
    RAISE NOTICE 'Sample data inserted for testing';
    RAISE NOTICE 'Maintenance procedures created';
END $$;
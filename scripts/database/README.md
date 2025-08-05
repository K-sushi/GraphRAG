# Database Setup for GraphRAG Implementation

This directory contains database initialization scripts for the GraphRAG Implementation project, supporting both LightRAG and n8n workflows with PostgreSQL and pgvector.

## Overview

The database setup creates a comprehensive PostgreSQL database with:

- **LightRAG Schema**: Knowledge graph storage with entities, relationships, and vector embeddings
- **n8n Schema**: Document processing workflow with conversation history
- **Shared Schema**: System metrics and API usage tracking
- **pgvector Extension**: Vector similarity search for embeddings
- **Advanced Functions**: Hybrid search, maintenance procedures

## Quick Start

### Prerequisites

1. **PostgreSQL 15+** with **pgvector extension**
2. **Python 3.8+** with required packages:
   ```bash
   pip install asyncpg psycopg2-binary
   ```

### Option 1: Automated Setup (Recommended)

```bash
# Set environment variables
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=lightrag_production
export POSTGRES_USER=lightrag
export POSTGRES_PASSWORD=your_secure_password

# Run setup script
python setup_database.py
```

### Option 2: Manual Setup

```bash
# 1. Create database and users manually
psql -U postgres -c "CREATE DATABASE lightrag_production;"
psql -U postgres -c "CREATE USER lightrag WITH PASSWORD 'your_password';"
psql -U postgres -c "CREATE USER n8n WITH PASSWORD 'your_password';"

# 2. Run initialization SQL
psql -U postgres -d lightrag_production -f init_postgres.sql
```

### Option 3: Docker Setup

```bash
# Using the provided Docker Compose
cd ../../deployment/docker
docker-compose up -d postgres-db

# Wait for startup, then run setup
python ../../scripts/database/setup_database.py
```

## Database Schema

### LightRAG Schema (`lightrag`)

#### Core Tables

**`lightrag.kv_store`** - Key-value storage for LightRAG metadata
- Stores configuration, cache data, and system state
- JSONB values with versioning support

**`lightrag.vectors`** - Vector embeddings storage
- OpenAI text-embedding-3-large (1536 dimensions)
- Full-text search with pg_trgm
- Chunk-based document storage

**`lightrag.entities`** - Knowledge graph entities
- Entity extraction from documents
- Type classification and confidence scoring
- Mention tracking and temporal data

**`lightrag.relationships`** - Knowledge graph relationships
- Entity-to-entity connections
- Weighted relationships with confidence scores
- Evidence counting and metadata

**`lightrag.doc_status`** - Document processing status
- Processing pipeline tracking
- Error handling and retry logic
- Content hash-based deduplication

### n8n Schema (`n8n`)

#### Core Tables

**`n8n.documents_v2`** - Enhanced document storage
- Document chunks with embeddings
- File metadata and processing status
- Full-text search capabilities

**`n8n.record_manager`** - Document lifecycle management
- Source tracking and cleanup
- Namespace organization
- Expiration and archival

**`n8n.conversation_history`** - Chat and query history
- Session-based conversation tracking
- Query/response pairs with embeddings
- Performance metrics and feedback

### Shared Schema (`shared`)

#### Monitoring Tables

**`shared.system_metrics`** - System performance metrics
- Time-series metrics collection
- Tag-based filtering
- Multi-system support

**`shared.api_usage`** - API usage tracking
- Request/response logging
- Performance monitoring
- Security audit trail

## Features

### Vector Search

**Hybrid Search Function**:
```sql
SELECT * FROM n8n.hybrid_search(
    query_text := 'artificial intelligence',
    query_embedding := '[0.1, 0.2, ...]'::vector(1536),
    match_count := 10,
    similarity_threshold := 0.3
);
```

**Features**:
- Vector similarity search with pgvector
- Text-based search with trigrams
- Combined hybrid search
- Metadata filtering
- Recency boosting

### Knowledge Graph Views

**Entity Network View**:
```sql
SELECT * FROM lightrag.entity_network 
WHERE source_entity = 'Machine Learning';
```

**Document Processing Status**:
```sql
SELECT * FROM shared.document_processing_status
WHERE lightrag_status = 'completed';
```

### Maintenance Functions

**Cleanup Old Data**:
```sql
SELECT shared.cleanup_old_conversations(30); -- Keep 30 days
```

**Database Maintenance**:
```sql
SELECT shared.maintenance_vacuum(); -- Vacuum and analyze
```

## Configuration

### Environment Variables

```bash
# Database Connection
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=lightrag_production
POSTGRES_USER=lightrag
POSTGRES_PASSWORD=your_secure_password

# Admin Access (for setup)
POSTGRES_ADMIN_USER=postgres
POSTGRES_ADMIN_PASSWORD=admin_password

# User Passwords
LIGHTRAG_PASSWORD=lightrag_password
N8N_PASSWORD=n8n_password
```

### Setup Script Options

```bash
# Full setup with sample data
python setup_database.py

# Setup without sample data
python setup_database.py --no-sample-data

# Verify existing setup
python setup_database.py --verify-only

# Custom configuration
python setup_database.py \
  --host localhost \
  --port 5432 \
  --database my_graphrag_db \
  --user my_user \
  --password my_password
```

## Monitoring and Maintenance

### Health Checks

```sql
-- Check table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname IN ('lightrag', 'n8n', 'shared')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Check vector index performance
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM lightrag.vectors 
ORDER BY vector <=> '[0.1,0.2,0.3]'::vector(1536) 
LIMIT 10;
```

### Performance Tuning

**PostgreSQL Configuration**:
```ini
# postgresql.conf optimizations for vector workloads
shared_preload_libraries = 'vector'
max_connections = 100
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
```

**Vector Index Tuning**:
```sql
-- Optimize vector indexes based on data size
-- For < 1M vectors: lists = 100
-- For 1M+ vectors: lists = sqrt(row_count)

-- Reindex if needed
REINDEX INDEX lightrag.idx_vectors_vector;
```

### Backup and Recovery

**Backup Script**:
```bash
#!/bin/bash
# backup_graphrag.sh

DB_NAME="lightrag_production"
BACKUP_DIR="/backups/graphrag"
DATE=$(date +%Y%m%d_%H%M%S)

# Create full backup
pg_dump -h localhost -U postgres -d $DB_NAME \
  --no-owner --no-privileges \
  -f "$BACKUP_DIR/graphrag_$DATE.sql"

# Compress
gzip "$BACKUP_DIR/graphrag_$DATE.sql"

echo "Backup completed: graphrag_$DATE.sql.gz"
```

**Restore Script**:
```bash
#!/bin/bash
# restore_graphrag.sh

BACKUP_FILE=$1
DB_NAME="lightrag_production"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.sql.gz>"
    exit 1
fi

# Decompress and restore
gunzip -c "$BACKUP_FILE" | psql -h localhost -U postgres -d $DB_NAME
```

### Security

**User Permissions**:
```sql
-- Verify user permissions
SELECT 
    grantee, 
    table_schema, 
    table_name, 
    privilege_type 
FROM information_schema.table_privileges 
WHERE grantee IN ('lightrag', 'n8n')
ORDER BY grantee, table_schema, table_name;
```

**Access Monitoring**:
```sql
-- Monitor database connections
SELECT 
    datname,
    usename,
    client_addr,
    state,
    query_start,
    query
FROM pg_stat_activity 
WHERE datname = 'lightrag_production'
ORDER BY query_start DESC;
```

## Troubleshooting

### Common Issues

**1. pgvector Extension Missing**
```bash
# Install pgvector
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# Restart PostgreSQL
sudo systemctl restart postgresql
```

**2. Permission Denied**
```sql
-- Grant missing permissions
GRANT USAGE ON SCHEMA lightrag TO lightrag;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA lightrag TO lightrag;
```

**3. Connection Issues**
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Check configuration
sudo -u postgres psql -c "SHOW config_file;"

# Test connection
psql -h localhost -U lightrag -d lightrag_production -c "\dt"
```

**4. Vector Index Performance**
```sql
-- Check index usage
SELECT 
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes 
WHERE schemaname = 'lightrag'
ORDER BY idx_scan DESC;

-- Rebuild vector indexes if needed
REINDEX INDEX CONCURRENTLY lightrag.idx_vectors_vector;
```

### Log Analysis

```bash
# PostgreSQL logs
sudo tail -f /var/log/postgresql/postgresql-15-main.log

# Query performance
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows
FROM pg_stat_statements 
ORDER BY total_time DESC 
LIMIT 10;
```

## Development and Testing

### Sample Data

The setup script includes sample data for:
- AI-related entities (Artificial Intelligence, Machine Learning, etc.)
- Entity relationships (part_of, uses, enables)
- Sample documents for testing search
- Conversation history examples

### Testing Queries

```sql
-- Test vector search
SELECT content, vector <=> '[0.1,0.2,0.3]'::vector(1536) as distance
FROM lightrag.vectors
ORDER BY distance
LIMIT 5;

-- Test hybrid search
SELECT * FROM n8n.hybrid_search(
    'machine learning applications',
    match_count := 5
);

-- Test entity relationships
SELECT * FROM lightrag.entity_network
WHERE source_entity ILIKE '%machine%';
```

## Integration

### LightRAG Configuration

```python
# config.py
DATABASE_URL = "postgresql://lightrag:password@localhost:5432/lightrag_production"
VECTOR_STORE_TYPE = "postgresql"
KV_STORE_TYPE = "postgresql"
```

### n8n Configuration

```json
{
  "database": {
    "type": "postgresdb",
    "host": "localhost",
    "port": 5432,
    "database": "lightrag_production",
    "username": "n8n",
    "password": "password"
  }
}
```

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review PostgreSQL and pgvector documentation
3. Check application logs for connection errors
4. Verify network connectivity and firewall settings

## License

This database schema and setup scripts are part of the GraphRAG Implementation project.
#!/bin/bash

# -----------------------------------------------------------------------------
# GraphRAG Implementation Setup Script
# Automated setup for LightRAG + n8n integration
# -----------------------------------------------------------------------------

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Global variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

# Logging functions
log() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Check Docker
    if ! command_exists docker; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    # Check Docker Compose
    if ! command_exists docker-compose && ! docker compose version >/dev/null 2>&1; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi
    
    # Check Python
    if ! command_exists python3; then
        error "Python 3 is not installed. Please install Python 3.8+ first."
    fi
    
    # Check Node.js (for n8n)
    if ! command_exists node; then
        warning "Node.js is not installed. This is required for n8n development."
    fi
    
    # Check Git
    if ! command_exists git; then
        error "Git is not installed. Please install Git first."
    fi
    
    success "System requirements check passed!"
}

# Create environment file
create_env_file() {
    log "Creating environment configuration..."
    
    if [[ -f "$ENV_FILE" ]]; then
        warning "Environment file already exists. Creating backup..."
        cp "$ENV_FILE" "$ENV_FILE.backup.$(date +%Y%m%d_%H%M%S)"
    fi
    
    # Copy example environment file
    if [[ -f "$PROJECT_ROOT/config/environment/.env.example" ]]; then
        cp "$PROJECT_ROOT/config/environment/.env.example" "$ENV_FILE"
        success "Environment file created from template"
    else
        error "Environment template file not found"
    fi
    
    # Generate random secrets
    log "Generating secure secrets..."
    
    # Generate API keys and secrets
    LIGHTRAG_API_KEY=$(openssl rand -hex 32 2>/dev/null || head -c 32 /dev/urandom | xxd -p -c 32)
    N8N_JWT_SECRET=$(openssl rand -hex 32 2>/dev/null || head -c 32 /dev/urandom | xxd -p -c 32)
    N8N_ENCRYPTION_KEY=$(openssl rand -hex 32 2>/dev/null || head -c 32 /dev/urandom | xxd -p -c 32)
    ENCRYPTION_KEY=$(openssl rand -hex 32 2>/dev/null || head -c 32 /dev/urandom | xxd -p -c 32)
    
    # Replace placeholders in .env file
    if command_exists sed; then
        sed -i.bak \
            -e "s/your_lightrag_api_key_here/$LIGHTRAG_API_KEY/g" \
            -e "s/your_jwt_secret_here/$N8N_JWT_SECRET/g" \
            -e "s/your_encryption_key_here/$N8N_ENCRYPTION_KEY/g" \
            -e "s/your_32_character_encryption_key_here/$ENCRYPTION_KEY/g" \
            "$ENV_FILE" && rm "$ENV_FILE.bak"
    fi
    
    success "Secure secrets generated and configured"
}

# Create necessary directories
create_directories() {
    log "Creating project directories..."
    
    # Create volume directories
    mkdir -p "$PROJECT_ROOT/volumes/lightrag"
    mkdir -p "$PROJECT_ROOT/volumes/n8n"
    mkdir -p "$PROJECT_ROOT/volumes/postgres"
    mkdir -p "$PROJECT_ROOT/volumes/neo4j/data"
    mkdir -p "$PROJECT_ROOT/volumes/neo4j/logs"
    mkdir -p "$PROJECT_ROOT/volumes/neo4j/import"
    mkdir -p "$PROJECT_ROOT/volumes/neo4j/plugins"
    
    # Create log directories
    mkdir -p "$PROJECT_ROOT/logs"
    
    # Create backup directories
    mkdir -p "$PROJECT_ROOT/backups"
    
    # Set permissions
    chmod 755 "$PROJECT_ROOT/volumes"
    chmod 755 "$PROJECT_ROOT/logs"
    chmod 755 "$PROJECT_ROOT/backups"
    
    success "Project directories created"
}

# Setup Python virtual environment
setup_python_env() {
    log "Setting up Python environment..."
    
    cd "$PROJECT_ROOT/lightrag/server"
    
    # Create virtual environment
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
        success "Python virtual environment created"
    else
        log "Python virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install requirements
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
        success "Python dependencies installed"
    else
        warning "requirements.txt not found"
    fi
    
    # Install development dependencies if in development mode
    if [[ "${1:-}" == "dev" ]] && [[ -f "requirements-dev.txt" ]]; then
        pip install -r requirements-dev.txt
        success "Development dependencies installed"
    fi
    
    deactivate
}

# Initialize database schemas
init_databases() {
    log "Initializing database schemas..."
    
    # Start PostgreSQL if not running
    cd "$PROJECT_ROOT"
    docker-compose -f deployment/docker/docker-compose.yml up -d postgres-db
    
    # Wait for PostgreSQL to be ready
    log "Waiting for PostgreSQL to be ready..."
    sleep 10
    
    # Create database schemas
    docker-compose -f deployment/docker/docker-compose.yml exec -T postgres-db psql -U "${POSTGRES_USER:-lightrag}" -d "${POSTGRES_DB:-lightrag_production}" << 'EOF'
-- Create pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create tables for LightRAG
CREATE TABLE IF NOT EXISTS lightrag_kv_store (
    id SERIAL PRIMARY KEY,
    key VARCHAR(255) UNIQUE NOT NULL,
    value TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS lightrag_vectors (
    id SERIAL PRIMARY KEY,
    key VARCHAR(255) UNIQUE NOT NULL,
    vector vector(1536),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS lightrag_doc_status (
    id SERIAL PRIMARY KEY,
    source_id VARCHAR(255) UNIQUE NOT NULL,
    content_hash VARCHAR(64),
    status VARCHAR(50),
    processed_at TIMESTAMP,
    metadata JSONB
);

-- Create tables for n8n
CREATE TABLE IF NOT EXISTS documents_v2 (
    id BIGSERIAL PRIMARY KEY,
    content TEXT,
    metadata JSONB,
    embedding vector(1536),
    file_id VARCHAR(255),
    content_hash VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS record_manager (
    id BIGSERIAL PRIMARY KEY,
    source_id VARCHAR(255) UNIQUE NOT NULL,
    content_hash VARCHAR(64),
    status VARCHAR(50),
    processed_at TIMESTAMP,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS conversation_history (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(255),
    user_query TEXT,
    ai_response TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_lightrag_vectors_vector ON lightrag_vectors USING ivfflat (vector vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_documents_v2_embedding ON documents_v2 USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_conversation_history_session ON conversation_history(session_id);
CREATE INDEX IF NOT EXISTS idx_record_manager_source ON record_manager(source_id);

-- Create hybrid search function
CREATE OR REPLACE FUNCTION hybrid_search(
    query_text TEXT,
    match_count INT DEFAULT 10,
    filter_metadata JSONB DEFAULT '{}'::jsonb
)
RETURNS TABLE (
    content TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.content,
        d.metadata,
        1 - (d.embedding <=> query_text::vector) AS similarity
    FROM documents_v2 d
    WHERE 
        (filter_metadata = '{}'::jsonb OR d.metadata @> filter_metadata)
    ORDER BY d.embedding <=> query_text::vector
    LIMIT match_count;
END;
$$;

EOF
    
    success "Database schemas initialized"
}

# Build Docker images
build_images() {
    local mode="${1:-production}"
    
    log "Building Docker images for $mode mode..."
    
    cd "$PROJECT_ROOT"
    
    if [[ "$mode" == "development" ]]; then
        docker-compose -f deployment/docker/docker-compose.dev.yml build
    else
        docker-compose -f deployment/docker/docker-compose.yml build
    fi
    
    success "Docker images built successfully"
}

# Start services
start_services() {
    local mode="${1:-production}"
    
    log "Starting services in $mode mode..."
    
    cd "$PROJECT_ROOT"
    
    if [[ "$mode" == "development" ]]; then
        docker-compose -f deployment/docker/docker-compose.dev.yml up -d
    else
        docker-compose -f deployment/docker/docker-compose.yml up -d
    fi
    
    # Wait for services to be ready
    log "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    check_services_health "$mode"
    
    success "Services started successfully"
}

# Check services health
check_services_health() {
    local mode="${1:-production}"
    
    log "Checking services health..."
    
    # Check LightRAG health
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        success "LightRAG server is healthy"
    else
        warning "LightRAG server health check failed"
    fi
    
    # Check n8n health
    if curl -f http://localhost:5678/healthz >/dev/null 2>&1; then
        success "n8n server is healthy"
    else
        warning "n8n server health check failed"
    fi
    
    # Check PostgreSQL
    if docker-compose exec -T postgres-db pg_isready >/dev/null 2>&1; then
        success "PostgreSQL is ready"
    else
        warning "PostgreSQL health check failed"
    fi
}

# Setup monitoring (optional)
setup_monitoring() {
    log "Setting up monitoring (optional)..."
    
    read -p "Do you want to enable monitoring (Prometheus + Grafana)? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "$PROJECT_ROOT"
        docker-compose -f deployment/docker/docker-compose.yml --profile monitoring up -d
        
        log "Monitoring stack started. Access Grafana at http://localhost:3000"
        log "Default credentials: admin / admin"
        
        success "Monitoring setup completed"
    else
        log "Skipping monitoring setup"
    fi
}

# Display completion information
show_completion_info() {
    local mode="${1:-production}"
    
    echo
    success "GraphRAG Implementation setup completed!"
    echo
    log "Services Information:"
    echo "  - LightRAG Server: http://localhost:8000"
    echo "  - LightRAG API Docs: http://localhost:8000/docs"
    echo "  - n8n Workflow UI: http://localhost:5678"
    echo "  - PostgreSQL: localhost:5432"
    
    if [[ "$mode" == "development" ]]; then
        echo "  - Neo4j Browser: http://localhost:7474"
        echo "  - Adminer (DB): http://localhost:8080"
        echo "  - Redis Commander: http://localhost:8081"
    fi
    
    echo
    log "Next Steps:"
    echo "  1. Configure your API keys in .env file"
    echo "  2. Import n8n workflows from n8n/workflows/"
    echo "  3. Test the setup with sample queries"
    echo "  4. Check logs: docker-compose logs -f"
    echo
    log "Documentation:"
    echo "  - Project README: ./README.md"
    echo "  - API Documentation: http://localhost:8000/docs"
    echo "  - Troubleshooting: ./docs/guides/troubleshooting.md"
    echo
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    
    cd "$PROJECT_ROOT"
    
    # Stop all services
    docker-compose -f deployment/docker/docker-compose.yml down
    docker-compose -f deployment/docker/docker-compose.dev.yml down
    
    # Remove volumes (optional)
    read -p "Do you want to remove all data volumes? This will delete all data! (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose -f deployment/docker/docker-compose.yml down -v
        docker-compose -f deployment/docker/docker-compose.dev.yml down -v
        rm -rf volumes/
        warning "All data volumes removed!"
    fi
    
    success "Cleanup completed"
}

# Main function
main() {
    local mode="production"
    local skip_build=false
    local skip_init=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dev|--development)
                mode="development"
                shift
                ;;
            --skip-build)
                skip_build=true
                shift
                ;;
            --skip-init)
                skip_init=true
                shift
                ;;
            --cleanup)
                cleanup
                exit 0
                ;;
            --help|-h)
                echo "GraphRAG Implementation Setup Script"
                echo
                echo "Usage: $0 [OPTIONS]"
                echo
                echo "Options:"
                echo "  --dev, --development    Setup in development mode"
                echo "  --skip-build           Skip Docker image building"
                echo "  --skip-init            Skip database initialization"
                echo "  --cleanup              Clean up all services and data"
                echo "  --help, -h             Show this help message"
                echo
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                ;;
        esac
    done
    
    echo "🚀 GraphRAG Implementation Setup"
    echo "================================="
    echo "Mode: $mode"
    echo
    
    # Run setup steps
    check_requirements
    create_env_file
    create_directories
    setup_python_env "$mode"
    
    if [[ "$skip_build" != true ]]; then
        build_images "$mode"
    fi
    
    if [[ "$skip_init" != true ]]; then
        init_databases
    fi
    
    start_services "$mode"
    
    if [[ "$mode" == "production" ]]; then
        setup_monitoring
    fi
    
    show_completion_info "$mode"
}

# Run main function with all arguments
main "$@"
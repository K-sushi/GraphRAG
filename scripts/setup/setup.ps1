# -----------------------------------------------------------------------------
# GraphRAG Implementation Setup Script (PowerShell)
# Automated setup for LightRAG + n8n integration on Windows
# -----------------------------------------------------------------------------

param(
    [switch]$Development,
    [switch]$SkipBuild,
    [switch]$SkipInit,
    [switch]$Cleanup,
    [switch]$Help
)

# Global variables
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path "$ScriptDir\..\..\"
$EnvFile = "$ProjectRoot\.env"

# Color functions
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
    exit 1
}

# Test if command exists
function Test-Command {
    param([string]$Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

# Check system requirements
function Test-Requirements {
    Write-Info "Checking system requirements..."
    
    # Check Docker
    if (-not (Test-Command "docker")) {
        Write-Error "Docker is not installed. Please install Docker Desktop first."
    }
    
    # Check Docker Compose
    if (-not (Test-Command "docker-compose") -and -not (docker compose version 2>$null)) {
        Write-Error "Docker Compose is not installed. Please install Docker Compose first."
    }
    
    # Check Python
    if (-not (Test-Command "python") -and -not (Test-Command "python3")) {
        Write-Error "Python 3 is not installed. Please install Python 3.8+ first."
    }
    
    # Check Node.js (for n8n)
    if (-not (Test-Command "node")) {
        Write-Warning "Node.js is not installed. This is required for n8n development."
    }
    
    # Check Git
    if (-not (Test-Command "git")) {
        Write-Error "Git is not installed. Please install Git first."
    }
    
    Write-Success "System requirements check passed!"
}

# Create environment file
function New-EnvironmentFile {
    Write-Info "Creating environment configuration..."
    
    if (Test-Path $EnvFile) {
        Write-Warning "Environment file already exists. Creating backup..."
        $backupFile = "$EnvFile.backup.$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        Copy-Item $EnvFile $backupFile
    }
    
    # Copy example environment file
    $exampleFile = "$ProjectRoot\config\environment\.env.example"
    if (Test-Path $exampleFile) {
        Copy-Item $exampleFile $EnvFile
        Write-Success "Environment file created from template"
    } else {
        Write-Error "Environment template file not found"
    }
    
    # Generate random secrets
    Write-Info "Generating secure secrets..."
    
    # Generate API keys and secrets using .NET Crypto
    $LightRAGApiKey = [System.Web.Security.Membership]::GeneratePassword(64, 0)
    $N8NJwtSecret = [System.Web.Security.Membership]::GeneratePassword(64, 0)
    $N8NEncryptionKey = [System.Web.Security.Membership]::GeneratePassword(64, 0)
    $EncryptionKey = [System.Web.Security.Membership]::GeneratePassword(64, 0)
    
    # Replace placeholders in .env file
    $content = Get-Content $EnvFile -Raw
    $content = $content -replace 'your_lightrag_api_key_here', $LightRAGApiKey
    $content = $content -replace 'your_jwt_secret_here', $N8NJwtSecret
    $content = $content -replace 'your_encryption_key_here', $N8NEncryptionKey
    $content = $content -replace 'your_32_character_encryption_key_here', $EncryptionKey
    
    Set-Content -Path $EnvFile -Value $content
    
    Write-Success "Secure secrets generated and configured"
}

# Create necessary directories
function New-ProjectDirectories {
    Write-Info "Creating project directories..."
    
    # Create volume directories
    New-Item -ItemType Directory -Force -Path "$ProjectRoot\volumes\lightrag" | Out-Null
    New-Item -ItemType Directory -Force -Path "$ProjectRoot\volumes\n8n" | Out-Null
    New-Item -ItemType Directory -Force -Path "$ProjectRoot\volumes\postgres" | Out-Null
    New-Item -ItemType Directory -Force -Path "$ProjectRoot\volumes\neo4j\data" | Out-Null
    New-Item -ItemType Directory -Force -Path "$ProjectRoot\volumes\neo4j\logs" | Out-Null
    New-Item -ItemType Directory -Force -Path "$ProjectRoot\volumes\neo4j\import" | Out-Null
    New-Item -ItemType Directory -Force -Path "$ProjectRoot\volumes\neo4j\plugins" | Out-Null
    
    # Create log directories
    New-Item -ItemType Directory -Force -Path "$ProjectRoot\logs" | Out-Null
    
    # Create backup directories
    New-Item -ItemType Directory -Force -Path "$ProjectRoot\backups" | Out-Null
    
    Write-Success "Project directories created"
}

# Setup Python virtual environment
function Set-PythonEnvironment {
    param([string]$Mode = "production")
    
    Write-Info "Setting up Python environment..."
    
    Set-Location "$ProjectRoot\lightrag\server"
    
    # Determine Python command
    $pythonCmd = if (Test-Command "python") { "python" } else { "python3" }
    
    # Create virtual environment
    if (-not (Test-Path "venv")) {
        & $pythonCmd -m venv venv
        Write-Success "Python virtual environment created"
    } else {
        Write-Info "Python virtual environment already exists"
    }
    
    # Activate virtual environment
    & ".\venv\Scripts\Activate.ps1"
    
    # Upgrade pip
    python -m pip install --upgrade pip setuptools wheel
    
    # Install requirements
    if (Test-Path "requirements.txt") {
        python -m pip install -r requirements.txt
        Write-Success "Python dependencies installed"
    } else {
        Write-Warning "requirements.txt not found"
    }
    
    # Install development dependencies if in development mode
    if ($Mode -eq "development" -and (Test-Path "requirements-dev.txt")) {
        python -m pip install -r requirements-dev.txt
        Write-Success "Development dependencies installed"
    }
    
    deactivate
    Set-Location $ProjectRoot
}

# Initialize database schemas
function Initialize-Databases {
    Write-Info "Initializing database schemas..."
    
    # Start PostgreSQL if not running
    Set-Location $ProjectRoot
    docker-compose -f deployment/docker/docker-compose.yml up -d postgres-db
    
    # Wait for PostgreSQL to be ready
    Write-Info "Waiting for PostgreSQL to be ready..."
    Start-Sleep 10
    
    # Create database schemas
    $sqlScript = @"
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
AS `$`$
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
`$`$;
"@
    
    $sqlScript | docker-compose -f deployment/docker/docker-compose.yml exec -T postgres-db psql -U lightrag -d lightrag_production
    
    Write-Success "Database schemas initialized"
}

# Build Docker images
function Build-Images {
    param([string]$Mode = "production")
    
    Write-Info "Building Docker images for $Mode mode..."
    
    Set-Location $ProjectRoot
    
    if ($Mode -eq "development") {
        docker-compose -f deployment/docker/docker-compose.dev.yml build
    } else {
        docker-compose -f deployment/docker/docker-compose.yml build
    }
    
    Write-Success "Docker images built successfully"
}

# Start services
function Start-Services {
    param([string]$Mode = "production")
    
    Write-Info "Starting services in $Mode mode..."
    
    Set-Location $ProjectRoot
    
    if ($Mode -eq "development") {
        docker-compose -f deployment/docker/docker-compose.dev.yml up -d
    } else {
        docker-compose -f deployment/docker/docker-compose.yml up -d
    }
    
    # Wait for services to be ready
    Write-Info "Waiting for services to be ready..."
    Start-Sleep 30
    
    # Check service health
    Test-ServicesHealth $Mode
    
    Write-Success "Services started successfully"
}

# Check services health
function Test-ServicesHealth {
    param([string]$Mode = "production")
    
    Write-Info "Checking services health..."
    
    # Check LightRAG health
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Success "LightRAG server is healthy"
        }
    } catch {
        Write-Warning "LightRAG server health check failed"
    }
    
    # Check n8n health
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5678/healthz" -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Success "n8n server is healthy"
        }
    } catch {
        Write-Warning "n8n server health check failed"
    }
    
    # Check PostgreSQL
    try {
        $result = docker-compose exec -T postgres-db pg_isready 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "PostgreSQL is ready"
        }
    } catch {
        Write-Warning "PostgreSQL health check failed"
    }
}

# Setup monitoring (optional)
function Set-Monitoring {
    Write-Info "Setting up monitoring (optional)..."
    
    $response = Read-Host "Do you want to enable monitoring (Prometheus + Grafana)? (y/N)"
    
    if ($response -match '^[Yy]$') {
        Set-Location $ProjectRoot
        docker-compose -f deployment/docker/docker-compose.yml --profile monitoring up -d
        
        Write-Info "Monitoring stack started. Access Grafana at http://localhost:3000"
        Write-Info "Default credentials: admin / admin"
        
        Write-Success "Monitoring setup completed"
    } else {
        Write-Info "Skipping monitoring setup"
    }
}

# Display completion information
function Show-CompletionInfo {
    param([string]$Mode = "production")
    
    Write-Host ""
    Write-Success "GraphRAG Implementation setup completed!"
    Write-Host ""
    Write-Info "Services Information:"
    Write-Host "  - LightRAG Server: http://localhost:8000"
    Write-Host "  - LightRAG API Docs: http://localhost:8000/docs"
    Write-Host "  - n8n Workflow UI: http://localhost:5678"
    Write-Host "  - PostgreSQL: localhost:5432"
    
    if ($Mode -eq "development") {
        Write-Host "  - Neo4j Browser: http://localhost:7474"
        Write-Host "  - Adminer (DB): http://localhost:8080"
        Write-Host "  - Redis Commander: http://localhost:8081"
    }
    
    Write-Host ""
    Write-Info "Next Steps:"
    Write-Host "  1. Configure your API keys in .env file"
    Write-Host "  2. Import n8n workflows from n8n/workflows/"
    Write-Host "  3. Test the setup with sample queries"
    Write-Host "  4. Check logs: docker-compose logs -f"
    Write-Host ""
    Write-Info "Documentation:"
    Write-Host "  - Project README: .\README.md"
    Write-Host "  - API Documentation: http://localhost:8000/docs"
    Write-Host "  - Troubleshooting: .\docs\guides\troubleshooting.md"
    Write-Host ""
}

# Cleanup function
function Invoke-Cleanup {
    Write-Info "Cleaning up..."
    
    Set-Location $ProjectRoot
    
    # Stop all services
    docker-compose -f deployment/docker/docker-compose.yml down
    docker-compose -f deployment/docker/docker-compose.dev.yml down
    
    # Remove volumes (optional)
    $response = Read-Host "Do you want to remove all data volumes? This will delete all data! (y/N)"
    
    if ($response -match '^[Yy]$') {
        docker-compose -f deployment/docker/docker-compose.yml down -v
        docker-compose -f deployment/docker/docker-compose.dev.yml down -v
        Remove-Item -Recurse -Force "volumes" -ErrorAction SilentlyContinue
        Write-Warning "All data volumes removed!"
    }
    
    Write-Success "Cleanup completed"
}

# Show help
function Show-Help {
    Write-Host "GraphRAG Implementation Setup Script (PowerShell)"
    Write-Host ""
    Write-Host "Usage: .\setup.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Development       Setup in development mode"
    Write-Host "  -SkipBuild        Skip Docker image building"
    Write-Host "  -SkipInit         Skip database initialization"
    Write-Host "  -Cleanup          Clean up all services and data"
    Write-Host "  -Help             Show this help message"
    Write-Host ""
}

# Main function
function Main {
    if ($Help) {
        Show-Help
        return
    }
    
    if ($Cleanup) {
        Invoke-Cleanup
        return
    }
    
    $mode = if ($Development) { "development" } else { "production" }
    
    Write-Host "🚀 GraphRAG Implementation Setup" -ForegroundColor Cyan
    Write-Host "=================================" -ForegroundColor Cyan
    Write-Host "Mode: $mode"
    Write-Host ""
    
    # Add required assembly for password generation
    Add-Type -AssemblyName System.Web
    
    # Run setup steps
    Test-Requirements
    New-EnvironmentFile
    New-ProjectDirectories
    Set-PythonEnvironment $mode
    
    if (-not $SkipBuild) {
        Build-Images $mode
    }
    
    if (-not $SkipInit) {
        Initialize-Databases
    }
    
    Start-Services $mode
    
    if ($mode -eq "production") {
        Set-Monitoring
    }
    
    Show-CompletionInfo $mode
}

# Run main function
Main
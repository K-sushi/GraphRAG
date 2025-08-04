#!/bin/bash

# -----------------------------------------------------------------------------
# GraphRAG Implementation Quick Start Script
# Rapid setup for development and testing
# -----------------------------------------------------------------------------

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🚀 GraphRAG Implementation Quick Start${NC}"
echo "======================================"
echo

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Step 1: Check prerequisites
echo -e "${BLUE}Step 1: Checking prerequisites...${NC}"
if ! command -v docker >/dev/null 2>&1; then
    echo -e "${YELLOW}❌ Docker not found. Please install Docker first.${NC}"
    exit 1
fi

if ! command -v docker-compose >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1; then
    echo -e "${YELLOW}❌ Docker Compose not found. Please install Docker Compose first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites OK${NC}"
echo

# Step 2: Environment setup
echo -e "${BLUE}Step 2: Setting up environment...${NC}"

if [[ ! -f ".env" ]]; then
    if [[ -f "config/environment/.env.example" ]]; then
        cp config/environment/.env.example .env
        echo -e "${GREEN}✅ Environment file created${NC}"
    else
        echo -e "${YELLOW}❌ Environment template not found${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Environment file exists${NC}"
fi

# Generate basic secrets if needed
if grep -q "your_.*_here" .env; then
    echo "Generating basic secrets..."
    
    # Simple secret generation for quick start
    LIGHTRAG_KEY=$(head -c 32 /dev/urandom | base64 | tr -d '\n' 2>/dev/null || echo "dev_lightrag_key_$(date +%s)")
    JWT_SECRET=$(head -c 32 /dev/urandom | base64 | tr -d '\n' 2>/dev/null || echo "dev_jwt_secret_$(date +%s)")
    
    sed -i.bak \
        -e "s/your_lightrag_api_key_here/$LIGHTRAG_KEY/g" \
        -e "s/your_jwt_secret_here/$JWT_SECRET/g" \
        -e "s/your_encryption_key_here/$JWT_SECRET/g" \
        -e "s/your_32_character_encryption_key_here/$JWT_SECRET/g" \
        .env && rm .env.bak
    
    echo -e "${GREEN}✅ Basic secrets generated${NC}"
fi

echo

# Step 3: Create directories
echo -e "${BLUE}Step 3: Creating directories...${NC}"
mkdir -p volumes/{lightrag,n8n,postgres,neo4j/{data,logs,import,plugins}}
mkdir -p logs backups
echo -e "${GREEN}✅ Directories created${NC}"
echo

# Step 4: Start core services
echo -e "${BLUE}Step 4: Starting core services...${NC}"
echo "This may take a few minutes on first run..."

# Use development compose for quick start
docker-compose -f deployment/docker/docker-compose.dev.yml up -d lightrag-dev n8n-dev postgres-dev

echo "Waiting for services to start..."
sleep 20

# Quick health check
echo -e "${BLUE}Step 5: Checking service health...${NC}"

# Check LightRAG
if curl -f http://localhost:8000/health >/dev/null 2>&1; then
    echo -e "${GREEN}✅ LightRAG server is running${NC}"
else
    echo -e "${YELLOW}⏳ LightRAG server is starting (may take a moment)${NC}"
fi

# Check n8n
if curl -f http://localhost:5678 >/dev/null 2>&1; then
    echo -e "${GREEN}✅ n8n server is running${NC}"
else
    echo -e "${YELLOW}⏳ n8n server is starting (may take a moment)${NC}"
fi

echo

# Step 6: Show access information
echo -e "${BLUE}🎉 Quick Start Complete!${NC}"
echo "======================="
echo
echo -e "${GREEN}Access your services:${NC}"
echo "🔗 LightRAG API:     http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🔄 n8n Workflows:    http://localhost:5678"
echo "🗄️  Database (Adminer): http://localhost:8080"
echo
echo -e "${BLUE}Next steps:${NC}"
echo "1. 📝 Configure API keys in .env file:"
echo "   - GEMINI_API_KEY"
echo "   - OPENAI_API_KEY"
echo "   - SUPABASE_URL and keys"
echo
echo "2. 🔄 Import n8n workflows:"
echo "   - Open http://localhost:5678"
echo "   - Import workflows from n8n/workflows/"
echo
echo "3. 🧪 Test with a sample query:"
echo "   curl -X POST http://localhost:8000/query \\"
echo "        -H 'Content-Type: application/json' \\"
echo "        -H 'Authorization: Bearer \$LIGHTRAG_API_KEY' \\"
echo "        -d '{\"query\": \"Hello, GraphRAG!\"}'"
echo
echo "4. 📖 Read the documentation:"
echo "   - README.md for detailed setup"
echo "   - docs/guides/ for advanced configuration"
echo
echo -e "${BLUE}Useful commands:${NC}"
echo "🛑 Stop services:    docker-compose -f deployment/docker/docker-compose.dev.yml down"
echo "📊 View logs:        docker-compose -f deployment/docker/docker-compose.dev.yml logs -f"
echo "🔄 Restart services: docker-compose -f deployment/docker/docker-compose.dev.yml restart"
echo "🧹 Full cleanup:     ./scripts/setup/setup.sh --cleanup"
echo
echo -e "${GREEN}Happy GraphRAG-ing! 🚀${NC}"
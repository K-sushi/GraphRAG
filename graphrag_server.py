#!/usr/bin/env python3
"""
Microsoft GraphRAG FastAPI Server with Gemini Integration
SuperClaude Wave Orchestration Implementation - Phase 2

Replaces broken LightRAG server with proper Microsoft GraphRAG implementation
"""

import os
import sys
import asyncio
import logging
import yaml
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# GraphRAG imports
try:
    from graphrag.index import create_pipeline_config
    from graphrag.index.run import run_pipeline_with_config  
    from graphrag.query.structured_search.local_search.search import LocalSearch
    from graphrag.query.structured_search.global_search.search import GlobalSearch
    from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
    from graphrag.vector_stores import VectorStoreFactory, VectorStoreType
    from graphrag.index.graph.extractors.community_reports import CommunityReportsExtractor
except ImportError as e:
    print(f"GraphRAG not installed or incompatible version. Please install: pip install graphrag==0.3.0")
    print(f"Error: {e}")
    sys.exit(1)

# Local imports
from gemini_llm_provider import GeminiLLMProvider, create_gemini_llm, GraphRAGGeminiAdapter

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
graphrag_instance: Optional[Any] = None
local_search_engine: Optional[LocalSearch] = None
global_search_engine: Optional[GlobalSearch] = None
gemini_provider: Optional[GeminiLLMProvider] = None
config: Optional[Dict[str, Any]] = None

# Security
security = HTTPBearer(auto_error=False)

# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Query text")
    mode: str = Field(default="local", description="Search mode: local, global, or hybrid")
    top_k: int = Field(default=10, description="Number of top results to return")
    include_context: bool = Field(default=True, description="Include context information")
    conversation_history: Optional[List[Dict[str, str]]] = Field(default=None, description="Previous conversation")

class QueryResponse(BaseModel):
    response: str = Field(..., description="Generated response")
    entities: List[Dict[str, Any]] = Field(default=[], description="Extracted entities")
    relationships: List[Dict[str, Any]] = Field(default=[], description="Found relationships")
    communities: List[Dict[str, Any]] = Field(default=[], description="Relevant communities")
    sources: List[Dict[str, Any]] = Field(default=[], description="Source documents")
    metadata: Dict[str, Any] = Field(default={}, description="Response metadata")

class InsertRequest(BaseModel):
    content: str = Field(..., description="Document content to insert")
    document_id: Optional[str] = Field(default=None, description="Document identifier")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Document metadata")
    chunk_size: Optional[int] = Field(default=None, description="Custom chunk size")

class InsertResponse(BaseModel):
    success: bool = Field(..., description="Operation success status")
    document_id: str = Field(..., description="Assigned document ID")
    message: str = Field(..., description="Status message")
    entities_extracted: int = Field(default=0, description="Number of entities extracted")
    relationships_extracted: int = Field(default=0, description="Number of relationships extracted")
    communities_created: int = Field(default=0, description="Number of communities created")
    processing_time: float = Field(default=0.0, description="Processing time in seconds")

class IndexingStatus(BaseModel):
    status: str = Field(..., description="Indexing status")
    progress: float = Field(..., description="Progress percentage")
    current_step: str = Field(..., description="Current processing step")
    total_documents: int = Field(default=0, description="Total documents to process")
    processed_documents: int = Field(default=0, description="Documents processed")
    estimated_time_remaining: Optional[float] = Field(default=None, description="Estimated time remaining")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global graphrag_instance, local_search_engine, global_search_engine, gemini_provider, config
    
    # Startup
    logger.info("Starting Microsoft GraphRAG server with Gemini integration...")
    
    try:
        # Load configuration
        config = load_graphrag_config()
        
        # Initialize Gemini LLM provider
        gemini_provider = create_gemini_llm({
            "api_key": os.getenv("GEMINI_API_KEY"),
            "model": os.getenv("GRAPHRAG_LLM_MODEL", "gemini-1.5-pro-002"),
            "requests_per_minute": int(os.getenv("GRAPHRAG_REQUESTS_PER_MINUTE", "10000")),
            "tokens_per_minute": int(os.getenv("GRAPHRAG_TOKENS_PER_MINUTE", "150000")),
            "concurrent_requests": int(os.getenv("GRAPHRAG_CONCURRENT_REQUESTS", "25")),
        })
        
        # Initialize GraphRAG components
        await initialize_graphrag(config)
        
        # Create directories
        create_directories()
        
        logger.info("Microsoft GraphRAG server started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start GraphRAG server: {e}")
        raise
    
    # Shutdown
    logger.info("Shutting down GraphRAG server...")
    logger.info("GraphRAG server shutdown complete")

def load_graphrag_config() -> Dict[str, Any]:
    """Load GraphRAG configuration"""
    config_path = os.getenv("GRAPHRAG_CONFIG_PATH", "./graphrag_config.yaml")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Environment variable substitution
        config = substitute_env_vars(config)
        
        logger.info(f"Loaded GraphRAG configuration from {config_path}")
        return config
        
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

def substitute_env_vars(config: Any) -> Any:
    """Recursively substitute environment variables in config"""
    if isinstance(config, dict):
        return {k: substitute_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [substitute_env_vars(item) for item in config]
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        env_var = config[2:-1]
        return os.getenv(env_var, config)
    else:
        return config

def create_directories():
    """Create necessary directories"""
    directories = [
        os.getenv("GRAPHRAG_ROOT_DIR", "./graphrag_workspace"),
        os.getenv("GRAPHRAG_DATA_DIR", "./input"),
        os.getenv("GRAPHRAG_OUTPUT_DIR", "./output"),
        os.getenv("GRAPHRAG_CACHE_DIR", "./cache"),
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

async def initialize_graphrag(config: Dict[str, Any]):
    """Initialize GraphRAG components"""
    global local_search_engine, global_search_engine
    
    try:
        # Import our GraphRAG implementations
        from graphrag_search import create_search_engines
        
        # Create search engines
        data_dir = os.getenv("GRAPHRAG_OUTPUT_DIR", "./output")
        local_search_engine, global_search_engine, hybrid_search_engine = create_search_engines(
            data_dir, gemini_provider
        )
        
        # Store hybrid search engine globally for use in endpoints
        globals()['hybrid_search_engine'] = hybrid_search_engine
        
        logger.info("GraphRAG search engines initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize GraphRAG: {e}")
        # Don't raise - we can still run basic operations
        logger.warning("Running in basic mode without full GraphRAG functionality")

# Create FastAPI app
app = FastAPI(
    title="Microsoft GraphRAG Server with Gemini",
    description="High-performance Graph-based Retrieval-Augmented Generation using Microsoft GraphRAG and Google Gemini",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5678").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Authentication dependency
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key authentication"""
    if not credentials:
        raise HTTPException(status_code=401, detail="API key required")
    
    valid_api_keys = os.getenv("GRAPHRAG_API_KEY", "").split(",")
    if credentials.credentials not in valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return credentials.credentials

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "service": "Microsoft GraphRAG with Gemini",
        "components": {
            "gemini_provider": gemini_provider is not None,
            "graphrag_config": config is not None,
            "local_search": local_search_engine is not None,
            "global_search": global_search_engine is not None,
        }
    }

# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_knowledge_graph(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Query Microsoft GraphRAG knowledge graph"""
    if not gemini_provider:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")
    
    try:
        start_time = datetime.utcnow()
        
        # Use the appropriate search engine based on mode
        search_result = None
        
        if 'hybrid_search_engine' in globals():
            # Use full GraphRAG search engines
            hybrid_engine = globals()['hybrid_search_engine']
            
            if request.mode == "local":
                search_result = await hybrid_engine.local_search.search(
                    request.query,
                    conversation_history=request.conversation_history
                )
            elif request.mode == "global":
                search_result = await hybrid_engine.global_search.search(
                    request.query,
                    conversation_history=request.conversation_history
                )
            else:  # hybrid or auto
                search_result = await hybrid_engine.search(
                    request.query,
                    mode=request.mode,
                    conversation_history=request.conversation_history
                )
        
        if search_result:
            # Use GraphRAG search results
            response_text = search_result.get("response", "")
            entities = search_result.get("entities", [])
            relationships = search_result.get("relationships", [])
            communities_data = search_result.get("communities", [])
            sources = search_result.get("text_units", [])
            
            # Format communities for response
            communities = []
            if isinstance(communities_data, list):
                for community_id in communities_data:
                    if isinstance(community_id, str):
                        communities.append({"id": community_id, "name": f"Community {community_id}"})
                    elif isinstance(community_id, dict):
                        communities.append(community_id)
        
        else:
            # Fallback to basic Gemini query
            logger.warning("Using fallback Gemini query - GraphRAG search engines not available")
            
            query_context = f"""
            You are a helpful assistant that answers questions based on a knowledge graph.
            
            User Query: {request.query}
            
            Please provide a comprehensive answer based on the available information.
            """
            
            if request.conversation_history:
                history_text = "\n".join([
                    f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                    for msg in request.conversation_history[-5:]  # Last 5 turns
                ])
                query_context += f"\n\nConversation History:\n{history_text}"
            
            # Generate response using Gemini
            response_text = await gemini_provider.generate([
                {"role": "user", "content": query_context}
            ])
            
            # Empty results for fallback
            entities = []
            relationships = []
            communities = []
            sources = []
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Log metrics
        background_tasks.add_task(log_query_metrics, request.query, processing_time)
        
        return QueryResponse(
            response=response_text,
            entities=entities,
            relationships=relationships,
            communities=communities,
            sources=sources,
            metadata={
                "mode": request.mode,
                "top_k": request.top_k,
                "processing_time": processing_time,
                "model_used": gemini_provider.config.model,
                "timestamp": datetime.utcnow().isoformat(),
                "search_strategy": search_result.get("search_strategy", "fallback") if search_result else "fallback",
                "context_stats": search_result.get("context_stats", {}) if search_result else {},
            }
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

# Insert document endpoint
@app.post("/insert", response_model=InsertResponse)
async def insert_document(
    request: InsertRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Insert document into GraphRAG knowledge graph"""
    if not gemini_provider:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")
    
    try:
        start_time = datetime.utcnow()
        
        # Generate document ID if not provided
        document_id = request.document_id or f"doc_{int(start_time.timestamp())}"
        
        # Save document to input directory
        input_dir = Path(os.getenv("GRAPHRAG_DATA_DIR", "./input"))
        doc_path = input_dir / f"{document_id}.txt"
        
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(request.content)
        
        # TODO: Trigger GraphRAG indexing pipeline
        # For now, we'll return placeholder results
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Log metrics
        background_tasks.add_task(log_insert_metrics, document_id, len(request.content), processing_time)
        
        return InsertResponse(
            success=True,
            document_id=document_id,
            message="Document saved for indexing",
            entities_extracted=0,  # Placeholder
            relationships_extracted=0,  # Placeholder
            communities_created=0,  # Placeholder
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Document insertion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Insertion failed: {str(e)}")

# Upload file endpoint
@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """Upload file for processing"""
    try:
        input_dir = Path(os.getenv("GRAPHRAG_DATA_DIR", "./input"))
        file_path = input_dir / file.filename
        
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        return {
            "success": True,
            "filename": file.filename,
            "size": len(content),
            "message": "File uploaded successfully"
        }
        
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# Indexing status endpoint
@app.get("/indexing/status", response_model=IndexingStatus)
async def get_indexing_status(api_key: str = Depends(verify_api_key)):
    """Get indexing pipeline status"""
    # Placeholder implementation
    return IndexingStatus(
        status="ready",
        progress=100.0,
        current_step="ready",
        total_documents=0,
        processed_documents=0,
        estimated_time_remaining=None
    )

# Trigger indexing endpoint
@app.post("/indexing/start")
async def start_indexing(
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Start GraphRAG indexing pipeline"""
    try:
        # TODO: Implement actual GraphRAG indexing
        background_tasks.add_task(run_indexing_pipeline)
        
        return {
            "success": True,
            "message": "Indexing pipeline started",
            "status": "started"
        }
        
    except Exception as e:
        logger.error(f"Failed to start indexing: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing start failed: {str(e)}")

# System status endpoint
@app.get("/status")
async def get_system_status(api_key: str = Depends(verify_api_key)):
    """Get system status and statistics"""
    try:
        return {
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "Microsoft GraphRAG with Gemini",
            "version": "1.0.0",
            "configuration": {
                "llm_model": os.getenv("GRAPHRAG_LLM_MODEL", "gemini-1.5-pro-002"),
                "embedding_model": os.getenv("GRAPHRAG_EMBEDDING_MODEL", "text-embedding-3-small"),
                "data_directory": os.getenv("GRAPHRAG_DATA_DIR", "./input"),
                "output_directory": os.getenv("GRAPHRAG_OUTPUT_DIR", "./output"),
            },
            "components": {
                "gemini_provider": gemini_provider is not None,
                "local_search": local_search_engine is not None,
                "global_search": global_search_engine is not None,
            }
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

# Background task functions
async def log_query_metrics(query: str, processing_time: float):
    """Log query metrics"""
    logger.info(f"Query processed - Time: {processing_time:.2f}s, Length: {len(query)} chars")

async def log_insert_metrics(doc_id: str, content_length: int, processing_time: float):
    """Log insertion metrics"""
    logger.info(f"Document inserted - ID: {doc_id}, Size: {content_length} chars, Time: {processing_time:.2f}s")

async def run_indexing_pipeline():
    """Run GraphRAG indexing pipeline in background"""
    try:
        logger.info("Starting GraphRAG indexing pipeline...")
        
        # Import and use our GraphRAG pipeline
        from graphrag_pipeline import create_graphrag_pipeline
        
        # Create pipeline with current configuration
        pipeline = create_graphrag_pipeline({
            "input_dir": os.getenv("GRAPHRAG_DATA_DIR", "./input"),
            "output_dir": os.getenv("GRAPHRAG_OUTPUT_DIR", "./output"),
            "cache_dir": os.getenv("GRAPHRAG_CACHE_DIR", "./cache"),
            "chunk_size": int(os.getenv("GRAPHRAG_CHUNK_SIZE", "1200")),
            "chunk_overlap": int(os.getenv("GRAPHRAG_CHUNK_OVERLAP", "100")),
        })
        
        # Run the pipeline
        result = await pipeline.run_complete_pipeline()
        
        logger.info(f"GraphRAG indexing pipeline completed successfully: {result}")
        
        # Reinitialize search engines with new data
        await initialize_graphrag(config or {})
        
    except Exception as e:
        logger.error(f"Indexing pipeline failed: {e}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url)
        }
    )

# Development endpoints
if os.getenv("DEBUG", "false").lower() == "true":
    @app.get("/debug/config")
    async def debug_config():
        """Debug endpoint to view configuration"""
        return config if config else {"error": "Configuration not loaded"}
    
    @app.get("/debug/gemini")
    async def debug_gemini():
        """Debug endpoint to test Gemini connection"""
        if not gemini_provider:
            return {"error": "Gemini provider not initialized"}
        
        try:
            test_response = await gemini_provider.generate([
                {"role": "user", "content": "Say 'GraphRAG with Gemini is working!'"}
            ])
            return {"success": True, "response": test_response}
        except Exception as e:
            return {"error": str(e)}

def main():
    """Main entry point"""
    host = os.getenv("GRAPHRAG_HOST", "0.0.0.0")
    port = int(os.getenv("GRAPHRAG_PORT", "8000"))
    workers = int(os.getenv("GRAPHRAG_WORKERS", "1"))
    
    if os.getenv("DEBUG", "false").lower() == "true":
        # Development mode
        uvicorn.run(
            "graphrag_server:app",
            host=host,
            port=port,
            reload=True,
            log_level="debug"
        )
    else:
        # Production mode
        uvicorn.run(
            app,
            host=host,
            port=port,
            workers=workers,
            log_level="info"
        )

if __name__ == "__main__":
    main()
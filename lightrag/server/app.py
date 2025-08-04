#!/usr/bin/env python3
"""
LightRAG Server Application
GraphRAG Implementation with FastAPI
"""

import os
import sys
import asyncio
import logging
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# LightRAG imports
try:
    from lightrag import LightRAG
    from lightrag.llm import gpt_4o_mini_complete
    from lightrag.embed import openai_embed
except ImportError:
    print("LightRAG not installed. Please install it with: pip install lightrag")
    sys.exit(1)

# Configuration
from config import Config
from models import QueryRequest, QueryResponse, InsertRequest, InsertResponse
from utils import TokenTracker, HealthChecker, MetricsCollector

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
lightrag_instance: Optional[LightRAG] = None
config: Optional[Config] = None
token_tracker: Optional[TokenTracker] = None
health_checker: Optional[HealthChecker] = None
metrics_collector: Optional[MetricsCollector] = None

# Security
security = HTTPBearer(auto_error=False)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global lightrag_instance, config, token_tracker, health_checker, metrics_collector
    
    # Startup
    logger.info("Starting LightRAG server...")
    
    try:
        # Load configuration
        config = Config()
        
        # Initialize utilities
        token_tracker = TokenTracker()
        health_checker = HealthChecker()
        metrics_collector = MetricsCollector()
        
        # Initialize LightRAG
        lightrag_instance = await initialize_lightrag(config)
        
        # Start background tasks
        asyncio.create_task(health_checker.start_monitoring())
        asyncio.create_task(metrics_collector.start_collection())
        
        logger.info("LightRAG server started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start LightRAG server: {e}")
        raise
    
    # Shutdown
    logger.info("Shutting down LightRAG server...")
    
    if health_checker:
        await health_checker.stop_monitoring()
    
    if metrics_collector:
        await metrics_collector.stop_collection()
    
    logger.info("LightRAG server shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="LightRAG GraphRAG Server",
    description="High-performance Retrieval-Augmented Generation with Knowledge Graphs",
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
    
    valid_api_keys = os.getenv("LIGHTRAG_API_KEY", "").split(",")
    if credentials.credentials not in valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return credentials.credentials

async def initialize_lightrag(config: Config) -> LightRAG:
    """Initialize LightRAG instance"""
    try:
        # Configure LLM function based on model selection
        if config.lightrag.llm.model_name.startswith("gemini"):
            from llm_integrations.gemini import gemini_complete
            llm_func = gemini_complete
        elif config.lightrag.llm.model_name.startswith("gpt"):
            llm_func = gpt_4o_mini_complete
        else:
            raise ValueError(f"Unsupported model: {config.lightrag.llm.model_name}")
        
        # Configure embedding function
        if config.lightrag.embedding.model_name.startswith("text-embedding"):
            embed_func = openai_embed
        else:
            from embed_integrations.custom import custom_embed
            embed_func = custom_embed
        
        # Create LightRAG instance
        rag = LightRAG(
            working_dir=config.lightrag.system.working_dir,
            workspace=config.lightrag.system.workspace,
            llm_model_func=llm_func,
            llm_model_name=config.lightrag.llm.model_name,
            llm_model_max_async=config.lightrag.llm.llm_model_max_async,
            embedding_func=embed_func,
            embedding_batch_num=config.lightrag.embedding.embedding_batch_num,
            embedding_func_max_async=config.lightrag.embedding.embedding_func_max_async,
            chunk_token_size=config.lightrag.document_processing.chunk_token_size,
            chunk_overlap_token_size=config.lightrag.document_processing.chunk_overlap_token_size,
            enable_llm_cache=config.lightrag.system.enable_llm_cache,
            enable_llm_cache_for_entity_extract=config.lightrag.system.enable_llm_cache_for_entity_extract,
        )
        
        logger.info(f"LightRAG initialized with model: {config.lightrag.llm.model_name}")
        return rag
        
    except Exception as e:
        logger.error(f"Failed to initialize LightRAG: {e}")
        raise

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not lightrag_instance:
        raise HTTPException(status_code=503, detail="LightRAG not initialized")
    
    health_status = await health_checker.get_health_status()
    
    return {
        "status": "healthy" if health_status["healthy"] else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "components": health_status["components"]
    }

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics endpoint"""
    if not metrics_collector:
        raise HTTPException(status_code=503, detail="Metrics collector not initialized")
    
    return await metrics_collector.get_prometheus_metrics()

# Insert document endpoint
@app.post("/insert", response_model=InsertResponse)
async def insert_document(
    request: InsertRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Insert document into LightRAG knowledge graph"""
    if not lightrag_instance:
        raise HTTPException(status_code=503, detail="LightRAG not initialized")
    
    try:
        start_time = datetime.utcnow()
        
        # Track token usage
        input_tokens = token_tracker.estimate_tokens(request.content)
        
        # Insert document
        result = await lightrag_instance.ainsert(request.content)
        
        # Log metrics
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        background_tasks.add_task(
            metrics_collector.record_insertion,
            processing_time,
            input_tokens,
            len(request.content),
            request.metadata
        )
        
        return InsertResponse(
            success=True,
            message="Document inserted successfully",
            document_id=result.get("document_id"),
            entities_extracted=result.get("entities_count", 0),
            relationships_extracted=result.get("relationships_count", 0),
            processing_time_seconds=processing_time,
            tokens_used=input_tokens
        )
        
    except Exception as e:
        logger.error(f"Document insertion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Insertion failed: {str(e)}")

# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_knowledge_graph(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Query LightRAG knowledge graph"""
    if not lightrag_instance:
        raise HTTPException(status_code=503, detail="LightRAG not initialized")
    
    try:
        start_time = datetime.utcnow()
        
        # Track token usage
        input_tokens = token_tracker.estimate_tokens(request.query)
        
        # Execute query
        result = await lightrag_instance.aquery(
            query=request.query,
            mode=request.mode,
            top_k=request.top_k,
            enable_rerank=request.enable_rerank,
            conversation_history=request.conversation_history
        )
        
        # Estimate output tokens
        output_tokens = token_tracker.estimate_tokens(result.get("response", ""))
        
        # Log metrics
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        background_tasks.add_task(
            metrics_collector.record_query,
            processing_time,
            input_tokens,
            output_tokens,
            request.mode,
            request.top_k
        )
        
        return QueryResponse(
            response=result.get("response", ""),
            entities=result.get("entities", []),
            relationships=result.get("relationships", []),
            contexts=result.get("contexts", []),
            metadata={
                "mode": request.mode,
                "top_k": request.top_k,
                "processing_time_seconds": processing_time,
                "tokens_used": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens
                },
                "entities_found": len(result.get("entities", [])),
                "relationships_found": len(result.get("relationships", [])),
                "context_chunks": len(result.get("contexts", []))
            }
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

# Delete document endpoint
@app.delete("/delete/{document_id}")
async def delete_document(
    document_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Delete document from knowledge graph"""
    if not lightrag_instance:
        raise HTTPException(status_code=503, detail="LightRAG not initialized")
    
    try:
        # Note: LightRAG may not support deletion - implement based on your needs
        result = await lightrag_instance.adelete(document_id)
        
        return {"success": True, "message": f"Document {document_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Document deletion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

# Status endpoint
@app.get("/status")
async def get_status(api_key: str = Depends(verify_api_key)):
    """Get system status and statistics"""
    if not lightrag_instance:
        raise HTTPException(status_code=503, detail="LightRAG not initialized")
    
    try:
        stats = await metrics_collector.get_system_stats()
        
        return {
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "statistics": stats,
            "configuration": {
                "model": config.lightrag.llm.model_name,
                "embedding_model": config.lightrag.embedding.model_name,
                "workspace": config.lightrag.system.workspace,
                "cache_enabled": config.lightrag.system.enable_llm_cache
            }
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

# Clear cache endpoint
@app.post("/admin/clear-cache")
async def clear_cache(
    cache_type: str = "all",
    api_key: str = Depends(verify_api_key)
):
    """Clear LightRAG caches"""
    if not lightrag_instance:
        raise HTTPException(status_code=503, detail="LightRAG not initialized")
    
    try:
        # Clear specified cache
        if cache_type == "all":
            result = await lightrag_instance.aclear_cache()
        elif cache_type == "llm":
            result = await lightrag_instance.aclear_llm_cache()
        elif cache_type == "embedding":
            result = await lightrag_instance.aclear_embedding_cache()
        else:
            raise HTTPException(status_code=400, detail="Invalid cache type")
        
        return {
            "success": True,
            "message": f"Cache '{cache_type}' cleared successfully",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Cache clearing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clearing failed: {str(e)}")

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

# Development endpoints (only in debug mode)
if os.getenv("DEBUG", "false").lower() == "true":
    @app.get("/debug/config")
    async def debug_config():
        """Debug endpoint to view configuration"""
        return config.dict() if config else {"error": "Configuration not loaded"}
    
    @app.get("/debug/memory")
    async def debug_memory():
        """Debug endpoint to view memory usage"""
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "memory_usage_mb": memory_info.rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "threads": process.num_threads()
        }

def main():
    """Main entry point"""
    host = os.getenv("LIGHTRAG_HOST", "0.0.0.0")
    port = int(os.getenv("LIGHTRAG_PORT", "8000"))
    workers = int(os.getenv("LIGHTRAG_WORKERS", "1"))
    
    if os.getenv("DEBUG", "false").lower() == "true":
        # Development mode
        uvicorn.run(
            "app:app",
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
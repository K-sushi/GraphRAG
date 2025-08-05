#!/usr/bin/env python3
"""
Enhanced Microsoft GraphRAG FastAPI Server with Real-time Features
SuperClaude Wave Orchestration - Phase 2 Enhancement

Features:
- Real-time indexing with WebSocket notifications
- Background task processing
- File monitoring and hot reload
- Performance metrics and monitoring
- Comprehensive API endpoints
"""

import os
import sys
import asyncio
import logging
import yaml
import json
import time
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, File, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# Import our enhanced components
try:
    from realtime_indexing import RealtimeIndexingManager, IndexingTask, IndexingStats
    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False
    logging.warning("Real-time indexing not available")

# Import existing components
from gemini_llm_provider import GeminiLLMProvider, create_gemini_llm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
gemini_provider: Optional[GeminiLLMProvider] = None
realtime_manager: Optional[RealtimeIndexingManager] = None
config: Optional[Dict[str, Any]] = None
server_stats = {
    "start_time": datetime.utcnow(),
    "requests_processed": 0,
    "errors_encountered": 0,
    "active_connections": 0,
}

# Security
security = HTTPBearer(auto_error=False)

# Enhanced Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Query text")
    mode: str = Field(default="hybrid", description="Search mode: local, global, or hybrid")
    top_k: int = Field(default=10, description="Number of top results to return")
    include_context: bool = Field(default=True, description="Include context information")
    conversation_history: Optional[List[Dict[str, str]]] = Field(default=None, description="Previous conversation")
    stream: bool = Field(default=False, description="Stream response")

class QueryResponse(BaseModel):
    response: str = Field(..., description="Generated response")
    entities: List[Dict[str, Any]] = Field(default=[], description="Extracted entities")
    relationships: List[Dict[str, Any]] = Field(default=[], description="Found relationships")
    communities: List[Dict[str, Any]] = Field(default=[], description="Relevant communities")
    sources: List[Dict[str, Any]] = Field(default=[], description="Source documents")
    metadata: Dict[str, Any] = Field(default={}, description="Response metadata")
    search_strategy: Optional[str] = Field(default=None, description="Search strategy used")

class InsertRequest(BaseModel):
    content: str = Field(..., description="Document content to insert")
    document_id: Optional[str] = Field(default=None, description="Document identifier")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Document metadata")
    trigger_indexing: bool = Field(default=True, description="Trigger immediate indexing")
    priority: int = Field(default=5, description="Indexing priority (1-10)")

class InsertResponse(BaseModel):
    success: bool = Field(..., description="Operation success status")
    document_id: str = Field(..., description="Assigned document ID")
    message: str = Field(..., description="Status message")
    task_id: Optional[str] = Field(default=None, description="Background task ID")
    estimated_completion: Optional[str] = Field(default=None, description="Estimated completion time")

class IndexingStatusResponse(BaseModel):
    status: str = Field(..., description="System status")
    active_tasks: int = Field(..., description="Number of active indexing tasks")
    completed_tasks: int = Field(..., description="Number of completed tasks")
    failed_tasks: int = Field(..., description="Number of failed tasks")
    average_processing_time: float = Field(..., description="Average task processing time")
    last_indexing_time: Optional[str] = Field(default=None, description="Last indexing timestamp")
    system_stats: Dict[str, Any] = Field(default={}, description="System statistics")

class SystemStatusResponse(BaseModel):
    status: str = Field(..., description="System status")
    uptime: float = Field(..., description="System uptime in seconds")
    version: str = Field(..., description="Server version")
    components: Dict[str, bool] = Field(..., description="Component health status")
    performance: Dict[str, Any] = Field(..., description="Performance metrics")
    configuration: Dict[str, Any] = Field(..., description="System configuration")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application lifespan manager"""
    global gemini_provider, realtime_manager, config
    
    # Startup
    logger.info("Starting Enhanced GraphRAG Server with Real-time Features...")
    
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
        
        # Initialize real-time indexing if available
        if REALTIME_AVAILABLE:
            realtime_manager = RealtimeIndexingManager(config_path="./graphrag_config.yaml")
            await realtime_manager.start()
            logger.info("Real-time indexing system started successfully")
        
        # Create directories
        create_directories()
        
        # Initialize GraphRAG components
        await initialize_graphrag(config)
        
        logger.info("Enhanced GraphRAG Server started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start Enhanced GraphRAG Server: {e}")
        raise
    
    # Shutdown
    logger.info("Shutting down Enhanced GraphRAG Server...")
    
    if realtime_manager:
        await realtime_manager.stop()
        logger.info("Real-time indexing system stopped")
        
    logger.info("Enhanced GraphRAG Server shutdown complete")

def load_graphrag_config() -> Dict[str, Any]:
    """Load GraphRAG configuration with defaults"""
    config_path = os.getenv("GRAPHRAG_CONFIG_PATH", "./graphrag_config.yaml")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Environment variable substitution
        config = substitute_env_vars(config)
        
        logger.info(f"Loaded GraphRAG configuration from {config_path}")
        return config
        
    except FileNotFoundError:
        logger.warning(f"Configuration file not found: {config_path}, using defaults")
        return get_default_config()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}, using defaults")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        "paths": {
            "input": "./input",
            "output": "./output",
            "cache": "./cache"
        },
        "indexing": {
            "max_workers": 4,
            "batch_size": 10,
            "chunk_size": 1200,
            "chunk_overlap": 100
        },
        "websocket": {
            "host": "localhost",
            "port": 8765
        },
        "performance": {
            "enable_metrics": True,
            "metrics_interval": 10
        }
    }

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
        "./uploads",  # For file uploads
        "./logs",     # For log files
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

async def initialize_graphrag(config: Dict[str, Any]):
    """Initialize GraphRAG components with fallback"""
    try:
        # Try to import and initialize search engines
        from graphrag_search import create_search_engines
        
        data_dir = os.getenv("GRAPHRAG_OUTPUT_DIR", "./output")
        local_search_engine, global_search_engine, hybrid_search_engine = create_search_engines(
            data_dir, gemini_provider
        )
        
        # Store search engines globally
        globals()['local_search_engine'] = local_search_engine
        globals()['global_search_engine'] = global_search_engine
        globals()['hybrid_search_engine'] = hybrid_search_engine
        
        logger.info("GraphRAG search engines initialized successfully")
        
    except Exception as e:
        logger.warning(f"Failed to initialize full GraphRAG functionality: {e}")
        logger.info("Running in basic mode with Gemini-only functionality")

# Create Enhanced FastAPI app
app = FastAPI(
    title="Enhanced Microsoft GraphRAG Server",
    description="High-performance Graph-based RAG with Real-time Indexing, Monitoring, and Advanced Features",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5678,http://localhost:8080").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Authentication dependency
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Enhanced API key verification"""
    if not credentials:
        # Allow access in development mode
        if os.getenv("DEBUG", "false").lower() == "true":
            return "development"
        raise HTTPException(status_code=401, detail="API key required")
    
    valid_api_keys = os.getenv("GRAPHRAG_API_KEY", "").split(",")
    if credentials.credentials not in valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return credentials.credentials

# Request tracking middleware
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track request metrics"""
    start_time = time.time()
    server_stats["active_connections"] += 1
    
    try:
        response = await call_next(request)
        server_stats["requests_processed"] += 1
        return response
    except Exception as e:
        server_stats["errors_encountered"] += 1
        raise
    finally:
        server_stats["active_connections"] -= 1
        processing_time = time.time() - start_time
        
        # Log slow requests
        if processing_time > 5.0:
            logger.warning(f"Slow request: {request.url} took {processing_time:.2f}s")

# Enhanced endpoints

@app.get("/health")
async def enhanced_health_check():
    """Enhanced health check endpoint"""
    uptime = (datetime.utcnow() - server_stats["start_time"]).total_seconds()
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "service": "Enhanced Microsoft GraphRAG with Real-time Features",
        "uptime": uptime,
        "components": {
            "gemini_provider": gemini_provider is not None,
            "realtime_indexing": realtime_manager is not None and realtime_manager.running,
            "websocket_server": realtime_manager is not None and len(realtime_manager.websocket_server.clients) >= 0,
            "graphrag_config": config is not None,
            "search_engines": 'hybrid_search_engine' in globals(),
        },
        "performance": {
            "requests_processed": server_stats["requests_processed"],
            "errors_encountered": server_stats["errors_encountered"],
            "active_connections": server_stats["active_connections"],
            "error_rate": server_stats["errors_encountered"] / max(server_stats["requests_processed"], 1) * 100,
        }
    }

@app.post("/query", response_model=QueryResponse)
async def enhanced_query_endpoint(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Enhanced query endpoint with streaming and advanced features"""
    if not gemini_provider:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")
    
    start_time = time.time()
    
    try:
        # Enhanced query processing logic
        search_result = None
        search_strategy = "fallback"
        
        # Try full GraphRAG search engines first
        if 'hybrid_search_engine' in globals():
            hybrid_engine = globals()['hybrid_search_engine']
            
            if request.mode == "local":
                search_result = await hybrid_engine.local_search.search(
                    request.query,
                    conversation_history=request.conversation_history
                )
                search_strategy = "local_search"
            elif request.mode == "global":
                search_result = await hybrid_engine.global_search.search(
                    request.query,
                    conversation_history=request.conversation_history
                )
                search_strategy = "global_search"
            else:  # hybrid or auto
                search_result = await hybrid_engine.search(
                    request.query,
                    mode=request.mode,
                    conversation_history=request.conversation_history
                )
                search_strategy = "hybrid_search"
        
        # Process search results or fallback to basic Gemini
        if search_result:
            response_text = search_result.get("response", "")
            entities = search_result.get("entities", [])
            relationships = search_result.get("relationships", [])
            communities_data = search_result.get("communities", [])
            sources = search_result.get("text_units", [])
            
            # Format communities
            communities = []
            if isinstance(communities_data, list):
                for community_id in communities_data:
                    if isinstance(community_id, str):
                        communities.append({"id": community_id, "name": f"Community {community_id}"})
                    elif isinstance(community_id, dict):
                        communities.append(community_id)
        
        else:
            # Enhanced fallback to Gemini with context
            logger.info(f"Using enhanced Gemini fallback for query: {request.query[:50]}...")
            
            query_context = f"""
            You are an advanced AI assistant with access to a knowledge graph system.
            
            User Query: {request.query}
            
            Please provide a comprehensive, well-structured answer. If you need to reference 
            specific entities, relationships, or communities, explain your reasoning clearly.
            
            Search Mode: {request.mode}
            Context Requested: {request.include_context}
            """
            
            if request.conversation_history:
                history_text = "\n".join([
                    f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                    for msg in request.conversation_history[-5:]
                ])
                query_context += f"\n\nConversation History:\n{history_text}"
            
            response_text = await gemini_provider.generate([
                {"role": "user", "content": query_context}
            ])
            
            # Empty results for fallback
            entities = []
            relationships = []
            communities = []
            sources = []
            search_strategy = "gemini_fallback"
        
        processing_time = time.time() - start_time
        
        # Log query metrics
        background_tasks.add_task(
            log_query_metrics, 
            request.query, 
            processing_time, 
            search_strategy,
            len(entities),
            len(relationships)
        )
        
        # Create enhanced response
        response = QueryResponse(
            response=response_text,
            entities=entities[:request.top_k],
            relationships=relationships[:request.top_k],
            communities=communities,
            sources=sources[:10],
            search_strategy=search_strategy,
            metadata={
                "mode": request.mode,
                "top_k": request.top_k,
                "processing_time": processing_time,
                "model_used": gemini_provider.config.model,
                "timestamp": datetime.utcnow().isoformat(),
                "search_strategy": search_strategy,
                "context_stats": search_result.get("context_stats", {}) if search_result else {},
                "performance": {
                    "query_length": len(request.query),
                    "response_length": len(response_text),
                    "entities_found": len(entities),
                    "relationships_found": len(relationships),
                    "communities_found": len(communities),
                }
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Enhanced query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/insert", response_model=InsertResponse)
async def enhanced_insert_endpoint(
    request: InsertRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Enhanced document insertion with real-time indexing"""
    start_time = time.time()
    
    try:
        # Generate document ID if not provided
        document_id = request.document_id or f"doc_{int(start_time)}_{hash(request.content[:100]) % 10000}"
        
        # Save document to input directory
        input_dir = Path(os.getenv("GRAPHRAG_DATA_DIR", "./input"))
        doc_path = input_dir / f"{document_id}.txt"
        
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(request.content)
        
        # Trigger real-time indexing if available and requested
        task_id = None
        estimated_completion = None
        
        if request.trigger_indexing and realtime_manager:
            task = IndexingTask(
                task_id=f"doc_{document_id}_{int(time.time())}",
                task_type="document",
                file_path=str(doc_path),
                metadata={
                    "document_id": document_id,
                    "content_length": len(request.content),
                    "user_metadata": request.metadata,
                    "trigger": "api_insert"
                },
                priority=request.priority
            )
            
            realtime_manager.add_task(task)
            task_id = task.task_id
            
            # Estimate completion time (simple heuristic)
            base_time = len(request.content) / 1000 * 2  # 2 seconds per 1000 chars
            queue_delay = len(realtime_manager.active_tasks) * 0.5  # 0.5s per queued task
            estimated_completion = (datetime.utcnow() + 
                                  timedelta(seconds=base_time + queue_delay)).isoformat()
        
        processing_time = time.time() - start_time
        
        # Log insertion metrics
        background_tasks.add_task(
            log_insert_metrics, 
            document_id, 
            len(request.content), 
            processing_time,
            task_id is not None
        )
        
        return InsertResponse(
            success=True,
            document_id=document_id,
            message=f"Document saved and {'indexing triggered' if task_id else 'ready for indexing'}",
            task_id=task_id,
            estimated_completion=estimated_completion
        )
        
    except Exception as e:
        logger.error(f"Enhanced document insertion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Insertion failed: {str(e)}")

@app.get("/indexing/status", response_model=IndexingStatusResponse)
async def get_enhanced_indexing_status(api_key: str = Depends(verify_api_key)):
    """Get enhanced indexing system status"""
    if not realtime_manager:
        return IndexingStatusResponse(
            status="disabled",
            active_tasks=0,
            completed_tasks=0,
            failed_tasks=0,
            average_processing_time=0.0,
            system_stats={"realtime_indexing": False}
        )
    
    status = realtime_manager.get_status()
    stats = realtime_manager.stats
    
    return IndexingStatusResponse(
        status="active" if status["running"] else "stopped",
        active_tasks=status["active_tasks"],
        completed_tasks=stats.completed_tasks,
        failed_tasks=stats.failed_tasks,
        average_processing_time=stats.average_processing_time,
        last_indexing_time=stats.last_indexing_time.isoformat() if stats.last_indexing_time else None,
        system_stats={
            "file_monitoring": status["file_monitoring"],
            "websocket_clients": status["websocket_clients"],
            "workers": status["workers"],
            "documents_indexed": stats.documents_indexed,
            "entities_extracted": stats.entities_extracted,
            "relationships_extracted": stats.relationships_extracted,
            "communities_detected": stats.communities_detected,
        }
    )

@app.get("/status", response_model=SystemStatusResponse)
async def get_enhanced_system_status(api_key: str = Depends(verify_api_key)):
    """Get comprehensive system status"""
    uptime = (datetime.utcnow() - server_stats["start_time"]).total_seconds()
    
    components = {
        "gemini_provider": gemini_provider is not None,
        "realtime_indexing": realtime_manager is not None and realtime_manager.running,
        "file_monitoring": False,
        "websocket_server": False,
        "search_engines": 'hybrid_search_engine' in globals(),
    }
    
    if realtime_manager:
        status = realtime_manager.get_status()
        components.update({
            "file_monitoring": status["file_monitoring"],
            "websocket_server": status["websocket_clients"] >= 0,
        })
    
    performance = {
        "uptime": uptime,
        "requests_processed": server_stats["requests_processed"],
        "errors_encountered": server_stats["errors_encountered"],
        "active_connections": server_stats["active_connections"],
        "error_rate": server_stats["errors_encountered"] / max(server_stats["requests_processed"], 1) * 100,
        "requests_per_second": server_stats["requests_processed"] / max(uptime, 1),
    }
    
    if realtime_manager:
        stats = realtime_manager.stats
        performance.update({
            "indexing_tasks_completed": stats.completed_tasks,
            "indexing_tasks_failed": stats.failed_tasks,
            "average_indexing_time": stats.average_processing_time,
            "documents_indexed": stats.documents_indexed,
        })
    
    configuration = {
        "llm_model": os.getenv("GRAPHRAG_LLM_MODEL", "gemini-1.5-pro-002"),
        "data_directory": os.getenv("GRAPHRAG_DATA_DIR", "./input"),
        "output_directory": os.getenv("GRAPHRAG_OUTPUT_DIR", "./output"),
        "cache_directory": os.getenv("GRAPHRAG_CACHE_DIR", "./cache"),
        "debug_mode": os.getenv("DEBUG", "false").lower() == "true",
        "realtime_features": REALTIME_AVAILABLE,
    }
    
    return SystemStatusResponse(
        status="operational",
        uptime=uptime,
        version="2.0.0",
        components=components,
        performance=performance,
        configuration=configuration
    )

# WebSocket endpoint for real-time notifications
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time notifications"""
    await websocket.accept()
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connection",
            "message": "Connected to Enhanced GraphRAG Server",
            "features": {
                "realtime_indexing": REALTIME_AVAILABLE,
                "search_engines": 'hybrid_search_engine' in globals(),
                "performance_monitoring": True,
            },
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep connection alive and send periodic updates
        while True:
            try:
                # Wait for client message or timeout
                message = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                
                # Handle different message types
                if message.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                elif message.get("type") == "status_request":
                    # Send current status
                    uptime = (datetime.utcnow() - server_stats["start_time"]).total_seconds()
                    status = {
                        "type": "status_update",
                        "uptime": uptime,
                        "performance": {
                            "requests_processed": server_stats["requests_processed"],
                            "active_connections": server_stats["active_connections"],
                            "error_rate": server_stats["errors_encountered"] / max(server_stats["requests_processed"], 1) * 100,
                        },
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    if realtime_manager:
                        status["indexing"] = {
                            "active_tasks": len(realtime_manager.active_tasks),
                            "completed_tasks": realtime_manager.stats.completed_tasks,
                            "failed_tasks": realtime_manager.stats.failed_tasks,
                        }
                    
                    await websocket.send_json(status)
                
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
    except Exception as e:
        logger.info(f"WebSocket connection closed: {e}")

# Background task functions
async def log_query_metrics(query: str, processing_time: float, strategy: str, entities: int, relationships: int):
    """Enhanced query metrics logging"""
    logger.info(
        f"Query processed - Strategy: {strategy}, Time: {processing_time:.2f}s, "
        f"Length: {len(query)} chars, Entities: {entities}, Relationships: {relationships}"
    )

async def log_insert_metrics(doc_id: str, content_length: int, processing_time: float, indexed: bool):
    """Enhanced insertion metrics logging"""
    logger.info(
        f"Document inserted - ID: {doc_id}, Size: {content_length} chars, "
        f"Time: {processing_time:.2f}s, Indexed: {indexed}"
    )

# Enhanced development endpoints
if os.getenv("DEBUG", "false").lower() == "true":
    
    @app.get("/")
    async def root():
        """Development root endpoint with dashboard"""
        dashboard_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enhanced GraphRAG Server Dashboard</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; }
                .card { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .status { display: inline-block; padding: 4px 8px; border-radius: 4px; color: white; font-size: 12px; }
                .status.healthy { background: #4CAF50; }
                .status.warning { background: #FF9800; }
                .status.error { background: #F44336; }
                .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                h1 { color: #333; text-align: center; }
                h2 { color: #666; border-bottom: 2px solid #eee; padding-bottom: 10px; }
                .metric { display: flex; justify-content: space-between; margin: 10px 0; }
                .metric-value { font-weight: bold; color: #2196F3; }
                a { color: #2196F3; text-decoration: none; }
                a:hover { text-decoration: underline; }
                pre { background: #f8f8f8; padding: 15px; border-radius: 4px; overflow-x: auto; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 Enhanced GraphRAG Server Dashboard</h1>
                
                <div class="card">
                    <h2>🔗 Quick Links</h2>
                    <p><a href="/docs">📖 API Documentation (Swagger)</a></p>
                    <p><a href="/redoc">📚 API Documentation (ReDoc)</a></p>
                    <p><a href="/health">💚 Health Check</a></p>
                    <p><a href="/status">📊 System Status</a></p>
                    <p><a href="/indexing/status">⚡ Indexing Status</a></p>
                </div>
                
                <div class="grid">
                    <div class="card">
                        <h2>🎯 Features</h2>
                        <ul>
                            <li>✅ Microsoft GraphRAG Integration</li>
                            <li>✅ Google Gemini LLM Provider</li>
                            <li>✅ Local, Global & Hybrid Search</li>
                            <li>⚡ Real-time Indexing</li>
                            <li>📡 WebSocket Notifications</li>
                            <li>📊 Performance Monitoring</li>
                            <li>🔄 Background Task Processing</li>
                            <li>📁 File Monitoring & Hot Reload</li>
                        </ul>
                    </div>
                    
                    <div class="card">
                        <h2>🔧 Usage Examples</h2>
                        <h3>Query Knowledge Graph:</h3>
                        <pre>curl -X POST "http://localhost:8000/query" \\
  -H "Content-Type: application/json" \\
  -d '{"query": "What is artificial intelligence?"}'</pre>
                        
                        <h3>Insert Document:</h3>
                        <pre>curl -X POST "http://localhost:8000/insert" \\
  -H "Content-Type: application/json" \\
  -d '{"content": "AI is transforming the world..."}'</pre>
                    </div>
                </div>
                
                <div class="card">
                    <h2>📡 WebSocket Connection</h2>
                    <p>Connect to <code>ws://localhost:8000/ws</code> for real-time updates</p>
                    <pre>
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Real-time update:', data);
};
                    </pre>
                </div>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=dashboard_html)
    
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
                {"role": "user", "content": "Say 'Enhanced GraphRAG with real-time features is working!'"}
            ])
            return {"success": True, "response": test_response}
        except Exception as e:
            return {"error": str(e)}

def main():
    """Enhanced main entry point"""
    host = os.getenv("GRAPHRAG_HOST", "0.0.0.0")
    port = int(os.getenv("GRAPHRAG_PORT", "8000"))
    workers = int(os.getenv("GRAPHRAG_WORKERS", "1"))
    
    print("🚀 Starting Enhanced Microsoft GraphRAG Server")
    print(f"📍 Host: {host}:{port}")
    print(f"👥 Workers: {workers}")
    print(f"🐛 Debug: {os.getenv('DEBUG', 'false')}")
    print(f"⚡ Real-time: {REALTIME_AVAILABLE}")
    
    if os.getenv("DEBUG", "false").lower() == "true":
        # Development mode
        uvicorn.run(
            "enhanced_graphrag_server:app",
            host=host,
            port=port,
            reload=True,
            log_level="debug",
            access_log=True
        )
    else:
        # Production mode
        uvicorn.run(
            app,
            host=host,
            port=port,
            workers=workers,
            log_level="info",
            access_log=True
        )

if __name__ == "__main__":
    main()
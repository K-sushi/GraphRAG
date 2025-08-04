"""
Pydantic models for LightRAG API
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator

# Enums for query modes and other options
class QueryMode(str, Enum):
    """Query execution modes"""
    LOCAL = "local"
    GLOBAL = "global"
    HYBRID = "hybrid"
    NAIVE = "naive"
    MIX = "mix"
    BYPASS = "bypass"

class EntityType(str, Enum):
    """Entity types for filtering"""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"
    CONCEPT = "concept"
    PRODUCT = "product"
    OTHER = "other"

class RelationshipType(str, Enum):
    """Relationship types"""
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    LOCATED_IN = "located_in"
    WORKS_FOR = "works_for"
    CREATED_BY = "created_by"
    INFLUENCES = "influences"
    OTHER = "other"

# Request Models
class InsertRequest(BaseModel):
    """Request model for document insertion"""
    content: str = Field(..., description="Document content to insert", min_length=10, max_length=1000000)
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata for the document")
    chunk_strategy: Optional[str] = Field(default="auto", description="Chunking strategy: auto, fixed, semantic")
    force_reprocess: Optional[bool] = Field(default=False, description="Force reprocessing even if document exists")
    
    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError("Content cannot be empty or whitespace only")
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "content": "This is a sample document about artificial intelligence and machine learning...",
                "metadata": {
                    "title": "AI and ML Overview",
                    "author": "John Doe",
                    "category": "Technology",
                    "tags": ["AI", "ML", "Technology"]
                },
                "chunk_strategy": "auto",
                "force_reprocess": False
            }
        }

class QueryRequest(BaseModel):
    """Request model for knowledge graph queries"""
    query: str = Field(..., description="Query string", min_length=3, max_length=10000)
    mode: QueryMode = Field(default=QueryMode.MIX, description="Query execution mode")
    top_k: int = Field(default=10, description="Number of top results to return", ge=1, le=100)
    chunk_top_k: int = Field(default=20, description="Number of chunks to retrieve initially", ge=1, le=200)
    enable_rerank: bool = Field(default=True, description="Enable result reranking")
    conversation_history: List[Dict[str, str]] = Field(default=[], description="Previous conversation context")
    user_prompt: Optional[str] = Field(default=None, description="Additional user prompt for context")
    metadata_filters: Optional[Dict[str, Any]] = Field(default={}, description="Metadata-based filtering")
    max_entity_tokens: Optional[int] = Field(default=8000, description="Maximum tokens for entities", ge=1000, le=32000)
    max_relation_tokens: Optional[int] = Field(default=4000, description="Maximum tokens for relations", ge=1000, le=16000)
    max_total_tokens: Optional[int] = Field(default=16000, description="Maximum total tokens", ge=2000, le=64000)
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()
    
    @validator('conversation_history')
    def validate_conversation_history(cls, v):
        for item in v:
            if not isinstance(item, dict) or 'role' not in item or 'content' not in item:
                raise ValueError("Conversation history items must have 'role' and 'content' fields")
            if item['role'] not in ['user', 'assistant', 'system']:
                raise ValueError("Role must be 'user', 'assistant', or 'system'")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What are the main applications of artificial intelligence in healthcare?",
                "mode": "mix",
                "top_k": 10,
                "enable_rerank": True,
                "conversation_history": [
                    {"role": "user", "content": "Tell me about AI"},
                    {"role": "assistant", "content": "AI is a broad field..."}
                ],
                "metadata_filters": {
                    "category": ["Healthcare", "Technology"],
                    "date_range": {"start": "2020-01-01", "end": "2024-12-31"}
                }
            }
        }

class ClearCacheRequest(BaseModel):
    """Request model for cache clearing"""
    cache_type: str = Field(default="all", description="Cache type to clear: all, llm, embedding, default")
    confirm: bool = Field(default=False, description="Confirmation flag")
    
    @validator('cache_type')
    def validate_cache_type(cls, v):
        allowed_types = ["all", "llm", "embedding", "default", "naive", "local", "global", "hybrid", "mix"]
        if v not in allowed_types:
            raise ValueError(f"Cache type must be one of: {', '.join(allowed_types)}")
        return v

# Response Models
class Entity(BaseModel):
    """Entity model"""
    id: str = Field(..., description="Unique entity identifier")
    name: str = Field(..., description="Entity name")
    type: str = Field(..., description="Entity type")
    description: Optional[str] = Field(None, description="Entity description")
    properties: Optional[Dict[str, Any]] = Field(default={}, description="Additional entity properties")
    confidence_score: Optional[float] = Field(None, description="Confidence score", ge=0.0, le=1.0)
    mentions: Optional[int] = Field(None, description="Number of mentions in corpus", ge=0)

class Relationship(BaseModel):
    """Relationship model"""
    id: str = Field(..., description="Unique relationship identifier")
    source: str = Field(..., description="Source entity ID")
    target: str = Field(..., description="Target entity ID")
    relation_type: str = Field(..., description="Type of relationship")
    description: Optional[str] = Field(None, description="Relationship description")
    properties: Optional[Dict[str, Any]] = Field(default={}, description="Additional relationship properties")
    confidence_score: Optional[float] = Field(None, description="Confidence score", ge=0.0, le=1.0)
    weight: Optional[float] = Field(None, description="Relationship weight", ge=0.0)

class Context(BaseModel):
    """Context chunk model"""
    content: str = Field(..., description="Context content")
    relevance_score: float = Field(..., description="Relevance score", ge=0.0, le=1.0)
    source: Optional[str] = Field(None, description="Source document or identifier")
    chunk_id: Optional[str] = Field(None, description="Chunk identifier")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Context metadata")

class InsertResponse(BaseModel):
    """Response model for document insertion"""
    success: bool = Field(..., description="Whether insertion was successful")
    message: str = Field(..., description="Status message")
    document_id: Optional[str] = Field(None, description="Generated document ID")
    entities_extracted: int = Field(default=0, description="Number of entities extracted", ge=0)
    relationships_extracted: int = Field(default=0, description="Number of relationships extracted", ge=0)
    chunks_created: Optional[int] = Field(None, description="Number of chunks created", ge=0)
    processing_time_seconds: float = Field(..., description="Processing time in seconds", ge=0.0)
    tokens_used: int = Field(default=0, description="Tokens consumed", ge=0)
    warnings: Optional[List[str]] = Field(default=[], description="Processing warnings")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Document inserted successfully",
                "document_id": "doc_abc123",
                "entities_extracted": 15,
                "relationships_extracted": 23,
                "chunks_created": 8,
                "processing_time_seconds": 2.45,
                "tokens_used": 1250,
                "warnings": []
            }
        }

class QueryResponse(BaseModel):
    """Response model for knowledge graph queries"""
    response: str = Field(..., description="Generated response")
    entities: List[Entity] = Field(default=[], description="Related entities found")
    relationships: List[Relationship] = Field(default=[], description="Related relationships found")
    contexts: List[Context] = Field(default=[], description="Retrieved context chunks")
    metadata: Dict[str, Any] = Field(default={}, description="Query metadata and statistics")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")
    
    class Config:
        schema_extra = {
            "example": {
                "response": "Artificial intelligence has several key applications in healthcare...",
                "entities": [
                    {
                        "id": "ent_ai_001",
                        "name": "Artificial Intelligence",
                        "type": "concept",
                        "description": "Machine intelligence that mimics human cognitive functions",
                        "confidence_score": 0.95,
                        "mentions": 42
                    }
                ],
                "relationships": [
                    {
                        "id": "rel_001",
                        "source": "ent_ai_001",
                        "target": "ent_healthcare_001",
                        "relation_type": "applied_to",
                        "confidence_score": 0.87,
                        "weight": 0.6
                    }
                ],
                "contexts": [
                    {
                        "content": "AI systems in healthcare can assist with diagnosis...",
                        "relevance_score": 0.92,
                        "source": "doc_healthcare_ai",
                        "chunk_id": "chunk_001"
                    }
                ],
                "metadata": {
                    "mode": "mix",
                    "processing_time_seconds": 1.23,
                    "tokens_used": {"input": 25, "output": 340, "total": 365},
                    "entities_found": 8,
                    "relationships_found": 12,
                    "context_chunks": 5
                }
            }
        }

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="Application version")
    components: Dict[str, Dict[str, Any]] = Field(..., description="Component health details")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-08-04T12:00:00Z",
                "version": "1.0.0",
                "components": {
                    "database": {"status": "healthy", "response_time_ms": 12},
                    "llm_service": {"status": "healthy", "model": "gemini-2.5-flash"},
                    "embedding_service": {"status": "healthy", "model": "text-embedding-3-large"},
                    "cache": {"status": "healthy", "hit_rate": 0.85}
                }
            }
        }

class StatusResponse(BaseModel):
    """System status response model"""
    status: str = Field(..., description="System status")
    timestamp: datetime = Field(..., description="Status timestamp")
    statistics: Dict[str, Any] = Field(..., description="System statistics")
    configuration: Dict[str, Any] = Field(..., description="System configuration")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "operational",
                "timestamp": "2024-08-04T12:00:00Z",
                "statistics": {
                    "total_documents": 1250,
                    "total_entities": 8500,
                    "total_relationships": 12000,
                    "queries_processed": 2500,
                    "average_query_time": 1.2,
                    "cache_hit_rate": 0.78
                },
                "configuration": {
                    "model": "gemini-2.5-flash",
                    "embedding_model": "text-embedding-3-large",
                    "workspace": "graphrag_production",
                    "cache_enabled": True
                }
            }
        }

class MetricsResponse(BaseModel):
    """Metrics response for Prometheus"""
    metrics: str = Field(..., description="Prometheus-formatted metrics")
    content_type: str = Field(default="text/plain; version=0.0.4", description="Content type")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: bool = Field(True, description="Error flag")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(..., description="Error timestamp")
    path: Optional[str] = Field(None, description="Request path")
    
    class Config:
        schema_extra = {
            "example": {
                "error": True,
                "message": "Invalid query format",
                "details": {"field": "query", "issue": "Query too short"},
                "timestamp": "2024-08-04T12:00:00Z",
                "path": "/query"
            }
        }

# Utility models for complex nested data
class TokenUsage(BaseModel):
    """Token usage statistics"""
    input_tokens: int = Field(..., description="Input tokens consumed", ge=0)
    output_tokens: int = Field(..., description="Output tokens generated", ge=0)
    total_tokens: int = Field(..., description="Total tokens used", ge=0)
    estimated_cost: Optional[float] = Field(None, description="Estimated cost in USD", ge=0.0)

class ProcessingMetrics(BaseModel):
    """Processing performance metrics"""
    processing_time_seconds: float = Field(..., description="Total processing time", ge=0.0)
    queue_time_seconds: Optional[float] = Field(None, description="Time spent in queue", ge=0.0)
    model_time_seconds: Optional[float] = Field(None, description="LLM processing time", ge=0.0)
    embedding_time_seconds: Optional[float] = Field(None, description="Embedding processing time", ge=0.0)
    retrieval_time_seconds: Optional[float] = Field(None, description="Retrieval time", ge=0.0)

class SystemStats(BaseModel):
    """System statistics"""
    total_documents: int = Field(..., description="Total documents in system", ge=0)
    total_entities: int = Field(..., description="Total entities extracted", ge=0)
    total_relationships: int = Field(..., description="Total relationships found", ge=0)
    total_queries: int = Field(..., description="Total queries processed", ge=0)
    average_query_time: float = Field(..., description="Average query time in seconds", ge=0.0)
    cache_hit_rate: float = Field(..., description="Cache hit rate", ge=0.0, le=1.0)
    uptime_seconds: float = Field(..., description="System uptime in seconds", ge=0.0)
    memory_usage_mb: float = Field(..., description="Memory usage in MB", ge=0.0)

# Batch processing models
class BatchInsertRequest(BaseModel):
    """Batch document insertion request"""
    documents: List[InsertRequest] = Field(..., description="List of documents to insert", min_items=1, max_items=100)
    batch_id: Optional[str] = Field(None, description="Batch identifier")
    parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    
    @validator('documents')
    def validate_documents(cls, v):
        if len(v) > 100:
            raise ValueError("Maximum 100 documents per batch")
        return v

class BatchInsertResponse(BaseModel):
    """Batch document insertion response"""
    batch_id: str = Field(..., description="Batch identifier")
    total_documents: int = Field(..., description="Total documents processed", ge=0)
    successful_inserts: int = Field(..., description="Successful insertions", ge=0)
    failed_inserts: int = Field(..., description="Failed insertions", ge=0)
    results: List[InsertResponse] = Field(..., description="Individual results")
    total_processing_time_seconds: float = Field(..., description="Total batch processing time", ge=0.0)
    
    class Config:
        schema_extra = {
            "example": {
                "batch_id": "batch_abc123",
                "total_documents": 50,
                "successful_inserts": 48,
                "failed_inserts": 2,
                "results": [],
                "total_processing_time_seconds": 123.45
            }
        }
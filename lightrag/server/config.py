"""
Configuration management for LightRAG Server
CLAUDEFLOW implementation with environment-based settings
"""

import os
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from pydantic import BaseSettings, Field, validator
from enum import Enum

class Environment(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(str, Enum):
    """Log levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

class StorageType(str, Enum):
    """Storage backend types"""
    JSON = "json"
    POSTGRESQL = "postgresql"
    NEO4J = "neo4j"
    REDIS = "redis"

class LLMProvider(str, Enum):
    """LLM provider types"""
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"

class SystemConfig(BaseSettings):
    """System-level configuration"""
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="NODE_ENV")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    working_dir: str = Field(default="./lightrag_cache", env="LIGHTRAG_WORKING_DIR")
    workspace: str = Field(default="graphrag_production", env="LIGHTRAG_WORKSPACE")
    
    # Performance settings
    max_async: int = Field(default=8, env="LIGHTRAG_MAX_ASYNC")
    max_parallel_insert: int = Field(default=8, env="LIGHTRAG_MAX_PARALLEL_INSERT")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    
    # Cache settings
    enable_llm_cache: bool = Field(default=True, env="ENABLE_LLM_CACHE")
    enable_embedding_cache: bool = Field(default=True, env="ENABLE_EMBEDDING_CACHE")
    cache_ttl_hours: int = Field(default=24, env="CACHE_TTL_HOURS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class APIConfig(BaseSettings):
    """API server configuration"""
    host: str = Field(default="0.0.0.0", env="LIGHTRAG_HOST")
    port: int = Field(default=8000, env="LIGHTRAG_PORT")
    workers: int = Field(default=4, env="LIGHTRAG_WORKERS")
    
    # Security
    api_keys: List[str] = Field(default_factory=list, env="LIGHTRAG_API_KEY")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5678"],
        env="CORS_ORIGINS"
    )
    
    # Rate limiting
    rate_limit_requests_per_minute: int = Field(default=100, env="RATE_LIMIT_REQUESTS_PER_MINUTE")
    rate_limit_requests_per_hour: int = Field(default=1000, env="RATE_LIMIT_REQUESTS_PER_HOUR")
    
    @validator('api_keys', pre=True)
    def parse_api_keys(cls, v):
        if isinstance(v, str):
            return [key.strip() for key in v.split(',') if key.strip()]
        return v
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',') if origin.strip()]
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class LLMConfig(BaseSettings):
    """LLM configuration"""
    provider: LLMProvider = Field(default=LLMProvider.GEMINI, env="LLM_PROVIDER")
    model_name: str = Field(default="gemini-2.5-flash", env="LLM_MODEL_NAME")
    temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")
    max_tokens: int = Field(default=2048, env="LLM_MAX_TOKENS")
    
    # API configurations
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    huggingface_api_key: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    
    # Model-specific settings
    context_length: int = Field(default=64000, env="LLM_CONTEXT_LENGTH")
    max_async_requests: int = Field(default=8, env="LLM_MAX_ASYNC")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class EmbeddingConfig(BaseSettings):
    """Embedding model configuration"""
    provider: str = Field(default="openai", env="EMBEDDING_PROVIDER")
    model_name: str = Field(default="text-embedding-3-large", env="EMBEDDING_MODEL_NAME")
    batch_size: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")
    max_async: int = Field(default=16, env="EMBEDDING_MAX_ASYNC")
    dimension: int = Field(default=1536, env="EMBEDDING_DIMENSION")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class DatabaseConfig(BaseSettings):
    """Database configuration"""
    # PostgreSQL
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_db: str = Field(default="lightrag_production", env="POSTGRES_DB")
    postgres_user: str = Field(default="lightrag", env="POSTGRES_USER")
    postgres_password: str = Field(default="", env="POSTGRES_PASSWORD")
    postgres_ssl_mode: str = Field(default="prefer", env="POSTGRES_SSL_MODE")
    
    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", env="NEO4J_USER")
    neo4j_password: str = Field(default="", env="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="lightrag", env="NEO4J_DATABASE")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    @property
    def postgres_url(self) -> str:
        """Generate PostgreSQL connection URL"""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class StorageConfig(BaseSettings):
    """Storage backend configuration"""
    kv_storage_type: StorageType = Field(default=StorageType.JSON, env="KV_STORAGE_TYPE")
    vector_storage_type: str = Field(default="nano", env="VECTOR_STORAGE_TYPE")
    graph_storage_type: str = Field(default="networkx", env="GRAPH_STORAGE_TYPE")
    doc_status_storage_type: str = Field(default="json", env="DOC_STATUS_STORAGE_TYPE")
    
    # File paths for JSON storage
    kv_storage_path: str = Field(default="kv_store.json", env="KV_STORAGE_PATH")
    graph_storage_path: str = Field(default="graph.pkl", env="GRAPH_STORAGE_PATH")
    doc_status_path: str = Field(default="doc_status.json", env="DOC_STATUS_PATH")
    
    # Backup settings
    enable_backup: bool = Field(default=True, env="ENABLE_BACKUP")
    backup_interval_hours: int = Field(default=6, env="BACKUP_INTERVAL_HOURS")
    backup_retention_days: int = Field(default=30, env="BACKUP_RETENTION_DAYS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class DocumentProcessingConfig(BaseSettings):
    """Document processing configuration"""
    chunk_size: int = Field(default=1200, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, env="CHUNK_OVERLAP")
    max_document_size_mb: int = Field(default=50, env="MAX_DOCUMENT_SIZE_MB")
    
    # Supported file types
    supported_file_types: List[str] = Field(
        default=["pdf", "txt", "md", "docx", "html", "json", "csv"],
        env="SUPPORTED_FILE_TYPES"
    )
    
    # Processing settings
    enable_multimodal: bool = Field(default=True, env="ENABLE_MULTIMODAL")
    ocr_provider: str = Field(default="mistral", env="OCR_PROVIDER")
    
    @validator('supported_file_types', pre=True)
    def parse_file_types(cls, v):
        if isinstance(v, str):
            return [ft.strip() for ft in v.split(',') if ft.strip()]
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class MonitoringConfig(BaseSettings):
    """Monitoring and observability configuration"""
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    enable_health_checks: bool = Field(default=True, env="ENABLE_HEALTH_CHECKS")
    enable_tracing: bool = Field(default=False, env="ENABLE_TRACING")
    
    # Prometheus
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # Health check intervals
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    
    # Alerting
    alert_webhook_url: Optional[str] = Field(default=None, env="ALERT_WEBHOOK_URL")
    discord_webhook_url: Optional[str] = Field(default=None, env="DISCORD_WEBHOOK_URL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class SecurityConfig(BaseSettings):
    """Security configuration"""
    encryption_key: str = Field(default="", env="ENCRYPTION_KEY")
    jwt_secret: str = Field(default="", env="N8N_JWT_SECRET")
    
    # SSL/TLS
    enable_ssl: bool = Field(default=False, env="ENABLE_SSL")
    ssl_cert_path: Optional[str] = Field(default=None, env="SSL_CERT_PATH")
    ssl_key_path: Optional[str] = Field(default=None, env="SSL_KEY_PATH")
    
    # Content filtering
    enable_content_filtering: bool = Field(default=True, env="ENABLE_CONTENT_FILTERING")
    enable_pii_detection: bool = Field(default=True, env="ENABLE_PII_DETECTION")
    
    # Data retention
    data_retention_days: int = Field(default=90, env="DATA_RETENTION_DAYS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class Config:
    """Main configuration class combining all sub-configurations"""
    
    def __init__(self):
        self.system = SystemConfig()
        self.api = APIConfig()
        self.llm = LLMConfig()
        self.embedding = EmbeddingConfig()
        self.database = DatabaseConfig()
        self.storage = StorageConfig()
        self.document_processing = DocumentProcessingConfig()
        self.monitoring = MonitoringConfig()
        self.security = SecurityConfig()
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration consistency"""
        # Ensure required API keys are present
        if self.llm.provider == LLMProvider.OPENAI and not self.llm.openai_api_key:
            raise ValueError("OpenAI API key is required when using OpenAI provider")
        
        if self.llm.provider == LLMProvider.GEMINI and not self.llm.gemini_api_key:
            raise ValueError("Gemini API key is required when using Gemini provider")
        
        # Ensure working directory exists
        Path(self.system.working_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate API keys format
        if not self.api.api_keys:
            raise ValueError("At least one API key must be configured")
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration for LightRAG"""
        return {
            "model_name": self.llm.model_name,
            "temperature": self.llm.temperature,
            "max_tokens": self.llm.max_tokens,
            "context_length": self.llm.context_length,
            "max_async": self.llm.max_async_requests,
        }
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration for LightRAG"""
        return {
            "model_name": self.embedding.model_name,
            "batch_size": self.embedding.batch_size,
            "max_async": self.embedding.max_async,
            "dimension": self.embedding.dimension,
        }
    
    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration for LightRAG"""
        config = {
            "kv_storage_type": self.storage.kv_storage_type.value,
            "vector_storage_type": self.storage.vector_storage_type,
            "graph_storage_type": self.storage.graph_storage_type,
        }
        
        # Add database URLs if using database storage
        if self.storage.kv_storage_type == StorageType.POSTGRESQL:
            config["postgres_url"] = self.database.postgres_url
        
        return config
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.system.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.system.environment == Environment.PRODUCTION
    
    def dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "system": self.system.dict(),
            "api": self.api.dict(),
            "llm": self.llm.dict(),
            "embedding": self.embedding.dict(),
            "database": self.database.dict(),
            "storage": self.storage.dict(),
            "document_processing": self.document_processing.dict(),
            "monitoring": self.monitoring.dict(),
            "security": self.security.dict(),
        }

# Global configuration instance
config = Config()
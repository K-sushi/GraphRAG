"""
Utility classes for LightRAG Server
CLAUDEFLOW implementation with monitoring and tracking
"""

import asyncio
import logging
import time
import psutil
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from pathlib import Path
import tiktoken
import aiohttp
from prometheus_client import Counter, Histogram, Gauge, generate_latest

logger = logging.getLogger(__name__)

@dataclass
class TokenUsage:
    """Token usage tracking"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        self.total_tokens = self.input_tokens + self.output_tokens

@dataclass
class QueryMetrics:
    """Query performance metrics"""
    query_id: str
    query_text: str
    processing_time_seconds: float
    token_usage: TokenUsage
    mode: str
    top_k: int
    entities_found: int
    relationships_found: int
    contexts_retrieved: int
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class SystemHealth:
    """System health status"""
    component: str
    status: str  # healthy, degraded, unhealthy
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    last_check: datetime = None
    
    def __post_init__(self):
        if self.last_check is None:
            self.last_check = datetime.utcnow()

class TokenTracker:
    """Track token usage across LLM calls"""
    
    def __init__(self):
        self.usage_history: deque = deque(maxlen=10000)  # Keep last 10k entries
        self.daily_usage: Dict[str, TokenUsage] = {}
        self.encoding = None
        self._initialize_tokenizer()
    
    def _initialize_tokenizer(self):
        """Initialize tiktoken tokenizer"""
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        except Exception as e:
            logger.warning(f"Failed to initialize tiktoken: {e}")
            self.encoding = None
    
    def estimate_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Estimate token count for text"""
        if not text:
            return 0
        
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception as e:
                logger.warning(f"Token estimation failed: {e}")
        
        # Fallback: rough estimation (4 characters per token)
        return len(text) // 4
    
    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "unknown",
        operation: str = "query"
    ) -> TokenUsage:
        """Record token usage"""
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=self._estimate_cost(input_tokens, output_tokens, model)
        )
        
        self.usage_history.append({
            "timestamp": usage.timestamp.isoformat(),
            "model": model,
            "operation": operation,
            "usage": asdict(usage)
        })
        
        # Update daily usage
        date_key = usage.timestamp.strftime("%Y-%m-%d")
        if date_key not in self.daily_usage:
            self.daily_usage[date_key] = TokenUsage()
        
        daily = self.daily_usage[date_key]
        daily.input_tokens += input_tokens
        daily.output_tokens += output_tokens
        daily.total_tokens += input_tokens + output_tokens
        daily.estimated_cost += usage.estimated_cost
        
        return usage
    
    def _estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Estimate cost based on model pricing"""
        # Rough pricing estimates (per 1K tokens)
        pricing = {
            "gemini-2.5-pro": {"input": 0.001, "output": 0.002},
            "gemini-2.5-flash": {"input": 0.0005, "output": 0.001},
            "gemini-2.5-flash-lite": {"input": 0.0002, "output": 0.0005},
            "gpt-4o": {"input": 0.01, "output": 0.03},
            "gpt-4o-mini": {"input": 0.0015, "output": 0.006},
            "text-embedding-3-large": {"input": 0.00013, "output": 0.00013},
            "default": {"input": 0.001, "output": 0.002}
        }
        
        model_pricing = pricing.get(model, pricing["default"])
        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]
        
        return input_cost + output_cost
    
    def get_daily_usage(self, date: Optional[str] = None) -> TokenUsage:
        """Get token usage for specific date"""
        if date is None:
            date = datetime.utcnow().strftime("%Y-%m-%d")
        
        return self.daily_usage.get(date, TokenUsage())
    
    def get_usage_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get usage statistics for recent period"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent_usage = [
            entry for entry in self.usage_history
            if datetime.fromisoformat(entry["timestamp"]) > cutoff
        ]
        
        if not recent_usage:
            return {"total_entries": 0, "total_tokens": 0, "estimated_cost": 0.0}
        
        total_tokens = sum(entry["usage"]["total_tokens"] for entry in recent_usage)
        total_cost = sum(entry["usage"]["estimated_cost"] for entry in recent_usage)
        
        # Model breakdown
        model_stats = defaultdict(lambda: {"count": 0, "tokens": 0, "cost": 0.0})
        for entry in recent_usage:
            model = entry["model"]
            model_stats[model]["count"] += 1
            model_stats[model]["tokens"] += entry["usage"]["total_tokens"]
            model_stats[model]["cost"] += entry["usage"]["estimated_cost"]
        
        return {
            "total_entries": len(recent_usage),
            "total_tokens": total_tokens,
            "estimated_cost": total_cost,
            "average_tokens_per_request": total_tokens / len(recent_usage),
            "model_breakdown": dict(model_stats),
            "period_hours": hours
        }

class HealthChecker:
    """System health monitoring"""
    
    def __init__(self):
        self.health_status: Dict[str, SystemHealth] = {}
        self.monitoring_active = False
        self.check_interval = 30  # seconds
        self._monitoring_task = None
    
    async def start_monitoring(self):
        """Start background health monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop background health monitoring"""
        self.monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                await self.check_all_components()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def check_all_components(self):
        """Check health of all system components"""
        checks = [
            self._check_system_resources(),
            self._check_database_connection(),
            self._check_llm_service(),
            self._check_embedding_service(),
            self._check_cache_service()
        ]
        
        await asyncio.gather(*checks, return_exceptions=True)
    
    async def _check_system_resources(self):
        """Check system resource usage"""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Determine status
            if memory_usage > 90 or cpu_usage > 90 or disk_usage > 90:
                status = "unhealthy"
                error_msg = f"High resource usage: Memory {memory_usage:.1f}%, CPU {cpu_usage:.1f}%, Disk {disk_usage:.1f}%"
            elif memory_usage > 75 or cpu_usage > 75 or disk_usage > 75:
                status = "degraded"
                error_msg = f"Elevated resource usage: Memory {memory_usage:.1f}%, CPU {cpu_usage:.1f}%, Disk {disk_usage:.1f}%"
            else:
                status = "healthy"
                error_msg = None
            
            self.health_status["system_resources"] = SystemHealth(
                component="system_resources",
                status=status,
                error_message=error_msg
            )
            
        except Exception as e:
            self.health_status["system_resources"] = SystemHealth(
                component="system_resources",
                status="unhealthy",
                error_message=str(e)
            )
    
    async def _check_database_connection(self):
        """Check database connectivity"""
        try:
            # This would be implemented based on your database configuration
            # For now, we'll simulate a basic check
            start_time = time.time()
            
            # Simulate database check (replace with actual DB connection test)
            await asyncio.sleep(0.01)  # Simulate network latency
            
            response_time = (time.time() - start_time) * 1000
            
            self.health_status["database"] = SystemHealth(
                component="database",
                status="healthy",
                response_time_ms=response_time
            )
            
        except Exception as e:
            self.health_status["database"] = SystemHealth(
                component="database",
                status="unhealthy",
                error_message=str(e)
            )
    
    async def _check_llm_service(self):
        """Check LLM service connectivity"""
        try:
            # This would test actual LLM connectivity
            # For now, simulate a basic check
            start_time = time.time()
            
            # Simulate LLM API check
            await asyncio.sleep(0.05)  # Simulate API latency
            
            response_time = (time.time() - start_time) * 1000
            
            self.health_status["llm_service"] = SystemHealth(
                component="llm_service",
                status="healthy",
                response_time_ms=response_time
            )
            
        except Exception as e:
            self.health_status["llm_service"] = SystemHealth(
                component="llm_service",
                status="unhealthy",
                error_message=str(e)
            )
    
    async def _check_embedding_service(self):
        """Check embedding service connectivity"""
        try:
            start_time = time.time()
            
            # Simulate embedding service check
            await asyncio.sleep(0.02)
            
            response_time = (time.time() - start_time) * 1000
            
            self.health_status["embedding_service"] = SystemHealth(
                component="embedding_service",
                status="healthy",
                response_time_ms=response_time
            )
            
        except Exception as e:
            self.health_status["embedding_service"] = SystemHealth(
                component="embedding_service",
                status="unhealthy",
                error_message=str(e)
            )
    
    async def _check_cache_service(self):
        """Check cache service connectivity"""
        try:
            start_time = time.time()
            
            # Simulate cache service check
            await asyncio.sleep(0.005)
            
            response_time = (time.time() - start_time) * 1000
            
            self.health_status["cache"] = SystemHealth(
                component="cache",
                status="healthy",
                response_time_ms=response_time
            )
            
        except Exception as e:
            self.health_status["cache"] = SystemHealth(
                component="cache",
                status="unhealthy",
                error_message=str(e)
            )
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        if not self.health_status:
            await self.check_all_components()
        
        # Determine overall health
        statuses = [health.status for health in self.health_status.values()]
        if "unhealthy" in statuses:
            overall_status = "unhealthy"
        elif "degraded" in statuses:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        components = {}
        for name, health in self.health_status.items():
            components[name] = {
                "status": health.status,
                "response_time_ms": health.response_time_ms,
                "error_message": health.error_message,
                "last_check": health.last_check.isoformat()
            }
        
        return {
            "healthy": overall_status == "healthy",
            "overall_status": overall_status,
            "components": components,
            "last_updated": datetime.utcnow().isoformat()
        }

class MetricsCollector:
    """Prometheus metrics collection"""
    
    def __init__(self):
        # Request metrics
        self.request_counter = Counter(
            'lightrag_requests_total',
            'Total requests processed',
            ['method', 'endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'lightrag_request_duration_seconds',
            'Request processing time',
            ['method', 'endpoint']
        )
        
        # Query metrics
        self.query_counter = Counter(
            'lightrag_queries_total',
            'Total queries processed',
            ['mode', 'status']
        )
        
        self.query_duration = Histogram(
            'lightrag_query_duration_seconds',
            'Query processing time',
            ['mode']
        )
        
        # Token metrics
        self.token_usage = Counter(
            'lightrag_tokens_total',
            'Total tokens consumed',
            ['model', 'type']  # type: input, output
        )
        
        # System metrics
        self.memory_usage = Gauge(
            'lightrag_memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self.cpu_usage = Gauge(
            'lightrag_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        # Active connections
        self.active_connections = Gauge(
            'lightrag_active_connections',
            'Number of active connections'
        )
        
        # Collection state
        self.collecting = False
        self._collection_task = None
        self.collection_interval = 15  # seconds
    
    async def start_collection(self):
        """Start metrics collection"""
        if self.collecting:
            return
        
        self.collecting = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Metrics collection started")
    
    async def stop_collection(self):
        """Stop metrics collection"""
        self.collecting = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        logger.info("Metrics collection stopped")
    
    async def _collection_loop(self):
        """Background metrics collection loop"""
        while self.collecting:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_system_metrics(self):
        """Collect system metrics"""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.used)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            self.cpu_usage.set(cpu_percent)
            
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics"""
        self.request_counter.labels(
            method=method,
            endpoint=endpoint,
            status=str(status)
        ).inc()
        
        self.request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_query(self, mode: str, duration: float, status: str = "success"):
        """Record query metrics"""
        self.query_counter.labels(mode=mode, status=status).inc()
        self.query_duration.labels(mode=mode).observe(duration)
    
    def record_tokens(self, model: str, input_tokens: int, output_tokens: int):
        """Record token usage"""
        self.token_usage.labels(model=model, type="input").inc(input_tokens)
        self.token_usage.labels(model=model, type="output").inc(output_tokens)
    
    def record_insertion(
        self,
        processing_time: float,
        input_tokens: int,
        content_size: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record document insertion metrics"""
        # This could be expanded with more specific insertion metrics
        self.request_duration.labels(
            method="POST",
            endpoint="/insert"
        ).observe(processing_time)
    
    async def get_prometheus_metrics(self) -> str:
        """Get Prometheus-formatted metrics"""
        return generate_latest().decode('utf-8')
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()
            
            # Get process-specific stats
            process = psutil.Process()
            process_memory = process.memory_info()
            
            return {
                "system": {
                    "memory_total_gb": memory.total / (1024**3),
                    "memory_used_gb": memory.used / (1024**3),
                    "memory_usage_percent": memory.percent,
                    "cpu_usage_percent": cpu_percent,
                },
                "process": {
                    "memory_rss_mb": process_memory.rss / (1024**2),
                    "memory_vms_mb": process_memory.vms / (1024**2),
                    "cpu_percent": process.cpu_percent(),
                    "threads": process.num_threads(),
                    "connections": len(process.connections()),
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"System stats collection failed: {e}")
            return {"error": str(e)}

class CircuitBreaker:
    """Circuit breaker for external service calls"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Exception = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
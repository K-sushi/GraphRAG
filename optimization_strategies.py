#!/usr/bin/env python3
"""
GraphRAG Performance Optimization Strategies
Implements specific optimizations for <10 second Perplexity-style responses

Key Optimization Areas:
1. Parallel Processing & Async Operations
2. Intelligent Caching Systems
3. Model Selection & Request Optimization
4. Memory Management & Resource Optimization
5. Streaming & Progressive Response
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict, deque
import weakref
import gc

# Caching imports
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Performance monitoring
import psutil
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    data: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: float = 300.0  # 5 minutes default
    
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl
    
    def access(self):
        self.access_count += 1

class IntelligentCacheManager:
    """
    Intelligent caching system for GraphRAG components
    Supports multi-level caching with TTL and LRU eviction
    """
    
    def __init__(self, max_memory_mb: int = 512, redis_url: Optional[str] = None):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory_bytes = 0
        self.cache = {}
        self.access_times = deque()
        self.lock = threading.RLock()
        
        # Redis connection for distributed caching
        self.redis_client = None
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("Redis cache enabled")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_usage_bytes': 0
        }
    
    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate cache key from data"""
        if isinstance(data, str):
            content = data
        elif isinstance(data, dict):
            content = json.dumps(data, sort_keys=True)
        else:
            content = str(data)
        
        hash_object = hashlib.md5(content.encode())
        return f"{prefix}:{hash_object.hexdigest()}"
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate memory size of data"""
        try:
            return len(pickle.dumps(data))
        except:
            return len(str(data)) * 2  # Rough estimate
    
    def _evict_lru(self, required_bytes: int):
        """Evict least recently used items"""
        with self.lock:
            evicted = 0
            while (self.current_memory_bytes + required_bytes > self.max_memory_bytes 
                   and self.cache):
                
                # Find LRU item
                lru_key = None
                lru_time = float('inf')
                
                for key, entry in self.cache.items():
                    if entry.timestamp < lru_time:
                        lru_time = entry.timestamp
                        lru_key = key
                
                if lru_key:
                    entry = self.cache.pop(lru_key)
                    self.current_memory_bytes -= entry.size_bytes
                    evicted += 1
                    self.stats['evictions'] += 1
            
            logger.debug(f"Evicted {evicted} cache entries to free memory")
    
    async def get(self, key: str, prefix: str = "default") -> Optional[Any]:
        """Get item from cache"""
        cache_key = self._generate_key(prefix, key)
        
        with self.lock:
            # Check local cache first
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                if not entry.is_expired():
                    entry.access()
                    self.stats['hits'] += 1
                    return entry.data
                else:
                    # Remove expired entry
                    self.current_memory_bytes -= entry.size_bytes
                    del self.cache[cache_key]
        
        # Check Redis cache if available
        if self.redis_client:
            try:
                redis_data = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.get, cache_key
                )
                if redis_data:
                    data = pickle.loads(redis_data)
                    # Store in local cache for faster access
                    await self.set(key, data, prefix=prefix, ttl=300)
                    self.stats['hits'] += 1
                    return data
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
        
        self.stats['misses'] += 1
        return None
    
    async def set(self, key: str, data: Any, prefix: str = "default", ttl: float = 300.0):
        """Set item in cache"""
        cache_key = self._generate_key(prefix, key)
        size_bytes = self._estimate_size(data)
        
        # Check if data is too large
        if size_bytes > self.max_memory_bytes * 0.1:  # 10% of max memory
            logger.warning(f"Cache item too large ({size_bytes} bytes), skipping")
            return
        
        with self.lock:
            # Evict items if necessary
            self._evict_lru(size_bytes)
            
            # Create cache entry
            entry = CacheEntry(
                data=data,
                timestamp=time.time(),
                size_bytes=size_bytes,
                ttl=ttl
            )
            
            # Remove existing entry if present
            if cache_key in self.cache:
                old_entry = self.cache[cache_key]
                self.current_memory_bytes -= old_entry.size_bytes
            
            # Store in local cache
            self.cache[cache_key] = entry
            self.current_memory_bytes += size_bytes
            self.stats['memory_usage_bytes'] = self.current_memory_bytes
        
        # Store in Redis if available
        if self.redis_client:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self.redis_client.setex(
                        cache_key, 
                        int(ttl), 
                        pickle.dumps(data)
                    )
                )
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            hit_rate = (self.stats['hits'] / 
                       (self.stats['hits'] + self.stats['misses']) 
                       if (self.stats['hits'] + self.stats['misses']) > 0 else 0)
            
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'cache_size': len(self.cache),
                'memory_usage_mb': self.current_memory_bytes / (1024 * 1024)
            }

class ParallelProcessingManager:
    """
    Manages parallel processing for GraphRAG operations
    Optimizes CPU-bound and I/O-bound operations separately
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(8, (psutil.cpu_count() or 1) + 4)
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.semaphore = asyncio.Semaphore(self.max_workers)
        
        logger.info(f"ParallelProcessingManager initialized with {self.max_workers} workers")
    
    async def execute_parallel_operations(self, operations: List[Tuple[callable, tuple, dict]]) -> List[Any]:
        """
        Execute multiple operations in parallel
        
        Args:
            operations: List of (function, args, kwargs) tuples
            
        Returns:
            List of results in same order as operations
        """
        async def execute_single(operation):
            func, args, kwargs = operation
            async with self.semaphore:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        self.thread_executor, 
                        lambda: func(*args, **kwargs)
                    )
        
        tasks = [execute_single(op) for op in operations]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def execute_with_timeout(self, operation: callable, timeout: float, *args, **kwargs) -> Any:
        """Execute operation with timeout"""
        try:
            if asyncio.iscoroutinefunction(operation):
                return await asyncio.wait_for(operation(*args, **kwargs), timeout=timeout)
            else:
                loop = asyncio.get_event_loop()
                return await asyncio.wait_for(
                    loop.run_in_executor(self.thread_executor, lambda: operation(*args, **kwargs)),
                    timeout=timeout
                )
        except asyncio.TimeoutError:
            logger.warning(f"Operation timed out after {timeout}s")
            raise
    
    def cleanup(self):
        """Clean up resources"""
        self.thread_executor.shutdown(wait=True)

class ModelSelectionOptimizer:
    """
    Optimizes model selection based on query characteristics and performance requirements
    """
    
    def __init__(self):
        self.model_performance = {
            'gemini-2.0-flash-exp': {
                'avg_response_time': 1.2,
                'cost_per_1k_tokens': 0.001,
                'max_context': 32768,
                'quality_score': 0.85,
                'best_for': ['quick_search', 'simple_queries', 'web_search']
            },
            'gemini-1.5-pro-002': {
                'avg_response_time': 3.5,
                'cost_per_1k_tokens': 0.005,
                'max_context': 131072,
                'quality_score': 0.95,
                'best_for': ['complex_analysis', 'reasoning', 'synthesis']
            },
            'gemini-1.5-flash-002': {
                'avg_response_time': 2.0,
                'cost_per_1k_tokens': 0.002,
                'max_context': 65536,
                'quality_score': 0.90,
                'best_for': ['balanced_queries', 'medium_complexity']
            }
        }
        
        self.query_patterns = {
            'simple': r'what is|define|explain simply',
            'current_info': r'current|latest|now|today|recent',
            'complex_analysis': r'analyze|compare|evaluate|assess|how does',
            'synthesis': r'combine|integrate|synthesize|comprehensive'
        }
    
    def select_optimal_model(self, query: str, context_length: int = 0, 
                           priority: str = 'balanced') -> str:
        """
        Select optimal model based on query characteristics
        
        Args:
            query: User query
            context_length: Estimated context length needed
            priority: 'speed', 'quality', 'balanced', or 'cost'
            
        Returns:
            Optimal model name
        """
        query_lower = query.lower()
        
        # Determine query complexity
        complexity_score = 0
        if any(pattern in query_lower for pattern in ['analyze', 'compare', 'evaluate']):
            complexity_score += 2
        if any(pattern in query_lower for pattern in ['current', 'latest', 'now']):
            complexity_score += 1
        if len(query) > 100:
            complexity_score += 1
        if context_length > 10000:
            complexity_score += 1
        
        # Filter models by context requirements
        suitable_models = {
            name: info for name, info in self.model_performance.items()
            if info['max_context'] >= context_length
        }
        
        if not suitable_models:
            return 'gemini-1.5-pro-002'  # Fallback to largest context model
        
        # Score models based on priority
        model_scores = {}
        for name, info in suitable_models.items():
            if priority == 'speed':
                score = 1.0 / info['avg_response_time']
            elif priority == 'quality':
                score = info['quality_score']
            elif priority == 'cost':
                score = 1.0 / info['cost_per_1k_tokens']
            else:  # balanced
                speed_score = 1.0 / info['avg_response_time']
                quality_score = info['quality_score']
                cost_score = 1.0 / info['cost_per_1k_tokens']
                score = (speed_score + quality_score + cost_score) / 3
            
            # Adjust for complexity
            if complexity_score >= 3 and name == 'gemini-1.5-pro-002':
                score *= 1.3  # Bonus for complex queries
            elif complexity_score <= 1 and name == 'gemini-2.0-flash-exp':
                score *= 1.2  # Bonus for simple queries
            
            model_scores[name] = score
        
        # Return highest scoring model
        return max(model_scores.items(), key=lambda x: x[1])[0]
    
    def get_model_recommendations(self, query_batch: List[str]) -> Dict[str, List[str]]:
        """Get model recommendations for a batch of queries"""
        recommendations = defaultdict(list)
        
        for query in query_batch:
            model = self.select_optimal_model(query)
            recommendations[model].append(query)
        
        return dict(recommendations)

class MemoryOptimizer:
    """
    Optimizes memory usage through intelligent garbage collection and object pooling
    """
    
    def __init__(self):
        self.object_pools = {}
        self.memory_threshold_mb = 1024  # 1GB threshold
        self.gc_frequency = 100  # GC every 100 operations
        self.operation_count = 0
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def should_trigger_gc(self) -> bool:
        """Determine if garbage collection should be triggered"""
        self.operation_count += 1
        
        # Check memory threshold
        current_memory = self.get_memory_usage()
        if current_memory > self.memory_threshold_mb:
            return True
        
        # Check operation frequency
        if self.operation_count % self.gc_frequency == 0:
            return True
        
        return False
    
    def optimize_memory(self):
        """Perform memory optimization"""
        if self.should_trigger_gc():
            initial_memory = self.get_memory_usage()
            
            # Force garbage collection
            gc.collect()
            
            final_memory = self.get_memory_usage()
            freed_mb = initial_memory - final_memory
            
            if freed_mb > 10:  # Only log if significant memory freed
                logger.info(f"Memory optimization freed {freed_mb:.1f} MB")
    
    def create_object_pool(self, object_type: str, factory_func: callable, max_size: int = 100):
        """Create an object pool for reusing expensive objects"""
        self.object_pools[object_type] = {
            'factory': factory_func,
            'pool': deque(),
            'max_size': max_size,
            'created': 0,
            'reused': 0
        }
    
    def get_pooled_object(self, object_type: str, *args, **kwargs):
        """Get object from pool or create new one"""
        if object_type not in self.object_pools:
            raise ValueError(f"Object pool '{object_type}' not found")
        
        pool_info = self.object_pools[object_type]
        
        if pool_info['pool']:
            obj = pool_info['pool'].popleft()
            pool_info['reused'] += 1
            return obj
        else:
            obj = pool_info['factory'](*args, **kwargs)
            pool_info['created'] += 1
            return obj
    
    def return_pooled_object(self, object_type: str, obj):
        """Return object to pool"""
        if object_type not in self.object_pools:
            return
        
        pool_info = self.object_pools[object_type]
        
        if len(pool_info['pool']) < pool_info['max_size']:
            # Reset object state if it has a reset method
            if hasattr(obj, 'reset'):
                obj.reset()
            pool_info['pool'].append(obj)

class StreamingResponseManager:
    """
    Manages streaming responses for progressive user experience
    """
    
    def __init__(self):
        self.active_streams = {}
        self.stream_buffers = defaultdict(list)
    
    async def create_stream(self, stream_id: str, total_steps: int) -> 'StreamingResponse':
        """Create a new streaming response"""
        stream = StreamingResponse(stream_id, total_steps)
        self.active_streams[stream_id] = stream
        return stream
    
    async def update_stream(self, stream_id: str, step: int, data: Any, partial: bool = True):
        """Update streaming response with new data"""
        if stream_id in self.active_streams:
            stream = self.active_streams[stream_id]
            await stream.update(step, data, partial)
    
    async def complete_stream(self, stream_id: str, final_data: Any):
        """Complete streaming response"""
        if stream_id in self.active_streams:
            stream = self.active_streams[stream_id]
            await stream.complete(final_data)
            del self.active_streams[stream_id]

class StreamingResponse:
    """Individual streaming response handler"""
    
    def __init__(self, stream_id: str, total_steps: int):
        self.stream_id = stream_id
        self.total_steps = total_steps
        self.current_step = 0
        self.data_chunks = []
        self.start_time = time.time()
        self.callbacks = []
    
    def add_callback(self, callback: callable):
        """Add callback for stream updates"""
        self.callbacks.append(callback)
    
    async def update(self, step: int, data: Any, partial: bool = True):
        """Update stream with new data"""
        self.current_step = step
        self.data_chunks.append({
            'step': step,
            'data': data,
            'partial': partial,
            'timestamp': time.time()
        })
        
        # Notify callbacks
        for callback in self.callbacks:
            try:
                await callback(self.stream_id, step, data, partial)
            except Exception as e:
                logger.warning(f"Stream callback error: {e}")
    
    async def complete(self, final_data: Any):
        """Complete the stream"""
        elapsed_time = time.time() - self.start_time
        completion_data = {
            'stream_id': self.stream_id,
            'final_data': final_data,
            'total_steps': self.total_steps,
            'elapsed_time': elapsed_time,
            'chunks': self.data_chunks
        }
        
        # Notify callbacks of completion
        for callback in self.callbacks:
            try:
                await callback(self.stream_id, -1, completion_data, False)
            except Exception as e:
                logger.warning(f"Stream completion callback error: {e}")

class OptimizedGraphRAGPipeline:
    """
    GraphRAG pipeline with integrated performance optimizations
    Combines all optimization strategies for maximum performance
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize optimization components
        self.cache_manager = IntelligentCacheManager(
            max_memory_mb=config.get('cache_memory_mb', 512),
            redis_url=config.get('redis_url')
        )
        
        self.parallel_manager = ParallelProcessingManager(
            max_workers=config.get('max_workers', 8)
        )
        
        self.model_optimizer = ModelSelectionOptimizer()
        self.memory_optimizer = MemoryOptimizer()
        self.streaming_manager = StreamingResponseManager()
        
        # Performance tracking
        self.query_stats = defaultdict(list)
        self.optimization_stats = {
            'cache_hits': 0,
            'parallel_operations': 0,
            'model_optimizations': 0,
            'memory_optimizations': 0
        }
        
        logger.info("OptimizedGraphRAGPipeline initialized with all optimizations")
    
    async def process_perplexity_query(self, query: str, user_id: str = "default") -> Dict[str, Any]:
        """
        Process Perplexity-style query with all optimizations enabled
        Target: <10 second response time
        """
        start_time = time.time()
        stream_id = f"{user_id}_{int(start_time)}"
        
        try:
            # Create streaming response
            stream = await self.streaming_manager.create_stream(stream_id, 6)
            
            # Step 1: Query analysis and caching check (0.5s target)
            await stream.update(1, {"status": "Analyzing query...", "progress": 16})
            
            cache_key = f"query_analysis_{query}"
            cached_analysis = await self.cache_manager.get(cache_key, "query_analysis")
            
            if cached_analysis:
                query_analysis = cached_analysis
                self.optimization_stats['cache_hits'] += 1
            else:
                query_analysis = await self._analyze_query_optimized(query)
                await self.cache_manager.set(cache_key, query_analysis, "query_analysis", ttl=1800)
            
            # Step 2: Parallel web search and knowledge retrieval (3s target)
            await stream.update(2, {"status": "Searching web and knowledge base...", "progress": 33})
            
            parallel_operations = []
            
            # Web search if needed
            if query_analysis.get('needs_realtime', True):
                parallel_operations.append((
                    self._web_search_optimized, 
                    (query, query_analysis), 
                    {}
                ))
            
            # Knowledge graph search
            parallel_operations.append((
                self._knowledge_search_optimized, 
                (query, query_analysis), 
                {}
            ))
            
            # Vector similarity search
            parallel_operations.append((
                self._vector_search_optimized, 
                (query, query_analysis), 
                {}
            ))
            
            # Execute searches in parallel
            search_results = await self.parallel_manager.execute_parallel_operations(parallel_operations)
            self.optimization_stats['parallel_operations'] += len(parallel_operations)
            
            # Step 3: Context synthesis (2s target)
            await stream.update(3, {"status": "Synthesizing context...", "progress": 50})
            
            context = await self._synthesize_context_optimized(search_results, query_analysis)
            
            # Step 4: Model selection and response generation (3s target)
            await stream.update(4, {"status": "Generating response...", "progress": 66})
            
            optimal_model = self.model_optimizer.select_optimal_model(
                query, 
                len(str(context)),
                priority='balanced'
            )
            self.optimization_stats['model_optimizations'] += 1
            
            response = await self._generate_response_optimized(query, context, optimal_model)
            
            # Step 5: Post-processing and formatting (1s target)
            await stream.update(5, {"status": "Formatting response...", "progress": 83})
            
            formatted_response = await self._format_response_optimized(response, search_results)
            
            # Step 6: Finalization
            await stream.update(6, {"status": "Complete", "progress": 100})
            
            # Calculate performance metrics
            total_time = time.time() - start_time
            
            final_result = {
                'response': formatted_response,
                'sources': self._extract_sources(search_results),
                'metadata': {
                    'query_time': total_time,
                    'model_used': optimal_model,
                    'cache_hits': query_analysis.get('cache_used', False),
                    'parallel_operations': len(parallel_operations),
                    'stream_id': stream_id
                },
                'performance': {
                    'target_met': total_time <= 10.0,
                    'response_time': total_time,
                    'optimization_stats': self.optimization_stats.copy()
                }
            }
            
            # Complete stream
            await self.streaming_manager.complete_stream(stream_id, final_result)
            
            # Memory optimization
            self.memory_optimizer.optimize_memory()
            self.optimization_stats['memory_optimizations'] += 1
            
            # Track query statistics
            self.query_stats[query[:50]].append(total_time)
            
            logger.info(f"Query processed in {total_time:.2f}s (target: 10s)")
            return final_result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            await stream.update(-1, {"status": "Error", "error": str(e)}, partial=False)
            raise
    
    async def _analyze_query_optimized(self, query: str) -> Dict[str, Any]:
        """Optimized query analysis"""
        # Mock implementation - replace with actual analysis
        return {
            'complexity': 'medium',
            'needs_realtime': 'current' in query.lower() or 'latest' in query.lower(),
            'topics': ['general'],
            'intent': 'information_seeking'
        }
    
    async def _web_search_optimized(self, query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimized web search with caching"""
        cache_key = f"web_search_{query}"
        cached_result = await self.cache_manager.get(cache_key, "web_search")
        
        if cached_result:
            return cached_result
        
        # Mock web search - replace with actual implementation
        await asyncio.sleep(0.5)  # Simulate search time
        result = {
            'sources': [
                {'title': 'Example Source 1', 'url': 'https://example.com/1', 'snippet': 'Relevant information...'},
                {'title': 'Example Source 2', 'url': 'https://example.com/2', 'snippet': 'More relevant information...'}
            ],
            'search_time': 0.5
        }
        
        await self.cache_manager.set(cache_key, result, "web_search", ttl=300)
        return result
    
    async def _knowledge_search_optimized(self, query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimized knowledge graph search"""
        # Mock knowledge search
        await asyncio.sleep(0.3)
        return {
            'entities': ['Entity1', 'Entity2'],
            'relationships': [('Entity1', 'relates_to', 'Entity2')],
            'communities': ['Community1']
        }
    
    async def _vector_search_optimized(self, query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimized vector similarity search"""
        # Mock vector search
        await asyncio.sleep(0.2)
        return {
            'similar_documents': [
                {'content': 'Similar document 1', 'similarity': 0.85},
                {'content': 'Similar document 2', 'similarity': 0.78}
            ]
        }
    
    async def _synthesize_context_optimized(self, search_results: List[Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimized context synthesis"""
        # Mock context synthesis
        await asyncio.sleep(0.4)
        return {
            'web_context': search_results[0] if len(search_results) > 0 else {},
            'knowledge_context': search_results[1] if len(search_results) > 1 else {},
            'vector_context': search_results[2] if len(search_results) > 2 else {}
        }
    
    async def _generate_response_optimized(self, query: str, context: Dict[str, Any], model: str) -> str:
        """Optimized response generation"""
        # Mock response generation
        await asyncio.sleep(1.0)
        return f"Optimized response for: {query} using {model}"
    
    async def _format_response_optimized(self, response: str, search_results: List[Any]) -> Dict[str, Any]:
        """Optimized response formatting"""
        return {
            'text': response,
            'formatted': True,
            'citations': self._extract_citations(search_results)
        }
    
    def _extract_sources(self, search_results: List[Any]) -> List[Dict[str, Any]]:
        """Extract sources from search results"""
        sources = []
        for result in search_results:
            if isinstance(result, dict) and 'sources' in result:
                sources.extend(result['sources'])
        return sources
    
    def _extract_citations(self, search_results: List[Any]) -> List[str]:
        """Extract citations from search results"""
        citations = []
        for result in search_results:
            if isinstance(result, dict) and 'sources' in result:
                for source in result['sources']:
                    if 'url' in source:
                        citations.append(source['url'])
        return citations
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        cache_stats = self.cache_manager.get_stats()
        
        # Calculate average query times
        avg_query_times = {}
        for query, times in self.query_stats.items():
            avg_query_times[query] = {
                'avg_time': np.mean(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'query_count': len(times)
            }
        
        return {
            'cache_performance': cache_stats,
            'optimization_stats': self.optimization_stats,
            'query_performance': avg_query_times,
            'memory_usage_mb': self.memory_optimizer.get_memory_usage(),
            'active_streams': len(self.streaming_manager.active_streams)
        }
    
    async def cleanup(self):
        """Clean up resources"""
        self.parallel_manager.cleanup()
        logger.info("OptimizedGraphRAGPipeline cleanup completed")

# Example usage and testing
async def test_optimized_pipeline():
    """Test the optimized pipeline"""
    config = {
        'cache_memory_mb': 256,
        'max_workers': 4,
        'redis_url': None  # Set to Redis URL if available
    }
    
    pipeline = OptimizedGraphRAGPipeline(config)
    
    test_queries = [
        "What is the current price of Bitcoin?",
        "How is AI impacting healthcare in 2024?",
        "What are the latest developments in renewable energy?"
    ]
    
    logger.info("Testing optimized GraphRAG pipeline")
    
    for i, query in enumerate(test_queries):
        logger.info(f"Processing query {i+1}: {query}")
        
        start_time = time.time()
        result = await pipeline.process_perplexity_query(query, f"test_user_{i}")
        elapsed_time = time.time() - start_time
        
        logger.info(f"Query {i+1} completed in {elapsed_time:.2f}s")
        logger.info(f"Target met: {result['performance']['target_met']}")
    
    # Print performance statistics
    stats = pipeline.get_performance_stats()
    logger.info(f"Performance stats: {json.dumps(stats, indent=2)}")
    
    await pipeline.cleanup()

if __name__ == "__main__":
    asyncio.run(test_optimized_pipeline())
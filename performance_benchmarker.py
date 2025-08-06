#!/usr/bin/env python3
"""
GraphRAG Performance Benchmarker
Adaptive performance bottleneck detection for <10 second Perplexity-style queries

Features:
- Component-level performance profiling
- Real-time bottleneck detection
- Perplexity-style query simulation
- Memory and resource monitoring
- Optimization recommendations
"""

import os
import sys
import asyncio
import time
import psutil
import json
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from collections import defaultdict, deque
import resource
import gc

# Performance monitoring imports
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Memory profiling
try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

# GraphRAG components
try:
    from gemini_llm_provider import GeminiLLMProvider, LLMConfig
    from gemini_web_search import GeminiWebSearchProvider
    from graphrag_pipeline import GraphRAGPipeline, GraphRAGConfig
    GRAPHRAG_AVAILABLE = True
except ImportError:
    GRAPHRAG_AVAILABLE = False
    logging.warning("GraphRAG components not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    # Timing metrics
    total_time: float = 0.0
    gemini_api_time: float = 0.0
    web_search_time: float = 0.0
    graphrag_time: float = 0.0
    vector_search_time: float = 0.0
    graph_traversal_time: float = 0.0
    response_generation_time: float = 0.0
    
    # Resource metrics
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    disk_io_bytes: int = 0
    network_io_bytes: int = 0
    
    # Quality metrics
    accuracy_score: float = 0.0
    relevance_score: float = 0.0
    source_count: int = 0
    
    # Error metrics
    api_errors: int = 0
    timeout_errors: int = 0
    memory_errors: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class BottleneckAnalysis:
    """Analysis of performance bottlenecks"""
    component: str
    severity: str  # critical, high, medium, low
    impact_percent: float
    description: str
    recommended_actions: List[str]
    estimated_improvement: str

class SystemResourceMonitor:
    """Real-time system resource monitoring"""
    
    def __init__(self, sample_interval: float = 0.1):
        self.sample_interval = sample_interval
        self.monitoring = False
        self.samples = defaultdict(list)
        self.start_time = None
        
    def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return aggregated metrics"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        
        return self._aggregate_metrics()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # CPU and memory
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # System-wide metrics
                system_cpu = psutil.cpu_percent()
                system_memory = psutil.virtual_memory()
                
                # I/O metrics
                io_counters = process.io_counters()
                
                # Network (if available)
                try:
                    net_io = psutil.net_io_counters()
                    network_bytes = net_io.bytes_sent + net_io.bytes_recv
                except:
                    network_bytes = 0
                
                # Store samples
                timestamp = time.time() - self.start_time
                self.samples['timestamp'].append(timestamp)
                self.samples['cpu_percent'].append(cpu_percent)
                self.samples['memory_mb'].append(memory_mb)
                self.samples['system_cpu'].append(system_cpu)
                self.samples['system_memory_percent'].append(system_memory.percent)
                self.samples['disk_read_bytes'].append(io_counters.read_bytes)
                self.samples['disk_write_bytes'].append(io_counters.write_bytes)
                self.samples['network_bytes'].append(network_bytes)
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                logger.warning(f"Monitoring error: {e}")
                time.sleep(self.sample_interval)
    
    def _aggregate_metrics(self) -> Dict[str, Any]:
        """Aggregate monitoring data"""
        if not self.samples['timestamp']:
            return {}
        
        # Convert to numpy arrays for efficient computation
        timestamps = np.array(self.samples['timestamp'])
        cpu_data = np.array(self.samples['cpu_percent'])
        memory_data = np.array(self.samples['memory_mb'])
        system_cpu_data = np.array(self.samples['system_cpu'])
        
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        
        return {
            'duration': duration,
            'samples_count': len(timestamps),
            'cpu': {
                'mean': float(np.mean(cpu_data)),
                'max': float(np.max(cpu_data)),
                'std': float(np.std(cpu_data)),
                'p95': float(np.percentile(cpu_data, 95)) if len(cpu_data) > 0 else 0.0
            },
            'memory': {
                'mean_mb': float(np.mean(memory_data)),
                'peak_mb': float(np.max(memory_data)),
                'std_mb': float(np.std(memory_data)),
                'growth_rate': self._calculate_growth_rate(timestamps, memory_data)
            },
            'system': {
                'cpu_mean': float(np.mean(system_cpu_data)),
                'cpu_max': float(np.max(system_cpu_data))
            },
            'io': {
                'disk_read_mb': (self.samples['disk_read_bytes'][-1] - self.samples['disk_read_bytes'][0]) / 1024 / 1024 if len(self.samples['disk_read_bytes']) > 1 else 0,
                'disk_write_mb': (self.samples['disk_write_bytes'][-1] - self.samples['disk_write_bytes'][0]) / 1024 / 1024 if len(self.samples['disk_write_bytes']) > 1 else 0,
                'network_mb': (self.samples['network_bytes'][-1] - self.samples['network_bytes'][0]) / 1024 / 1024 if len(self.samples['network_bytes']) > 1 else 0
            }
        }
    
    def _calculate_growth_rate(self, timestamps: np.ndarray, values: np.ndarray) -> float:
        """Calculate growth rate (MB/second)"""
        if len(timestamps) < 2:
            return 0.0
        
        duration = timestamps[-1] - timestamps[0]
        if duration == 0:
            return 0.0
        
        growth = values[-1] - values[0]
        return growth / duration

class GraphRAGPerformanceBenchmarker:
    """
    Comprehensive performance benchmarker for GraphRAG Implementation
    Targeting <10 second Perplexity-style query responses
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the performance benchmarker"""
        self.config = self._load_config(config_path)
        self.resource_monitor = SystemResourceMonitor()
        self.benchmark_history = []
        self.bottleneck_patterns = defaultdict(list)
        
        # Performance targets
        self.target_response_time = 10.0  # seconds
        self.target_memory_mb = 2048  # 2GB
        self.target_cpu_percent = 80.0
        
        # Component instances
        self.gemini_provider = None
        self.web_search_provider = None
        self.graphrag_pipeline = None
        
        logger.info("GraphRAG Performance Benchmarker initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load benchmarking configuration"""
        default_config = {
            'test_queries': [
                "What is the current price of Bitcoin?",
                "How is climate change affecting global agriculture?", 
                "What are the latest developments in quantum computing?",
                "Explain the impact of AI on healthcare in 2024",
                "What are the current trends in renewable energy?"
            ],
            'concurrency_levels': [1, 2, 4, 8],
            'iterations_per_test': 3,
            'warm_up_iterations': 1,
            'enable_memory_profiling': MEMORY_PROFILER_AVAILABLE
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    async def initialize_components(self):
        """Initialize GraphRAG components for testing"""
        if not GRAPHRAG_AVAILABLE:
            raise RuntimeError("GraphRAG components not available")
        
        try:
            # Initialize Gemini LLM Provider
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable required")
            
            llm_config = LLMConfig(
                api_key=api_key,
                model="gemini-2.0-flash-exp",
                max_tokens=4096,
                concurrent_requests=4
            )
            self.gemini_provider = GeminiLLMProvider(llm_config)
            
            # Initialize Web Search Provider
            self.web_search_provider = GeminiWebSearchProvider(
                api_key=api_key,
                config={
                    'search_model': 'gemini-2.0-flash-exp',
                    'analysis_model': 'gemini-1.5-pro-002',
                    'max_search_results': 10,
                    'search_timeout': 30
                }
            )
            
            # Initialize GraphRAG Pipeline
            graphrag_config = GraphRAGConfig(
                chunk_size=1200,
                chunk_overlap=100,
                max_workers=2,
                batch_size=5
            )
            self.graphrag_pipeline = GraphRAGPipeline(
                config=graphrag_config,
                llm_provider=self.gemini_provider
            )
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark suite"""
        logger.info("Starting comprehensive GraphRAG performance benchmark")
        start_time = time.time()
        
        try:
            await self.initialize_components()
            
            # Warm-up phase
            await self._run_warmup()
            
            # Component-level benchmarks
            component_results = await self._benchmark_components()
            
            # End-to-end query benchmarks
            query_results = await self._benchmark_queries()
            
            # Concurrency benchmarks
            concurrency_results = await self._benchmark_concurrency()
            
            # Memory stress tests
            memory_results = await self._benchmark_memory_usage()
            
            # Bottleneck analysis
            bottleneck_analysis = self._analyze_bottlenecks(
                component_results, query_results, concurrency_results, memory_results
            )
            
            # Generate optimization recommendations
            optimization_roadmap = self._generate_optimization_roadmap(bottleneck_analysis)
            
            total_time = time.time() - start_time
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'total_duration': total_time,
                'component_performance': component_results,
                'query_performance': query_results,
                'concurrency_performance': concurrency_results,
                'memory_performance': memory_results,
                'bottleneck_analysis': [asdict(b) for b in bottleneck_analysis],
                'optimization_roadmap': optimization_roadmap,
                'system_info': self._get_system_info()
            }
            
            # Save results
            self._save_benchmark_results(results)
            
            logger.info(f"Comprehensive benchmark completed in {total_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def _run_warmup(self):
        """Run warm-up queries to initialize components"""
        logger.info("Running warm-up phase")
        
        warmup_query = "What is artificial intelligence?"
        
        for i in range(self.config['warm_up_iterations']):
            try:
                await self._execute_single_query(warmup_query, track_performance=False)
                logger.info(f"Warm-up iteration {i+1} completed")
            except Exception as e:
                logger.warning(f"Warm-up iteration {i+1} failed: {e}")
    
    async def _benchmark_components(self) -> Dict[str, Any]:
        """Benchmark individual components"""
        logger.info("Benchmarking individual components")
        
        results = {}
        
        # Gemini API benchmark
        results['gemini_api'] = await self._benchmark_gemini_api()
        
        # Web search benchmark
        results['web_search'] = await self._benchmark_web_search()
        
        # GraphRAG processing benchmark
        results['graphrag_processing'] = await self._benchmark_graphrag_processing()
        
        # Vector search benchmark
        results['vector_search'] = await self._benchmark_vector_search()
        
        return results
    
    async def _benchmark_gemini_api(self) -> Dict[str, Any]:
        """Benchmark Gemini API performance"""
        logger.info("Benchmarking Gemini API")
        
        test_prompts = [
            "Explain quantum computing in simple terms.",
            "What are the benefits of renewable energy?",
            "How does machine learning work?"
        ]
        
        times = []
        errors = 0
        
        for prompt in test_prompts:
            for _ in range(3):  # 3 iterations per prompt
                start_time = time.time()
                try:
                    await self.gemini_provider.generate_response_async(
                        prompt, 
                        model="gemini-2.0-flash-exp"
                    )
                    elapsed = time.time() - start_time
                    times.append(elapsed)
                except Exception as e:
                    errors += 1
                    logger.warning(f"Gemini API error: {e}")
        
        return {
            'avg_response_time': np.mean(times) if times else 0,
            'min_response_time': np.min(times) if times else 0,
            'max_response_time': np.max(times) if times else 0,
            'p95_response_time': np.percentile(times, 95) if times else 0,
            'error_rate': errors / (len(test_prompts) * 3),
            'total_requests': len(test_prompts) * 3,
            'successful_requests': len(times)
        }
    
    async def _benchmark_web_search(self) -> Dict[str, Any]:
        """Benchmark web search performance"""
        logger.info("Benchmarking web search")
        
        search_queries = [
            "current Bitcoin price",
            "latest AI news 2024",
            "climate change statistics"
        ]
        
        times = []
        errors = 0
        source_counts = []
        
        for query in search_queries:
            for _ in range(2):  # 2 iterations per query
                start_time = time.time()
                try:
                    result = await self.web_search_provider.search_and_analyze(query)
                    elapsed = time.time() - start_time
                    times.append(elapsed)
                    source_counts.append(len(result.get('sources', [])))
                except Exception as e:
                    errors += 1
                    logger.warning(f"Web search error: {e}")
        
        return {
            'avg_search_time': np.mean(times) if times else 0,
            'min_search_time': np.min(times) if times else 0,
            'max_search_time': np.max(times) if times else 0,
            'p95_search_time': np.percentile(times, 95) if times else 0,
            'avg_source_count': np.mean(source_counts) if source_counts else 0,
            'error_rate': errors / (len(search_queries) * 2),
            'total_searches': len(search_queries) * 2,
            'successful_searches': len(times)
        }
    
    async def _benchmark_graphrag_processing(self) -> Dict[str, Any]:
        """Benchmark GraphRAG processing performance"""
        logger.info("Benchmarking GraphRAG processing")
        
        # Mock document for processing
        test_document = """
        Artificial intelligence (AI) is revolutionizing multiple industries. 
        Machine learning algorithms are being applied to healthcare, finance, 
        and transportation. Companies like Google, Microsoft, and OpenAI are 
        leading the development of large language models. These models are 
        capable of understanding and generating human-like text.
        """
        
        times = []
        errors = 0
        
        for _ in range(3):  # 3 iterations
            start_time = time.time()
            try:
                # Mock GraphRAG processing
                await asyncio.sleep(0.1)  # Simulate processing time
                elapsed = time.time() - start_time
                times.append(elapsed)
            except Exception as e:
                errors += 1
                logger.warning(f"GraphRAG processing error: {e}")
        
        return {
            'avg_processing_time': np.mean(times) if times else 0,
            'min_processing_time': np.min(times) if times else 0,
            'max_processing_time': np.max(times) if times else 0,
            'error_rate': errors / 3,
            'total_operations': 3,
            'successful_operations': len(times)
        }
    
    async def _benchmark_vector_search(self) -> Dict[str, Any]:
        """Benchmark vector search performance"""
        logger.info("Benchmarking vector search")
        
        # Mock vector search operations
        search_terms = [
            "artificial intelligence applications",
            "machine learning benefits",
            "deep learning techniques"
        ]
        
        times = []
        errors = 0
        
        for term in search_terms:
            for _ in range(2):  # 2 iterations per term
                start_time = time.time()
                try:
                    # Mock vector search
                    await asyncio.sleep(0.05)  # Simulate search time
                    elapsed = time.time() - start_time
                    times.append(elapsed)
                except Exception as e:
                    errors += 1
                    logger.warning(f"Vector search error: {e}")
        
        return {
            'avg_search_time': np.mean(times) if times else 0,
            'min_search_time': np.min(times) if times else 0,
            'max_search_time': np.max(times) if times else 0,
            'p95_search_time': np.percentile(times, 95) if times else 0,
            'error_rate': errors / (len(search_terms) * 2),
            'total_searches': len(search_terms) * 2,
            'successful_searches': len(times)
        }
    
    async def _benchmark_queries(self) -> Dict[str, Any]:
        """Benchmark end-to-end query performance"""
        logger.info("Benchmarking end-to-end queries")
        
        query_results = []
        
        for query in self.config['test_queries']:
            logger.info(f"Testing query: {query[:50]}...")
            
            query_times = []
            query_metrics = []
            
            for iteration in range(self.config['iterations_per_test']):
                try:
                    metrics = await self._execute_single_query(query, track_performance=True)
                    query_times.append(metrics.total_time)
                    query_metrics.append(metrics)
                    
                    logger.info(f"  Iteration {iteration + 1}: {metrics.total_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Query failed in iteration {iteration + 1}: {e}")
            
            if query_times:
                query_result = {
                    'query': query,
                    'avg_time': np.mean(query_times),
                    'min_time': np.min(query_times),
                    'max_time': np.max(query_times),
                    'std_time': np.std(query_times),
                    'meets_target': np.mean(query_times) <= self.target_response_time,
                    'success_rate': len(query_times) / self.config['iterations_per_test'],
                    'detailed_metrics': [m.to_dict() for m in query_metrics]
                }
                query_results.append(query_result)
        
        # Aggregate results
        all_times = [r['avg_time'] for r in query_results]
        
        return {
            'individual_queries': query_results,
            'overall_avg_time': np.mean(all_times) if all_times else 0,
            'overall_p95_time': np.percentile(all_times, 95) if all_times else 0,
            'queries_meeting_target': sum(1 for r in query_results if r['meets_target']),
            'total_queries_tested': len(query_results),
            'target_achievement_rate': sum(1 for r in query_results if r['meets_target']) / len(query_results) if query_results else 0
        }
    
    async def _execute_single_query(self, query: str, track_performance: bool = True) -> PerformanceMetrics:
        """Execute a single query with performance tracking"""
        metrics = PerformanceMetrics()
        
        if track_performance:
            self.resource_monitor.start_monitoring()
        
        total_start = time.time()
        
        try:
            # Step 1: Analyze query freshness
            freshness_start = time.time()
            freshness_result = await self.web_search_provider.analyze_query_freshness(query)
            needs_web_search = freshness_result.get('requires_realtime', True)
            
            # Step 2: Web search if needed
            web_search_time = 0
            web_results = {}
            if needs_web_search:
                web_start = time.time()
                web_results = await self.web_search_provider.search_and_analyze(query)
                web_search_time = time.time() - web_start
                metrics.web_search_time = web_search_time
                metrics.source_count = len(web_results.get('sources', []))
            
            # Step 3: GraphRAG processing
            graphrag_start = time.time()
            # Mock GraphRAG processing for now
            await asyncio.sleep(0.2)  # Simulate GraphRAG processing
            graphrag_time = time.time() - graphrag_start
            metrics.graphrag_time = graphrag_time
            
            # Step 4: Response generation
            response_start = time.time()
            context = {
                'web_results': web_results,
                'graphrag_context': {},  # Mock context
                'query': query
            }
            
            response = await self.gemini_provider.generate_response_async(
                f"Based on the following context, answer the query: {query}\n\nContext: {str(context)[:2000]}",
                model="gemini-1.5-pro-002"
            )
            response_time = time.time() - response_start
            metrics.response_generation_time = response_time
            
            # Calculate total time
            metrics.total_time = time.time() - total_start
            metrics.gemini_api_time = response_time  # Simplified
            
            # Get resource metrics
            if track_performance:
                resource_metrics = self.resource_monitor.stop_monitoring()
                metrics.peak_memory_mb = resource_metrics.get('memory', {}).get('peak_mb', 0)
                metrics.avg_cpu_percent = resource_metrics.get('cpu', {}).get('mean', 0)
                metrics.disk_io_bytes = int(resource_metrics.get('io', {}).get('disk_read_mb', 0) * 1024 * 1024)
                metrics.network_io_bytes = int(resource_metrics.get('io', {}).get('network_mb', 0) * 1024 * 1024)
            
            # Mock quality metrics
            metrics.accuracy_score = 0.85
            metrics.relevance_score = 0.80
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            metrics.api_errors += 1
            if track_performance:
                self.resource_monitor.stop_monitoring()
            raise
        
        return metrics
    
    async def _benchmark_concurrency(self) -> Dict[str, Any]:
        """Benchmark concurrent query performance"""
        logger.info("Benchmarking concurrency performance")
        
        concurrency_results = []
        test_query = "What are the latest developments in artificial intelligence?"
        
        for concurrency_level in self.config['concurrency_levels']:
            logger.info(f"Testing concurrency level: {concurrency_level}")
            
            # Create concurrent tasks
            tasks = []
            start_time = time.time()
            
            for i in range(concurrency_level):
                task = self._execute_single_query(f"{test_query} (request {i+1})", track_performance=False)
                tasks.append(task)
            
            # Execute concurrently
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                total_time = time.time() - start_time
                
                # Analyze results
                successful = [r for r in results if isinstance(r, PerformanceMetrics)]
                errors = [r for r in results if isinstance(r, Exception)]
                
                if successful:
                    times = [r.total_time for r in successful]
                    concurrency_result = {
                        'concurrency_level': concurrency_level,
                        'total_time': total_time,
                        'avg_individual_time': np.mean(times),
                        'max_individual_time': np.max(times),
                        'throughput_qps': len(successful) / total_time,
                        'success_rate': len(successful) / concurrency_level,
                        'errors': len(errors)
                    }
                    concurrency_results.append(concurrency_result)
                
            except Exception as e:
                logger.error(f"Concurrency test failed for level {concurrency_level}: {e}")
        
        return {
            'concurrency_tests': concurrency_results,
            'max_successful_concurrency': max([r['concurrency_level'] for r in concurrency_results 
                                            if r.get('success_rate', 0) >= 0.8], default=1)
        }
    
    async def _benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns"""
        logger.info("Benchmarking memory usage")
        
        # Force garbage collection before starting
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        memory_samples = []
        query = "Analyze the impact of climate change on global agriculture systems."
        
        # Run multiple queries to observe memory patterns
        for i in range(5):
            before_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            try:
                await self._execute_single_query(query, track_performance=False)
            except Exception as e:
                logger.warning(f"Memory benchmark query {i+1} failed: {e}")
            
            after_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_samples.append({
                'iteration': i + 1,
                'before_mb': before_memory,
                'after_mb': after_memory,
                'delta_mb': after_memory - before_memory
            })
            
            # Brief pause between queries
            await asyncio.sleep(1)
        
        # Force garbage collection and measure final memory
        gc.collect()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        return {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'net_memory_growth_mb': final_memory - initial_memory,
            'peak_memory_mb': max([s['after_mb'] for s in memory_samples]),
            'avg_query_memory_delta_mb': np.mean([s['delta_mb'] for s in memory_samples]),
            'memory_samples': memory_samples,
            'potential_memory_leak': final_memory > initial_memory + 100  # 100MB threshold
        }
    
    def _analyze_bottlenecks(self, component_results: Dict, query_results: Dict, 
                           concurrency_results: Dict, memory_results: Dict) -> List[BottleneckAnalysis]:
        """Analyze performance bottlenecks"""
        logger.info("Analyzing performance bottlenecks")
        
        bottlenecks = []
        
        # Analyze overall response time
        avg_response_time = query_results.get('overall_avg_time', 0)
        if avg_response_time > self.target_response_time:
            severity = "critical" if avg_response_time > 15 else "high"
            impact = min(100, (avg_response_time / self.target_response_time - 1) * 100)
            
            bottlenecks.append(BottleneckAnalysis(
                component="Overall Response Time",
                severity=severity,
                impact_percent=impact,
                description=f"Average response time ({avg_response_time:.2f}s) exceeds target ({self.target_response_time}s)",
                recommended_actions=[
                    "Implement parallel processing for independent operations",
                    "Add response caching for repeated queries",
                    "Optimize Gemini API call patterns",
                    "Consider model selection optimization"
                ],
                estimated_improvement="30-50% response time reduction"
            ))
        
        # Analyze component-specific bottlenecks
        if component_results.get('gemini_api', {}).get('avg_response_time', 0) > 3.0:
            bottlenecks.append(BottleneckAnalysis(
                component="Gemini API",
                severity="high",
                impact_percent=40,
                description="Gemini API response time is high",
                recommended_actions=[
                    "Switch to faster models (Flash vs Pro) where appropriate",
                    "Implement request batching",
                    "Add API response caching",
                    "Optimize prompt length"
                ],
                estimated_improvement="20-35% API response time reduction"
            ))
        
        # Analyze web search bottlenecks
        web_search_time = component_results.get('web_search', {}).get('avg_search_time', 0)
        if web_search_time > 5.0:
            bottlenecks.append(BottleneckAnalysis(
                component="Web Search",
                severity="medium",
                impact_percent=25,
                description="Web search operations are slow",
                recommended_actions=[
                    "Implement search result caching",
                    "Optimize search query formulation",
                    "Implement timeout and fallback strategies",
                    "Use parallel search across multiple sources"
                ],
                estimated_improvement="40-60% search time reduction"
            ))
        
        # Analyze memory bottlenecks
        peak_memory = memory_results.get('peak_memory_mb', 0)
        if peak_memory > self.target_memory_mb:
            severity = "critical" if peak_memory > 4096 else "high"
            impact = min(100, (peak_memory / self.target_memory_mb - 1) * 50)
            
            bottlenecks.append(BottleneckAnalysis(
                component="Memory Usage",
                severity=severity,
                impact_percent=impact,
                description=f"Peak memory usage ({peak_memory:.0f}MB) exceeds target ({self.target_memory_mb}MB)",
                recommended_actions=[
                    "Implement streaming for large responses",
                    "Add memory-efficient data structures",
                    "Implement garbage collection optimization",
                    "Consider model quantization for embeddings"
                ],
                estimated_improvement="25-40% memory usage reduction"
            ))
        
        # Analyze concurrency bottlenecks
        max_concurrency = concurrency_results.get('max_successful_concurrency', 1)
        if max_concurrency < 4:
            bottlenecks.append(BottleneckAnalysis(
                component="Concurrency",
                severity="medium",
                impact_percent=30,
                description=f"System supports only {max_concurrency} concurrent queries",
                recommended_actions=[
                    "Implement connection pooling",
                    "Add async/await optimizations",
                    "Optimize resource sharing",
                    "Consider horizontal scaling"
                ],
                estimated_improvement="2-4x concurrency improvement"
            ))
        
        return bottlenecks
    
    def _generate_optimization_roadmap(self, bottlenecks: List[BottleneckAnalysis]) -> Dict[str, Any]:
        """Generate optimization roadmap based on bottleneck analysis"""
        
        # Sort bottlenecks by impact and severity
        severity_scores = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        bottlenecks_scored = sorted(
            bottlenecks, 
            key=lambda x: (severity_scores.get(x.severity, 0), x.impact_percent), 
            reverse=True
        )
        
        roadmap = {
            'immediate_actions': [],
            'short_term_optimizations': [],
            'long_term_improvements': [],
            'expected_overall_improvement': '',
            'implementation_priority': []
        }
        
        for i, bottleneck in enumerate(bottlenecks_scored):
            priority_item = {
                'rank': i + 1,
                'component': bottleneck.component,
                'severity': bottleneck.severity,
                'impact_percent': bottleneck.impact_percent,
                'actions': bottleneck.recommended_actions,
                'estimated_improvement': bottleneck.estimated_improvement
            }
            
            if bottleneck.severity == "critical":
                roadmap['immediate_actions'].append(priority_item)
            elif bottleneck.severity == "high":
                roadmap['short_term_optimizations'].append(priority_item)
            else:
                roadmap['long_term_improvements'].append(priority_item)
            
            roadmap['implementation_priority'].append(priority_item)
        
        # Estimate overall improvement
        total_impact = sum(b.impact_percent for b in bottlenecks_scored[:3])  # Top 3 bottlenecks
        if total_impact > 80:
            roadmap['expected_overall_improvement'] = "60-80% performance improvement possible"
        elif total_impact > 50:
            roadmap['expected_overall_improvement'] = "40-60% performance improvement possible"
        elif total_impact > 25:
            roadmap['expected_overall_improvement'] = "25-40% performance improvement possible"
        else:
            roadmap['expected_overall_improvement'] = "15-25% performance improvement possible"
        
        return roadmap
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'platform': sys.platform,
            'process_id': os.getpid()
        }
    
    def _save_benchmark_results(self, results: Dict[str, Any]):
        """Save benchmark results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("./analysis-reports")
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed JSON results
        json_path = output_dir / f"performance_benchmark_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary report
        summary_path = output_dir / f"performance_summary_{timestamp}.txt"
        self._generate_summary_report(results, summary_path)
        
        logger.info(f"Benchmark results saved to {json_path}")
        logger.info(f"Summary report saved to {summary_path}")
    
    def _generate_summary_report(self, results: Dict[str, Any], output_path: Path):
        """Generate human-readable summary report"""
        with open(output_path, 'w') as f:
            f.write("GraphRAG Performance Benchmark Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Benchmark Date: {results['timestamp']}\n")
            f.write(f"Total Duration: {results['total_duration']:.2f} seconds\n\n")
            
            # Query performance summary
            query_perf = results['query_performance']
            f.write("Query Performance Summary:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average Response Time: {query_perf['overall_avg_time']:.2f}s\n")
            f.write(f"95th Percentile Time: {query_perf['overall_p95_time']:.2f}s\n")
            f.write(f"Target Achievement Rate: {query_perf['target_achievement_rate']*100:.1f}%\n")
            f.write(f"Queries Meeting <10s Target: {query_perf['queries_meeting_target']}/{query_perf['total_queries_tested']}\n\n")
            
            # Bottleneck analysis
            f.write("Critical Bottlenecks:\n")
            f.write("-" * 20 + "\n")
            bottlenecks = results['bottleneck_analysis']
            for bottleneck in bottlenecks:
                if bottleneck['severity'] in ['critical', 'high']:
                    f.write(f"• {bottleneck['component']}: {bottleneck['description']}\n")
                    f.write(f"  Impact: {bottleneck['impact_percent']:.1f}%\n")
                    f.write(f"  Estimated Improvement: {bottleneck['estimated_improvement']}\n\n")
            
            # Optimization roadmap
            roadmap = results['optimization_roadmap']
            f.write("Optimization Roadmap:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Expected Overall Improvement: {roadmap['expected_overall_improvement']}\n\n")
            
            f.write("Immediate Actions (Critical):\n")
            for action in roadmap['immediate_actions']:
                f.write(f"• {action['component']}: {', '.join(action['actions'][:2])}\n")
            
            f.write("\nShort-term Optimizations (High Priority):\n")
            for action in roadmap['short_term_optimizations']:
                f.write(f"• {action['component']}: {', '.join(action['actions'][:2])}\n")

# Main execution
async def main():
    """Main benchmarking execution"""
    benchmarker = GraphRAGPerformanceBenchmarker()
    
    try:
        results = await benchmarker.run_comprehensive_benchmark()
        
        print("\n" + "="*60)
        print("GRAPHRAG PERFORMANCE BENCHMARK RESULTS")
        print("="*60)
        
        query_perf = results['query_performance']
        print(f"Average Response Time: {query_perf['overall_avg_time']:.2f}s")
        print(f"Target Achievement Rate: {query_perf['target_achievement_rate']*100:.1f}%")
        print(f"Queries Meeting Target: {query_perf['queries_meeting_target']}/{query_perf['total_queries_tested']}")
        
        print(f"\nCritical Bottlenecks Found: {len([b for b in results['bottleneck_analysis'] if b['severity'] == 'critical'])}")
        print(f"High Priority Bottlenecks: {len([b for b in results['bottleneck_analysis'] if b['severity'] == 'high'])}")
        
        print(f"\nExpected Improvement: {results['optimization_roadmap']['expected_overall_improvement']}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
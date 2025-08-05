"""
Tests for utility classes.
"""

import asyncio
import time
import pytest
from unittest.mock import patch, MagicMock

from utils import TokenTracker, HealthChecker, MetricsCollector, CircuitBreaker


class TestTokenTracker:
    """Test TokenTracker functionality."""
    
    def test_token_tracker_initialization(self):
        """Test TokenTracker initialization."""
        tracker = TokenTracker(max_tokens_per_minute=1000)
        
        assert tracker.max_tokens_per_minute == 1000
        assert len(tracker.usage_history) == 0
        assert tracker.current_minute_tokens == 0
    
    def test_can_consume_tokens_within_limit(self):
        """Test token consumption within limits."""
        tracker = TokenTracker(max_tokens_per_minute=1000)
        
        assert tracker.can_consume(500) is True
        assert tracker.can_consume(1000) is True
        assert tracker.can_consume(1001) is False
    
    def test_consume_tokens(self):
        """Test token consumption tracking."""
        tracker = TokenTracker(max_tokens_per_minute=1000)
        
        tracker.consume_tokens(300)
        assert tracker.current_minute_tokens == 300
        
        tracker.consume_tokens(200)
        assert tracker.current_minute_tokens == 500
        
        assert tracker.can_consume(600) is False
        assert tracker.can_consume(500) is True
    
    def test_token_usage_reset_after_minute(self):
        """Test that token usage resets after a minute."""
        tracker = TokenTracker(max_tokens_per_minute=1000)
        
        # Consume tokens
        tracker.consume_tokens(800)
        assert tracker.current_minute_tokens == 800
        
        # Mock time to simulate minute passing
        with patch('time.time', return_value=time.time() + 61):
            tracker._reset_if_minute_passed()
            assert tracker.current_minute_tokens == 0
    
    def test_usage_history_tracking(self):
        """Test usage history tracking."""
        tracker = TokenTracker(max_tokens_per_minute=1000)
        
        tracker.consume_tokens(100, operation="insert")
        tracker.consume_tokens(200, operation="query")
        
        assert len(tracker.usage_history) == 2
        assert tracker.usage_history[0]['tokens'] == 100
        assert tracker.usage_history[0]['operation'] == "insert"
        assert tracker.usage_history[1]['tokens'] == 200
        assert tracker.usage_history[1]['operation'] == "query"
    
    def test_get_usage_stats(self):
        """Test usage statistics calculation."""
        tracker = TokenTracker(max_tokens_per_minute=1000)
        
        tracker.consume_tokens(100)
        tracker.consume_tokens(200)
        tracker.consume_tokens(150)
        
        stats = tracker.get_usage_stats()
        
        assert stats['total_tokens'] == 450
        assert stats['total_requests'] == 3
        assert stats['average_tokens_per_request'] == 150
        assert 'current_minute_tokens' in stats
        assert 'remaining_tokens' in stats


class TestHealthChecker:
    """Test HealthChecker functionality."""
    
    @pytest.fixture
    def health_checker(self):
        """Create a HealthChecker instance."""
        return HealthChecker()
    
    @pytest.mark.asyncio
    async def test_health_checker_initialization(self, health_checker):
        """Test HealthChecker initialization."""
        assert health_checker.is_monitoring is False
        assert len(health_checker.component_status) == 0
    
    @pytest.mark.asyncio
    async def test_system_health_check(self, health_checker):
        """Test system health check."""
        health_status = await health_checker.check_system_health()
        
        assert isinstance(health_status, dict)
        assert 'overall_health' in health_status
        assert 'components' in health_status
        assert 'timestamp' in health_status
    
    @pytest.mark.asyncio
    async def test_component_health_check(self, health_checker):
        """Test individual component health checks."""
        # Mock system resources
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu, \
             patch('psutil.disk_usage') as mock_disk:
            
            mock_memory.return_value = MagicMock(percent=50.0)
            mock_cpu.return_value = 30.0
            mock_disk.return_value = MagicMock(used=1000, total=10000)
            
            await health_checker._check_system_resources()
            
            assert 'system_resources' in health_checker.component_status
            status = health_checker.component_status['system_resources']
            assert status['status'] == 'healthy'
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, health_checker):
        """Test monitoring start/stop lifecycle."""
        # Start monitoring
        await health_checker.start_monitoring()
        assert health_checker.is_monitoring is True
        
        # Let it run briefly
        await asyncio.sleep(0.1)
        
        # Stop monitoring
        await health_checker.stop_monitoring()
        assert health_checker.is_monitoring is False
    
    @pytest.mark.asyncio
    async def test_unhealthy_system_detection(self, health_checker):
        """Test detection of unhealthy system conditions."""
        # Mock high resource usage
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            mock_memory.return_value = MagicMock(percent=95.0)  # High memory
            mock_cpu.return_value = 95.0  # High CPU
            
            await health_checker._check_system_resources()
            
            status = health_checker.component_status['system_resources']
            assert status['status'] == 'unhealthy'
            assert 'high resource usage' in status['message'].lower()


class TestMetricsCollector:
    """Test MetricsCollector functionality."""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create a MetricsCollector instance."""
        return MetricsCollector()
    
    def test_metrics_collector_initialization(self, metrics_collector):
        """Test MetricsCollector initialization."""
        assert metrics_collector.is_collecting is False
        assert isinstance(metrics_collector.request_metrics, dict)
        assert isinstance(metrics_collector.error_metrics, dict)
    
    def test_record_request_metrics(self, metrics_collector):
        """Test request metrics recording."""
        metrics_collector.record_request("GET", "/health")
        metrics_collector.record_request("POST", "/insert")
        metrics_collector.record_request("GET", "/health")
        
        metrics = metrics_collector.get_metrics()
        
        assert metrics['total_requests'] == 3
        assert metrics['requests_by_method']['GET'] == 2
        assert metrics['requests_by_method']['POST'] == 1
        assert metrics['requests_by_endpoint']['/health'] == 2
        assert metrics['requests_by_endpoint']['/insert'] == 1
    
    def test_record_response_metrics(self, metrics_collector):
        """Test response metrics recording."""
        metrics_collector.record_response("GET", "/health", 200, 0.1)
        metrics_collector.record_response("POST", "/insert", 201, 0.5)
        metrics_collector.record_response("GET", "/query", 500, 1.0)
        
        metrics = metrics_collector.get_metrics()
        
        assert metrics['responses_by_status'][200] == 1
        assert metrics['responses_by_status'][201] == 1
        assert metrics['responses_by_status'][500] == 1
        assert len(metrics['response_times']) == 3
        assert metrics['average_response_time'] == 0.53  # (0.1 + 0.5 + 1.0) / 3
    
    def test_record_error_metrics(self, metrics_collector):
        """Test error metrics recording."""
        metrics_collector.record_error("POST", "/insert", "Connection timeout")
        metrics_collector.record_error("GET", "/query", "Invalid query")
        metrics_collector.record_error("POST", "/insert", "Rate limit exceeded")
        
        metrics = metrics_collector.get_metrics()
        
        assert metrics['total_errors'] == 3
        assert metrics['errors_by_endpoint']['/insert'] == 2
        assert metrics['errors_by_endpoint']['/query'] == 1
        assert 'Connection timeout' in metrics['error_types']
        assert 'Invalid query' in metrics['error_types']
    
    @pytest.mark.asyncio
    async def test_metrics_collection_lifecycle(self, metrics_collector):
        """Test metrics collection start/stop lifecycle."""
        # Start collection
        await metrics_collector.start_collection()
        assert metrics_collector.is_collecting is True
        
        # Let it run briefly
        await asyncio.sleep(0.1)
        
        # Stop collection
        await metrics_collector.stop_collection()
        assert metrics_collector.is_collecting is False
    
    def test_metrics_aggregation(self, metrics_collector):
        """Test metrics aggregation and calculation."""
        # Record various metrics
        for i in range(10):
            metrics_collector.record_request("GET", "/test")
            metrics_collector.record_response("GET", "/test", 200, 0.1 * i)
        
        metrics = metrics_collector.get_metrics()
        
        assert metrics['total_requests'] == 10
        assert metrics['total_responses'] == 10
        assert metrics['min_response_time'] == 0.0
        assert metrics['max_response_time'] == 0.9
        assert metrics['average_response_time'] == 0.45


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create a CircuitBreaker instance."""
        return CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=10,
            expected_exception=Exception
        )
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_initialization(self, circuit_breaker):
        """Test CircuitBreaker initialization."""
        assert circuit_breaker.failure_threshold == 3
        assert circuit_breaker.recovery_timeout == 10
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.state == "closed"
        assert circuit_breaker.last_failure_time is None
    
    @pytest.mark.asyncio
    async def test_successful_calls(self, circuit_breaker):
        """Test successful function calls through circuit breaker."""
        async def successful_func():
            return "success"
        
        result = await circuit_breaker.call(successful_func)
        
        assert result == "success"
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.state == "closed"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self, circuit_breaker):
        """Test that circuit breaker opens after threshold failures."""
        async def failing_func():
            raise Exception("Test failure")
        
        # Fail 3 times (threshold)
        for i in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_func)
        
        assert circuit_breaker.failure_count == 3
        assert circuit_breaker.state == "open"
        assert circuit_breaker.last_failure_time is not None
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_when_open(self, circuit_breaker):
        """Test that circuit breaker blocks calls when open."""
        async def failing_func():
            raise Exception("Test failure")
        
        # Trigger circuit breaker to open
        for i in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_func)
        
        # Now it should block calls
        with pytest.raises(Exception, match="Circuit breaker is open"):
            await circuit_breaker.call(failing_func)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self, circuit_breaker):
        """Test circuit breaker recovery process."""
        async def failing_func():
            raise Exception("Test failure")
        
        async def successful_func():
            return "success"
        
        # Open the circuit breaker
        for i in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_func)
        
        assert circuit_breaker.state == "open"
        
        # Mock time passage to allow recovery attempt
        with patch('time.time', return_value=time.time() + 15):
            # Should enter half-open state and allow one call
            result = await circuit_breaker.call(successful_func)
            assert result == "success"
            assert circuit_breaker.state == "closed"
            assert circuit_breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_during_half_open(self, circuit_breaker):
        """Test circuit breaker behavior when call fails during half-open state."""
        async def failing_func():
            raise Exception("Test failure")
        
        # Open the circuit breaker
        for i in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_func)
        
        # Mock time passage to allow recovery attempt
        with patch('time.time', return_value=time.time() + 15):
            # Fail during half-open state
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_func)
            
            # Should go back to open state
            assert circuit_breaker.state == "open"
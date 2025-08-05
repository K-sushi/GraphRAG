"""
Pytest configuration and shared fixtures for LightRAG server tests.
"""

import asyncio
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
from fastapi.testclient import TestClient

# Set test environment variables before importing app components
os.environ["ENVIRONMENT"] = "testing"
os.environ["REQUIRE_AUTH"] = "false"
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["LIGHTRAG_API_KEYS"] = "test-api-key-1,test-api-key-2"

from app import app, app_state
from config import Config
from utils import TokenTracker, HealthChecker, MetricsCollector, CircuitBreaker


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def test_config(temp_dir):
    """Create a test configuration."""
    os.environ["WORKING_DIR"] = str(temp_dir / "lightrag_test")
    config = Config()
    return config


@pytest.fixture
def mock_lightrag():
    """Create a mock LightRAG instance."""
    mock = AsyncMock()
    mock.ainsert = AsyncMock(return_value=None)
    mock.aquery = AsyncMock(return_value="Test response from knowledge graph")
    return mock


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def authenticated_client():
    """Create a test client with authentication headers."""
    with TestClient(app) as client:
        client.headers.update({"Authorization": "Bearer test-api-key-1"})
        yield client


@pytest.fixture
def token_tracker():
    """Create a TokenTracker instance for testing."""
    return TokenTracker(max_tokens_per_minute=1000)


@pytest.fixture
def health_checker():
    """Create a HealthChecker instance for testing."""
    return HealthChecker()


@pytest.fixture
def metrics_collector():
    """Create a MetricsCollector instance for testing."""
    return MetricsCollector()


@pytest.fixture
def circuit_breaker():
    """Create a CircuitBreaker instance for testing."""
    return CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=10,
        expected_exception=Exception
    )


@pytest.fixture
async def app_with_mocks(test_config, mock_lightrag, token_tracker, health_checker, metrics_collector, circuit_breaker):
    """Set up app state with mocked dependencies."""
    # Store original state
    original_state = {
        'config': app_state.config,
        'lightrag': app_state.lightrag,
        'token_tracker': app_state.token_tracker,
        'health_checker': app_state.health_checker,
        'metrics_collector': app_state.metrics_collector,
        'circuit_breaker': app_state.circuit_breaker,
    }
    
    # Set mock state
    app_state.config = test_config
    app_state.lightrag = mock_lightrag
    app_state.token_tracker = token_tracker
    app_state.health_checker = health_checker
    app_state.metrics_collector = metrics_collector
    app_state.circuit_breaker = circuit_breaker
    
    yield app_state
    
    # Restore original state
    for key, value in original_state.items():
        setattr(app_state, key, value)


@pytest.fixture
def sample_insert_data():
    """Sample data for insert requests."""
    return {
        "valid_insert": {
            "text": "This is a sample document about artificial intelligence and machine learning. AI is transforming various industries through automation and intelligent decision-making systems."
        },
        "short_insert": {
            "text": "Short text"
        },
        "empty_insert": {
            "text": ""
        },
        "large_insert": {
            "text": "A" * 10000  # Large text for testing limits
        }
    }


@pytest.fixture
def sample_query_data():
    """Sample data for query requests."""
    return {
        "valid_query": {
            "query": "What is artificial intelligence?",
            "mode": "hybrid"
        },
        "local_query": {
            "query": "Tell me about machine learning applications",
            "mode": "local"
        },
        "global_query": {
            "query": "How does AI impact society?",
            "mode": "global"
        },
        "empty_query": {
            "query": "",
            "mode": "hybrid"
        },
        "long_query": {
            "query": "What are the implications of artificial intelligence " * 100,
            "mode": "hybrid"
        }
    }


@pytest.fixture
def auth_headers():
    """Authentication headers for testing."""
    return {
        "valid": {"Authorization": "Bearer test-api-key-1"},
        "invalid": {"Authorization": "Bearer invalid-key"},
        "malformed": {"Authorization": "InvalidFormat"},
        "missing": {}
    }


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": "This is a mock response from OpenAI API"
                }
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }


# Async test helpers
@pytest.fixture
def async_mock():
    """Helper to create async mocks."""
    def _async_mock(*args, **kwargs):
        m = MagicMock(*args, **kwargs)
        
        async def async_func(*args, **kwargs):
            return m(*args, **kwargs)
        
        m.side_effect = async_func
        return m
    
    return _async_mock
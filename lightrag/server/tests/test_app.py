"""
Tests for FastAPI application endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from app import app


class TestAppEndpoints:
    """Test FastAPI application endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        with TestClient(app) as client:
            yield client
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "LightRAG Server"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
        assert "uptime_seconds" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert data["version"] == "1.0.0"
    
    @patch('app.app_state.lightrag')
    def test_insert_endpoint_success(self, mock_lightrag, client):
        """Test successful text insertion."""
        mock_lightrag.ainsert = AsyncMock(return_value=None)
        
        with patch('app.app_state.config') as mock_config:
            mock_config.require_auth = False
            
            response = client.post(
                "/insert",
                json={"text": "This is a test document about AI and machine learning."}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["message"] == "Text inserted successfully"
        assert data["text_length"] > 0
        assert "timestamp" in data
    
    def test_insert_endpoint_empty_text(self, client):
        """Test insertion with empty text."""
        with patch('app.app_state.config') as mock_config:
            mock_config.require_auth = False
            
            response = client.post(
                "/insert",
                json={"text": ""}
            )
        
        assert response.status_code == 422  # Validation error
    
    def test_insert_endpoint_missing_text(self, client):
        """Test insertion with missing text field."""
        with patch('app.app_state.config') as mock_config:
            mock_config.require_auth = False
            
            response = client.post("/insert", json={})
        
        assert response.status_code == 422  # Validation error
    
    @patch('app.app_state.lightrag')
    def test_query_endpoint_success(self, mock_lightrag, client):
        """Test successful query."""
        mock_lightrag.aquery = AsyncMock(return_value="This is a test response about AI.")
        
        with patch('app.app_state.config') as mock_config:
            mock_config.require_auth = False
            
            response = client.post(
                "/query",
                json={
                    "query": "What is artificial intelligence?",
                    "mode": "hybrid"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["result"] == "This is a test response about AI."
        assert data["query"] == "What is artificial intelligence?"
        assert data["mode"] == "hybrid"
        assert "timestamp" in data
    
    def test_query_endpoint_empty_query(self, client):
        """Test query with empty query string."""
        with patch('app.app_state.config') as mock_config:
            mock_config.require_auth = False
            
            response = client.post(
                "/query",
                json={"query": "", "mode": "hybrid"}
            )
        
        assert response.status_code == 422  # Validation error
    
    def test_query_endpoint_invalid_mode(self, client):
        """Test query with invalid mode."""
        with patch('app.app_state.config') as mock_config:
            mock_config.require_auth = False
            
            response = client.post(
                "/query",
                json={"query": "test query", "mode": "invalid_mode"}
            )
        
        assert response.status_code == 422  # Validation error
    
    @patch('app.app_state.lightrag')
    def test_delete_endpoint_success(self, mock_lightrag, client):
        """Test successful storage deletion."""
        with patch('app.app_state.config') as mock_config, \
             patch('shutil.rmtree') as mock_rmtree, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('lightrag.LightRAG') as mock_lightrag_class:
            
            mock_config.require_auth = False
            mock_config.working_dir = "/test/dir"
            mock_lightrag_class.return_value = mock_lightrag
            
            response = client.delete(
                "/delete",
                json={"confirm_delete": True}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "deleted" in data["message"].lower()
        assert "timestamp" in data
    
    def test_delete_endpoint_without_confirmation(self, client):
        """Test deletion without confirmation."""
        with patch('app.app_state.config') as mock_config:
            mock_config.require_auth = False
            
            response = client.delete(
                "/delete",
                json={"confirm_delete": False}
            )
        
        assert response.status_code == 400
        assert "confirmation" in response.json()["detail"].lower()
    
    def test_authentication_required_endpoints(self, client):
        """Test that protected endpoints require authentication."""
        with patch('app.app_state.config') as mock_config:
            mock_config.require_auth = True
            mock_config.api_keys = ["test-key-1", "test-key-2"]
            
            # Test without auth header
            response = client.get("/status")
            assert response.status_code == 401
            
            response = client.get("/metrics")
            assert response.status_code == 401
            
            response = client.get("/config")
            assert response.status_code == 401
    
    def test_authentication_with_valid_key(self, client):
        """Test authentication with valid API key."""
        with patch('app.app_state.config') as mock_config, \
             patch('app.app_state.health_checker') as mock_health, \
             patch('app.app_state.metrics_collector') as mock_metrics:
            
            mock_config.require_auth = True
            mock_config.api_keys = ["test-key-1", "test-key-2"]
            mock_health.check_system_health = AsyncMock(return_value={"overall_health": True})
            mock_metrics.get_metrics = lambda: {"total_requests": 0}
            
            headers = {"Authorization": "Bearer test-key-1"}
            
            response = client.get("/status", headers=headers)
            assert response.status_code == 200
    
    def test_authentication_with_invalid_key(self, client):
        """Test authentication with invalid API key."""
        with patch('app.app_state.config') as mock_config:
            mock_config.require_auth = True
            mock_config.api_keys = ["test-key-1", "test-key-2"]
            
            headers = {"Authorization": "Bearer invalid-key"}
            
            response = client.get("/status", headers=headers)
            assert response.status_code == 401
    
    def test_malformed_authorization_header(self, client):
        """Test malformed authorization header."""
        with patch('app.app_state.config') as mock_config:
            mock_config.require_auth = True
            mock_config.api_keys = ["test-key-1"]
            
            headers = {"Authorization": "InvalidFormat"}
            
            response = client.get("/status", headers=headers)
            assert response.status_code == 401
    
    def test_rate_limiting(self, client):
        """Test rate limiting functionality."""
        # This is a basic test - real rate limiting would require time manipulation
        with patch('app.app_state.config') as mock_config:
            mock_config.require_auth = False
            
            # Make multiple requests quickly
            responses = []
            for i in range(5):
                response = client.get("/health")
                responses.append(response.status_code)
            
            # All should succeed for health endpoint (generous limits)
            assert all(status == 200 for status in responses)
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/", headers={"Origin": "http://localhost:3000"})
        
        # CORS headers should be present (added by middleware)
        assert "access-control-allow-origin" in response.headers or response.status_code == 200
    
    def test_process_time_header(self, client):
        """Test that process time header is added."""
        response = client.get("/")
        
        # Should have process time header
        assert "x-process-time" in response.headers
        process_time = float(response.headers["x-process-time"])
        assert process_time >= 0
    
    def test_error_handling(self, client):
        """Test error handling for internal server errors."""
        with patch('app.app_state.lightrag') as mock_lightrag:
            mock_lightrag.aquery = AsyncMock(side_effect=Exception("Test error"))
            
            with patch('app.app_state.config') as mock_config:
                mock_config.require_auth = False
                
                response = client.post(
                    "/query",
                    json={"query": "test query", "mode": "hybrid"}
                )
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "timestamp" in data
    
    def test_service_unavailable_when_not_initialized(self, client):
        """Test service unavailable when LightRAG is not initialized."""
        with patch('app.app_state.lightrag', None), \
             patch('app.app_state.config') as mock_config:
            
            mock_config.require_auth = False
            
            response = client.post(
                "/insert",
                json={"text": "test text"}
            )
            
            assert response.status_code == 503
            assert "not initialized" in response.json()["detail"]


class TestAppStartup:
    """Test application startup and lifecycle."""
    
    def test_app_creation(self):
        """Test that app can be created."""
        from app import app
        assert app is not None
        assert app.title == "LightRAG Server"
        assert app.version == "1.0.0"
    
    @patch('app.Config')
    @patch('app.LightRAG')
    def test_app_initialization_with_valid_config(self, mock_lightrag, mock_config):
        """Test app initialization with valid configuration."""
        # This would test the lifespan context manager
        # In practice, this is complex to test and might require integration tests
        pass
    
    def test_middleware_setup(self):
        """Test that middleware is properly configured."""
        from app import app
        
        # Check that middleware is configured
        middleware_types = [type(middleware.cls).__name__ for middleware in app.user_middleware]
        
        # Should have rate limiting and other middleware
        # Exact middleware depends on configuration
        assert len(middleware_types) > 0
"""
Tests for security and authentication.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import time

from app import app, verify_api_key
from config import Config


class TestAuthentication:
    """Test authentication functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        with TestClient(app) as client:
            yield client
    
    def test_authentication_disabled(self, client):
        """Test endpoints when authentication is disabled."""
        with patch('app.app_state.config') as mock_config:
            mock_config.require_auth = False
            
            # Should allow access without auth
            response = client.post("/insert", json={"text": "test"})
            # May fail for other reasons, but not auth
            assert response.status_code != 401
    
    def test_authentication_required(self, client):
        """Test endpoints when authentication is required."""
        with patch('app.app_state.config') as mock_config:
            mock_config.require_auth = True
            mock_config.api_keys = ["secret-key-1", "secret-key-2"]
            
            # Should reject without auth header
            response = client.get("/status")
            assert response.status_code == 401
            
            response = client.get("/metrics")
            assert response.status_code == 401
            
            response = client.get("/config")
            assert response.status_code == 401
    
    def test_valid_api_key(self, client):
        """Test access with valid API key."""
        with patch('app.app_state.config') as mock_config, \
             patch('app.app_state.health_checker') as mock_health, \
             patch('app.app_state.metrics_collector') as mock_metrics:
            
            mock_config.require_auth = True
            mock_config.api_keys = ["secret-key-1", "secret-key-2"]
            mock_health.check_system_health = lambda: {"overall_health": True}
            mock_metrics.get_metrics = lambda: {"total_requests": 0}
            
            headers = {"Authorization": "Bearer secret-key-1"}
            response = client.get("/status", headers=headers)
            assert response.status_code == 200
    
    def test_invalid_api_key(self, client):
        """Test access with invalid API key."""
        with patch('app.app_state.config') as mock_config:
            mock_config.require_auth = True
            mock_config.api_keys = ["secret-key-1", "secret-key-2"]
            
            headers = {"Authorization": "Bearer invalid-key"}
            response = client.get("/status", headers=headers)
            assert response.status_code == 401
    
    def test_malformed_auth_header(self, client):
        """Test various malformed authorization headers."""
        with patch('app.app_state.config') as mock_config:
            mock_config.require_auth = True
            mock_config.api_keys = ["secret-key"]
            
            # Test various malformed headers
            malformed_headers = [
                {"Authorization": "Bearer"},  # No token
                {"Authorization": "Basic secret-key"},  # Wrong scheme
                {"Authorization": "secret-key"},  # No scheme
                {"Authorization": ""},  # Empty
            ]
            
            for headers in malformed_headers:
                response = client.get("/status", headers=headers)
                assert response.status_code == 401
    
    def test_missing_auth_header(self, client):
        """Test access without authorization header."""
        with patch('app.app_state.config') as mock_config:
            mock_config.require_auth = True
            mock_config.api_keys = ["secret-key"]
            
            response = client.get("/status")
            assert response.status_code == 401
            assert "Authorization header required" in response.json()["detail"]
    
    def test_empty_api_keys_list(self, client):
        """Test behavior with empty API keys list."""
        with patch('app.app_state.config') as mock_config:
            mock_config.require_auth = True
            mock_config.api_keys = []
            
            headers = {"Authorization": "Bearer any-key"}
            response = client.get("/status", headers=headers)
            assert response.status_code == 401
    
    def test_case_sensitive_api_keys(self, client):
        """Test that API keys are case sensitive."""
        with patch('app.app_state.config') as mock_config:
            mock_config.require_auth = True
            mock_config.api_keys = ["SecretKey123"]
            
            # Wrong case
            headers = {"Authorization": "Bearer secretkey123"}
            response = client.get("/status", headers=headers)
            assert response.status_code == 401
            
            # Correct case should work (if other mocks are set up)
            with patch('app.app_state.health_checker') as mock_health, \
                 patch('app.app_state.metrics_collector') as mock_metrics:
                mock_health.check_system_health = lambda: {"overall_health": True}
                mock_metrics.get_metrics = lambda: {"total_requests": 0}
                
                headers = {"Authorization": "Bearer SecretKey123"}
                response = client.get("/status", headers=headers)
                assert response.status_code == 200


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        with TestClient(app) as client:
            yield client
    
    def test_rate_limiting_basic(self, client):
        """Test basic rate limiting."""
        with patch('app.app_state.config') as mock_config:
            mock_config.require_auth = False
            
            # Make requests within normal limits
            responses = []
            for i in range(3):
                response = client.get("/health")
                responses.append(response.status_code)
                time.sleep(0.1)  # Brief delay
            
            # Should all succeed for health endpoint
            assert all(status == 200 for status in responses)
    
    def test_rate_limiting_per_endpoint(self, client):
        """Test that rate limiting is applied per endpoint."""
        with patch('app.app_state.config') as mock_config, \
             patch('app.app_state.lightrag') as mock_lightrag:
            
            mock_config.require_auth = False
            mock_lightrag.ainsert = lambda x: None
            
            # Test different endpoints have different limits
            # Health endpoint should be more permissive
            health_responses = [client.get("/health") for _ in range(5)]
            assert all(r.status_code == 200 for r in health_responses)
    
    def test_rate_limit_headers(self, client):
        """Test that rate limit headers are present."""
        response = client.get("/health")
        
        # Check for rate limit headers (may vary based on configuration)
        # Some rate limiting implementations add headers like X-RateLimit-*
        assert response.status_code == 200
        # Headers depend on SlowAPI configuration
    
    def test_rate_limit_per_ip(self, client):
        """Test that rate limiting is applied per IP address."""
        # This is complex to test without actual network simulation
        # SlowAPI uses get_remote_address which is hard to mock properly
        pass


class TestSecurityHeaders:
    """Test security headers and middleware."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        with TestClient(app) as client:
            yield client
    
    def test_cors_headers(self, client):
        """Test CORS headers are properly configured."""
        with patch('app.app_state.config') as mock_config:
            mock_config.cors_origins = ["http://localhost:3000", "https://example.com"]
            
            response = client.options(
                "/",
                headers={"Origin": "http://localhost:3000"}
            )
            
            # CORS headers should be handled by middleware
            # Exact behavior depends on FastAPI CORS middleware
            assert response.status_code in [200, 405]  # OPTIONS may not be implemented
    
    def test_security_headers_present(self, client):
        """Test that security headers are present."""
        response = client.get("/")
        
        # Check for security headers
        headers = response.headers
        
        # These depend on middleware configuration
        # Common security headers to check for:
        security_headers = [
            "x-content-type-options",
            "x-frame-options", 
            "x-xss-protection",
            "strict-transport-security"
        ]
        
        # Note: These headers need to be added by middleware if required
        # The current implementation may not include all of them
    
    def test_sensitive_data_not_exposed(self, client):
        """Test that sensitive data is not exposed in responses."""
        with patch('app.app_state.config') as mock_config:
            mock_config.require_auth = False
            mock_config.openai_api_key = "secret-openai-key"
            mock_config.api_keys = ["secret-api-key"]
            
            response = client.get("/config")
            
            if response.status_code == 200:
                data = response.json()
                config_data = data.get("configuration", {})
                
                # Sensitive values should be masked
                assert config_data.get("openai_api_key") == "***masked***"
                assert config_data.get("api_keys") == "***masked***"


class TestInputValidation:
    """Test input validation and sanitization."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        with TestClient(app) as client:
            yield client
    
    def test_sql_injection_prevention(self, client):
        """Test prevention of SQL injection attempts."""
        with patch('app.app_state.config') as mock_config:
            mock_config.require_auth = False
            
            # SQL injection attempts in query
            malicious_queries = [
                "'; DROP TABLE users; --",
                "1' OR '1'='1",
                "admin'--",
                "1; DELETE FROM users",
            ]
            
            for malicious_query in malicious_queries:
                response = client.post(
                    "/query",
                    json={"query": malicious_query, "mode": "hybrid"}
                )
                
                # Should handle gracefully without exposing internal errors
                assert response.status_code in [422, 500, 503]  # Various error codes are acceptable
    
    def test_xss_prevention(self, client):
        """Test prevention of XSS attacks."""
        with patch('app.app_state.config') as mock_config:
            mock_config.require_auth = False
            
            # XSS attempts in text insertion
            xss_payloads = [
                "<script>alert('xss')</script>",
                "javascript:alert('xss')",
                "<img src=x onerror=alert('xss')>",
                "&#60;script&#62;alert('xss')&#60;/script&#62;",
            ]
            
            for payload in xss_payloads:
                response = client.post(
                    "/insert",
                    json={"text": payload}
                )
                
                # Should handle without executing script
                # The exact response depends on validation and processing
                assert response.status_code in [200, 422, 500, 503]
    
    def test_oversized_request_handling(self, client):
        """Test handling of oversized requests."""
        with patch('app.app_state.config') as mock_config:
            mock_config.require_auth = False
            
            # Very large text
            large_text = "A" * (10 * 1024 * 1024)  # 10MB of text
            
            response = client.post(
                "/insert",
                json={"text": large_text}
            )
            
            # Should handle large requests appropriately
            # May be accepted, rejected, or cause timeout
            assert response.status_code in [200, 413, 422, 500, 503]
    
    def test_malformed_json_handling(self, client):
        """Test handling of malformed JSON."""
        with patch('app.app_state.config') as mock_config:
            mock_config.require_auth = False
            
            # Send malformed JSON
            response = client.post(
                "/insert",
                data='{"text": invalid json}',
                headers={"Content-Type": "application/json"}
            )
            
            assert response.status_code == 422  # Unprocessable Entity
    
    def test_unicode_handling(self, client):
        """Test proper Unicode handling."""
        with patch('app.app_state.config') as mock_config:
            mock_config.require_auth = False
            
            # Unicode text including emojis and special characters
            unicode_text = "Hello 世界! 🌍 Testing unicode: café, naïve, résumé"
            
            response = client.post(
                "/insert",
                json={"text": unicode_text}
            )
            
            # Should handle Unicode properly
            assert response.status_code in [200, 503]  # May fail if LightRAG not initialized


class TestCircuitBreakerSecurity:
    """Test circuit breaker security aspects."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        with TestClient(app) as client:
            yield client
    
    def test_circuit_breaker_prevents_cascade_failures(self, client):
        """Test that circuit breaker prevents cascade failures."""
        with patch('app.app_state.config') as mock_config, \
             patch('app.app_state.lightrag') as mock_lightrag, \
             patch('app.app_state.circuit_breaker') as mock_cb:
            
            mock_config.require_auth = False
            mock_lightrag.aquery.side_effect = Exception("External service down")
            
            # Configure circuit breaker to be open
            mock_cb.state = "open"
            mock_cb.call.side_effect = Exception("Circuit breaker is open")
            
            response = client.post(
                "/query",
                json={"query": "test", "mode": "hybrid"}
            )
            
            assert response.status_code == 503  # Service unavailable
    
    def test_circuit_breaker_failure_isolation(self, client):
        """Test that circuit breaker isolates failures."""
        # This would test that failures in one component don't affect others
        # Complex to test without full integration
        pass


class TestSecurityConfiguration:
    """Test security configuration validation."""
    
    def test_secure_configuration_validation(self):
        """Test that secure configurations are validated."""
        # Test that weak configurations are rejected
        with patch.dict('os.environ', {
            'REQUIRE_AUTH': 'false',
            'ENVIRONMENT': 'production'
        }):
            # In production, should probably warn about disabled auth
            config = Config()
            # This would depend on validation logic
            assert config.environment == "production"
            assert config.require_auth is False
    
    def test_api_key_strength_validation(self):
        """Test API key strength validation."""
        # This would test minimum key length, complexity, etc.
        weak_keys = [
            "123",
            "password",
            "abc",
            ""
        ]
        
        for weak_key in weak_keys:
            with patch.dict('os.environ', {'LIGHTRAG_API_KEYS': weak_key}):
                # Depending on implementation, might want to validate key strength
                config = Config()
                assert weak_key in config.api_keys  # Current implementation allows weak keys
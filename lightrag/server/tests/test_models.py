"""
Tests for Pydantic models.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from models import (
    InsertRequest, InsertResponse, QueryRequest, QueryResponse,
    DeleteRequest, DeleteResponse, HealthResponse, MetricsResponse,
    SystemStatusResponse, ConfigResponse, ErrorResponse,
    QueryMode
)


class TestInsertRequest:
    """Test InsertRequest model."""
    
    def test_valid_insert_request(self):
        """Test valid insert request."""
        data = {
            "text": "This is a sample document about artificial intelligence."
        }
        request = InsertRequest(**data)
        
        assert request.text == data["text"]
        assert len(request.text) > 10  # Minimum length validation
    
    def test_insert_request_empty_text(self):
        """Test insert request with empty text."""
        with pytest.raises(ValidationError) as exc_info:
            InsertRequest(text="")
        
        assert "ensure this value has at least" in str(exc_info.value)
    
    def test_insert_request_whitespace_only(self):
        """Test insert request with whitespace-only text."""
        with pytest.raises(ValidationError):
            InsertRequest(text="   ")
    
    def test_insert_request_very_long_text(self):
        """Test insert request with very long text."""
        long_text = "A" * 100000  # 100k characters
        request = InsertRequest(text=long_text)
        
        assert len(request.text) == 100000
    
    def test_insert_request_missing_text(self):
        """Test insert request without text field."""
        with pytest.raises(ValidationError) as exc_info:
            InsertRequest()
        
        assert "field required" in str(exc_info.value)


class TestQueryRequest:
    """Test QueryRequest model."""
    
    def test_valid_query_request(self):
        """Test valid query request."""
        data = {
            "query": "What is artificial intelligence?",
            "mode": "hybrid"
        }
        request = QueryRequest(**data)
        
        assert request.query == data["query"]
        assert request.mode == QueryMode.HYBRID
    
    def test_query_request_with_all_modes(self):
        """Test query request with all valid modes."""
        valid_modes = ["naive", "local", "global", "hybrid"]
        
        for mode in valid_modes:
            request = QueryRequest(query="test query", mode=mode)
            assert request.mode.value == mode
    
    def test_query_request_invalid_mode(self):
        """Test query request with invalid mode."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query="test query", mode="invalid_mode")
        
        assert "value is not a valid enumeration member" in str(exc_info.value)
    
    def test_query_request_empty_query(self):
        """Test query request with empty query."""
        with pytest.raises(ValidationError):
            QueryRequest(query="", mode="hybrid")
    
    def test_query_request_default_mode(self):
        """Test query request with default mode."""
        request = QueryRequest(query="test query")
        assert request.mode == QueryMode.HYBRID  # Default mode
    
    def test_query_request_very_long_query(self):
        """Test query request with very long query."""
        long_query = "What is " * 1000 + "artificial intelligence?"
        request = QueryRequest(query=long_query, mode="hybrid")
        
        assert len(request.query) > 1000


class TestDeleteRequest:
    """Test DeleteRequest model."""
    
    def test_valid_delete_request(self):
        """Test valid delete request."""
        request = DeleteRequest(confirm_delete=True)
        assert request.confirm_delete is True
    
    def test_delete_request_false_confirmation(self):
        """Test delete request with false confirmation."""
        request = DeleteRequest(confirm_delete=False)
        assert request.confirm_delete is False
    
    def test_delete_request_default_confirmation(self):
        """Test delete request with default confirmation."""
        request = DeleteRequest()
        assert request.confirm_delete is False  # Default should be False


class TestInsertResponse:
    """Test InsertResponse model."""
    
    def test_valid_insert_response(self):
        """Test valid insert response."""
        data = {
            "status": "success",
            "message": "Text inserted successfully",
            "text_length": 100,
            "timestamp": 1234567890.0
        }
        response = InsertResponse(**data)
        
        assert response.status == "success"
        assert response.message == "Text inserted successfully"
        assert response.text_length == 100
        assert response.timestamp == 1234567890.0
    
    def test_insert_response_negative_text_length(self):
        """Test insert response with negative text length."""
        with pytest.raises(ValidationError):
            InsertResponse(
                status="success",
                message="Test",
                text_length=-1,
                timestamp=1234567890.0
            )
    
    def test_insert_response_required_fields(self):
        """Test insert response with missing required fields."""
        with pytest.raises(ValidationError):
            InsertResponse()


class TestQueryResponse:
    """Test QueryResponse model."""
    
    def test_valid_query_response(self):
        """Test valid query response."""
        data = {
            "result": "This is a response about AI.",
            "query": "What is AI?",
            "mode": "hybrid",
            "timestamp": 1234567890.0
        }
        response = QueryResponse(**data)
        
        assert response.result == data["result"]
        assert response.query == data["query"]
        assert response.mode == data["mode"]
        assert response.timestamp == data["timestamp"]
    
    def test_query_response_empty_result(self):
        """Test query response with empty result."""
        response = QueryResponse(
            result="",
            query="test query",
            mode="hybrid",
            timestamp=1234567890.0
        )
        
        assert response.result == ""


class TestHealthResponse:
    """Test HealthResponse model."""
    
    def test_valid_health_response(self):
        """Test valid health response."""
        data = {
            "status": "healthy",
            "timestamp": 1234567890.0,
            "version": "1.0.0",
            "details": {
                "system_resources": {"status": "healthy"},
                "database": {"status": "healthy"}
            }
        }
        response = HealthResponse(**data)
        
        assert response.status == "healthy"
        assert response.version == "1.0.0"
        assert isinstance(response.details, dict)
    
    def test_health_response_unhealthy_status(self):
        """Test health response with unhealthy status."""
        response = HealthResponse(
            status="unhealthy",
            timestamp=1234567890.0,
            version="1.0.0",
            details={"error": "System overloaded"}
        )
        
        assert response.status == "unhealthy"
        assert "error" in response.details


class TestErrorResponse:
    """Test ErrorResponse model."""
    
    def test_valid_error_response(self):
        """Test valid error response."""
        data = {
            "error": "Invalid request",
            "status_code": 400,
            "timestamp": 1234567890.0
        }
        response = ErrorResponse(**data)
        
        assert response.error == "Invalid request"
        assert response.status_code == 400
        assert response.timestamp == 1234567890.0
    
    def test_error_response_with_details(self):
        """Test error response with additional details."""
        response = ErrorResponse(
            error="Validation failed",
            status_code=422,
            timestamp=1234567890.0,
            details={
                "field": "query",
                "message": "Query too short"
            }
        )
        
        assert response.details["field"] == "query"
        assert response.details["message"] == "Query too short"
    
    def test_error_response_negative_status_code(self):
        """Test error response with negative status code."""
        with pytest.raises(ValidationError):
            ErrorResponse(
                error="Test error",
                status_code=-1,
                timestamp=1234567890.0
            )
    
    def test_error_response_invalid_status_code(self):
        """Test error response with invalid status code."""
        with pytest.raises(ValidationError):
            ErrorResponse(
                error="Test error",
                status_code=999,  # Invalid HTTP status code
                timestamp=1234567890.0
            )


class TestSystemStatusResponse:
    """Test SystemStatusResponse model."""
    
    def test_valid_system_status_response(self):
        """Test valid system status response."""
        data = {
            "status": "operational",
            "uptime_seconds": 3600.5,
            "environment": "production",
            "lightrag_initialized": True,
            "health_check": {"overall_health": True},
            "metrics": {"total_requests": 100},
            "circuit_breaker": {"state": "closed", "failure_count": 0}
        }
        response = SystemStatusResponse(**data)
        
        assert response.status == "operational"
        assert response.uptime_seconds == 3600.5
        assert response.environment == "production"
        assert response.lightrag_initialized is True
    
    def test_system_status_response_negative_uptime(self):
        """Test system status response with negative uptime."""
        with pytest.raises(ValidationError):
            SystemStatusResponse(
                status="operational",
                uptime_seconds=-1.0,
                environment="test",
                lightrag_initialized=True,
                health_check={},
                metrics={},
                circuit_breaker={}
            )


class TestConfigResponse:
    """Test ConfigResponse model."""
    
    def test_valid_config_response(self):
        """Test valid config response."""
        data = {
            "environment": "development",
            "configuration": {
                "host": "127.0.0.1",
                "port": 8000,
                "openai_api_key": "***masked***"
            }
        }
        response = ConfigResponse(**data)
        
        assert response.environment == "development"
        assert response.configuration["host"] == "127.0.0.1"
        assert response.configuration["openai_api_key"] == "***masked***"


class TestMetricsResponse:
    """Test MetricsResponse model."""
    
    def test_valid_metrics_response(self):
        """Test valid metrics response."""
        data = {
            "timestamp": 1234567890.0,
            "total_requests": 100,
            "total_responses": 95,
            "total_errors": 5,
            "average_response_time": 0.25,
            "requests_by_method": {"GET": 60, "POST": 40},
            "responses_by_status": {200: 85, 400: 5, 500: 5}
        }
        response = MetricsResponse(**data)
        
        assert response.total_requests == 100
        assert response.total_responses == 95
        assert response.total_errors == 5
        assert response.average_response_time == 0.25
    
    def test_metrics_response_negative_values(self):
        """Test metrics response with negative values."""
        with pytest.raises(ValidationError):
            MetricsResponse(
                timestamp=1234567890.0,
                total_requests=-1,  # Should be >= 0
                total_responses=0,
                total_errors=0,
                average_response_time=0.0,
                requests_by_method={},
                responses_by_status={}
            )


class TestModelValidation:
    """Test general model validation features."""
    
    def test_timestamp_validation(self):
        """Test timestamp validation across models."""
        # Test valid timestamp
        response = ErrorResponse(
            error="test",
            status_code=400,
            timestamp=1234567890.0
        )
        assert response.timestamp == 1234567890.0
        
        # Test negative timestamp (should be allowed for historical data)
        response = ErrorResponse(
            error="test",
            status_code=400,
            timestamp=0.0
        )
        assert response.timestamp == 0.0
    
    def test_string_length_validation(self):
        """Test string length validation."""
        # Test minimum length for queries
        with pytest.raises(ValidationError):
            QueryRequest(query="a")  # Too short
        
        # Test very long strings are accepted
        long_text = "a" * 10000
        request = InsertRequest(text=long_text)
        assert len(request.text) == 10000
    
    def test_enum_validation(self):
        """Test enum validation."""
        # Test valid enum values
        for mode in ["naive", "local", "global", "hybrid"]:
            request = QueryRequest(query="test", mode=mode)
            assert request.mode.value == mode
        
        # Test invalid enum value
        with pytest.raises(ValidationError):
            QueryRequest(query="test", mode="invalid")
    
    def test_model_serialization(self):
        """Test model serialization to dict."""
        request = QueryRequest(query="test query", mode="hybrid")
        data = request.dict()
        
        assert isinstance(data, dict)
        assert data["query"] == "test query"
        assert data["mode"] == "hybrid"
    
    def test_model_json_serialization(self):
        """Test model JSON serialization."""
        request = QueryRequest(query="test query", mode="hybrid")
        json_str = request.json()
        
        assert isinstance(json_str, str)
        assert "test query" in json_str
        assert "hybrid" in json_str
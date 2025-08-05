"""
Tests for configuration management.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from config import Config, Environment, LLMProvider, StorageType


class TestConfig:
    """Test configuration loading and validation."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = Config()
        
        assert config.environment == Environment.DEVELOPMENT
        assert config.host == "127.0.0.1"
        assert config.port == 8000
        assert config.log_level == "INFO"
        assert config.require_auth is True
        assert config.llm_provider == LLMProvider.OPENAI
    
    def test_environment_variables_override(self):
        """Test that environment variables override defaults."""
        with patch.dict(os.environ, {
            'ENVIRONMENT': 'production',
            'HOST': '0.0.0.0',
            'PORT': '9000',
            'LOG_LEVEL': 'WARNING',
            'LLM_PROVIDER': 'gemini'
        }):
            config = Config()
            
            assert config.environment == Environment.PRODUCTION
            assert config.host == "0.0.0.0"
            assert config.port == 9000
            assert config.log_level == "WARNING"
            assert config.llm_provider == LLMProvider.GEMINI
    
    def test_api_keys_parsing(self):
        """Test API keys parsing from environment."""
        with patch.dict(os.environ, {
            'LIGHTRAG_API_KEYS': 'key1,key2,key3',
            'CORS_ORIGINS': 'http://localhost:3000,https://example.com'
        }):
            config = Config()
            
            assert config.api_keys == ['key1', 'key2', 'key3']
            assert config.cors_origins == ['http://localhost:3000', 'https://example.com']
    
    def test_working_directory_creation(self, temp_dir):
        """Test that working directory is created."""
        working_dir = temp_dir / "test_lightrag"
        
        with patch.dict(os.environ, {
            'WORKING_DIR': str(working_dir)
        }):
            config = Config()
            
            assert Path(config.working_dir).exists()
            assert Path(config.working_dir).is_dir()
    
    def test_invalid_environment_raises_error(self):
        """Test that invalid environment raises validation error."""
        with patch.dict(os.environ, {'ENVIRONMENT': 'invalid'}):
            with pytest.raises(ValueError):
                Config()
    
    def test_missing_required_api_key_raises_error(self):
        """Test that missing required API key raises error."""
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'openai',
            'OPENAI_API_KEY': ''
        }):
            with pytest.raises(ValueError, match="OpenAI API key is required"):
                Config()
    
    def test_config_validation_with_gemini(self):
        """Test configuration validation with Gemini provider."""
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'gemini',
            'GEMINI_API_KEY': 'test-gemini-key'
        }):
            config = Config()
            assert config.llm_provider == LLMProvider.GEMINI
            assert config.gemini_api_key == 'test-gemini-key'
    
    def test_rate_limiting_configuration(self):
        """Test rate limiting configuration."""
        with patch.dict(os.environ, {
            'MAX_TOKENS_PER_MINUTE': '5000',
            'RATE_LIMIT_PER_MINUTE': '120'
        }):
            config = Config()
            
            assert config.max_tokens_per_minute == 5000
            assert config.rate_limit_per_minute == 120
    
    def test_circuit_breaker_configuration(self):
        """Test circuit breaker configuration."""
        with patch.dict(os.environ, {
            'CIRCUIT_BREAKER_FAILURE_THRESHOLD': '10',
            'CIRCUIT_BREAKER_RECOVERY_TIMEOUT': '120'
        }):
            config = Config()
            
            assert config.circuit_breaker_failure_threshold == 10
            assert config.circuit_breaker_recovery_timeout == 120
    
    def test_storage_configuration(self):
        """Test storage configuration options."""
        with patch.dict(os.environ, {
            'KV_STORAGE_TYPE': 'json',
            'VECTOR_STORAGE_TYPE': 'nano',
            'GRAPH_STORAGE_TYPE': 'networkx'
        }):
            config = Config()
            
            assert config.kv_storage_type == StorageType.JSON
            assert config.vector_storage_type == "nano"
            assert config.graph_storage_type == "networkx"
    
    def test_production_configuration(self):
        """Test production-specific configuration."""
        with patch.dict(os.environ, {
            'ENVIRONMENT': 'production',
            'HOST': '0.0.0.0',
            'PORT': '8000',
            'LOG_LEVEL': 'WARNING',
            'REQUIRE_AUTH': 'true',
            'CORS_ORIGINS': 'https://myapp.com'
        }):
            config = Config()
            
            assert config.environment == Environment.PRODUCTION
            assert config.host == "0.0.0.0"
            assert config.log_level == "WARNING"
            assert config.require_auth is True
    
    @pytest.mark.parametrize("env_value,expected", [
        ("true", True),
        ("True", True),  
        ("TRUE", True),
        ("false", False),
        ("False", False),
        ("FALSE", False),
        ("", False),
    ])
    def test_boolean_environment_parsing(self, env_value, expected):
        """Test boolean environment variable parsing."""
        with patch.dict(os.environ, {'REQUIRE_AUTH': env_value}):
            config = Config()
            assert config.require_auth == expected
    
    def test_config_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = Config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'environment' in config_dict
        assert 'host' in config_dict
        assert 'port' in config_dict
        
        # Sensitive values should be masked
        assert config_dict.get('openai_api_key') == '***masked***'
        assert config_dict.get('lightrag_api_keys') == '***masked***'
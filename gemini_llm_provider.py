#!/usr/bin/env python3
"""
Gemini LLM Provider for Microsoft GraphRAG
SuperClaude Wave Orchestration Implementation
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for Gemini LLM"""
    api_key: str
    model: str = "gemini-1.5-pro-002"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 8192
    requests_per_minute: int = 10000
    tokens_per_minute: int = 150000
    max_retries: int = 20
    concurrent_requests: int = 25

class GeminiLLMProvider:
    """
    Gemini LLM Provider compatible with Microsoft GraphRAG interface
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._configure_genai()
        self._setup_rate_limiting()
        
        # Model selection logic for different Gemini variants
        self.model_mapping = {
            "gemini-1.5-pro-002": "gemini-1.5-pro-002",
            "gemini-2.0-flash-exp": "gemini-2.0-flash-exp", 
            "gemini-1.5-flash-002": "gemini-1.5-flash-002",
            "gemini-pro": "gemini-1.5-pro-002",  # Fallback
            "gemini-flash": "gemini-2.0-flash-exp",  # Alias
        }
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to initialize tokenizer: {e}")
            self.tokenizer = None
    
    def _configure_genai(self):
        """Configure Google Generative AI"""
        try:
            genai.configure(api_key=self.config.api_key)
            logger.info(f"Configured Gemini with model: {self.config.model}")
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {e}")
            raise
    
    def _setup_rate_limiting(self):
        """Setup rate limiting for API calls"""
        self._request_semaphore = asyncio.Semaphore(self.config.concurrent_requests)
        self._last_request_time = 0
        self._request_interval = 60.0 / self.config.requests_per_minute
    
    def _get_model_name(self, model: str) -> str:
        """Get the actual model name for Gemini API"""
        return self.model_mapping.get(model, model)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough estimation: 1 token ≈ 4 characters
            return len(text) // 4
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _make_request(self, prompt: str, model: str) -> str:
        """Make rate-limited request to Gemini API"""
        async with self._request_semaphore:
            # Rate limiting
            current_time = asyncio.get_event_loop().time()
            time_since_last = current_time - self._last_request_time
            if time_since_last < self._request_interval:
                await asyncio.sleep(self._request_interval - time_since_last)
            
            self._last_request_time = asyncio.get_event_loop().time()
            
            try:
                # Get the actual model name
                model_name = self._get_model_name(model)
                
                # Create the model instance
                genai_model = genai.GenerativeModel(model_name)
                
                # Generate content
                response = genai_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        max_output_tokens=self.config.max_tokens,
                    )
                )
                
                if response.text:
                    return response.text.strip()
                else:
                    logger.warning("Empty response from Gemini API")
                    return ""
                    
            except Exception as e:
                logger.error(f"Gemini API request failed: {e}")
                raise
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate completion for GraphRAG compatibility
        """
        # Use provided model or config default
        model = model or self.config.model
        
        # Convert messages to prompt text
        if isinstance(messages, list):
            # Handle chat format messages
            prompt_parts = []
            for msg in messages:
                if isinstance(msg, dict):
                    content = msg.get('content', str(msg))
                    role = msg.get('role', 'user')
                    if role == 'system':
                        prompt_parts.append(f"System: {content}")
                    elif role == 'user':
                        prompt_parts.append(f"User: {content}")
                    else:
                        prompt_parts.append(content)
                else:
                    prompt_parts.append(str(msg))
            prompt = "\n\n".join(prompt_parts)
        else:
            prompt = str(messages)
        
        # Token count validation
        token_count = self._count_tokens(prompt)
        max_input_tokens = 30000  # Conservative limit for Gemini
        
        if token_count > max_input_tokens:
            logger.warning(f"Prompt too long ({token_count} tokens), truncating...")
            # Truncate from the beginning to preserve the most recent context
            prompt = prompt[-max_input_tokens * 4:]  # Rough conversion back to chars
        
        try:
            result = await self._make_request(prompt, model)
            logger.debug(f"Generated {len(result)} characters with model {model}")
            return result
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    # GraphRAG compatibility methods
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """Async generate method for GraphRAG compatibility"""
        return await self.generate([{"role": "user", "content": prompt}], **kwargs)
    
    def generate_sync(self, prompt: str, **kwargs) -> str:
        """Synchronous generate method"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.agenerate(prompt, **kwargs))
        finally:
            loop.close()

# Factory functions for GraphRAG integration
def create_gemini_llm(config_dict: Dict[str, Any]) -> GeminiLLMProvider:
    """Create Gemini LLM provider from config dictionary"""
    config = LLMConfig(
        api_key=config_dict.get("api_key", os.getenv("GEMINI_API_KEY")),
        model=config_dict.get("model", "gemini-1.5-pro-002"),
        temperature=config_dict.get("temperature", 0.0),
        top_p=config_dict.get("top_p", 1.0),
        max_tokens=config_dict.get("max_tokens", 8192),
        requests_per_minute=config_dict.get("requests_per_minute", 10000),
        tokens_per_minute=config_dict.get("tokens_per_minute", 150000),
        max_retries=config_dict.get("max_retries", 20),
        concurrent_requests=config_dict.get("concurrent_requests", 25),
    )
    
    if not config.api_key:
        raise ValueError("GEMINI_API_KEY is required")
    
    return GeminiLLMProvider(config)

# GraphRAG LLM interface adapter
class GraphRAGGeminiAdapter:
    """Adapter to make Gemini work with GraphRAG's expected interface"""
    
    def __init__(self, gemini_provider: GeminiLLMProvider):
        self.provider = gemini_provider
    
    async def __call__(self, prompt: str, **kwargs) -> str:
        """GraphRAG expects a callable LLM function"""
        return await self.provider.agenerate(prompt, **kwargs)

def get_llm_config_for_graphrag() -> Dict[str, Any]:
    """Get LLM configuration for GraphRAG yaml config"""
    return {
        "api_key": "${GEMINI_API_KEY}",
        "type": "azure_openai_chat",  # We'll override the actual implementation
        "model": "gemini-1.5-pro-002",
        "api_base": None,
        "api_version": None,
        "deployment_name": None,
        "tokens_per_minute": 150000,
        "requests_per_minute": 10000,
        "max_retries": 20,
        "max_retry_wait": 10.0,
        "sleep_on_rate_limit_recommendation": True,
        "concurrent_requests": 25,
        "temperature": 0,
        "top_p": 1,
        "n": 1,
    }

# Model selection utility
def select_best_gemini_model(task_type: str = "general") -> str:
    """
    Select the best Gemini model for different tasks
    
    Args:
        task_type: Type of task (entity_extraction, community_report, general)
    
    Returns:
        Model name
    """
    model_recommendations = {
        "entity_extraction": "gemini-2.0-flash-exp",  # Fast for structured extraction
        "community_report": "gemini-1.5-pro-002",    # Better for complex reasoning
        "summarization": "gemini-2.0-flash-exp",     # Fast for summaries
        "general": "gemini-1.5-pro-002",             # Balanced choice
        "embeddings": "text-embedding-3-small",     # For embeddings (OpenAI compatible)
    }
    
    return model_recommendations.get(task_type, "gemini-1.5-pro-002")

if __name__ == "__main__":
    # Test the provider
    import asyncio
    
    async def test_provider():
        config = LLMConfig(
            api_key=os.getenv("GEMINI_API_KEY"),
            model="gemini-2.0-flash-exp"
        )
        
        if not config.api_key:
            print("Please set GEMINI_API_KEY environment variable")
            return
        
        provider = GeminiLLMProvider(config)
        
        # Test simple generation
        result = await provider.generate([
            {"role": "user", "content": "What is Microsoft GraphRAG?"}
        ])
        
        print(f"Test result: {result[:200]}...")
    
    asyncio.run(test_provider())
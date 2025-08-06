#!/usr/bin/env python3
"""
Gemini Web Search Integration for Perplexity-Style System
SuperClaude Wave Orchestration - Phase 3A

Real-time web search + GraphRAG intelligence integration using Gemini API
"""

import os
import json
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import re
import hashlib

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiWebSearchError(Exception):
    """Gemini Web Search specific exceptions"""
    pass

class GeminiWebSearchProvider:
    """
    Gemini API Web Search Provider for Perplexity-style real-time search
    
    Integrates Google's Gemini models with web search capabilities for:
    - Real-time information retrieval
    - Dynamic content analysis
    - Source attribution and verification
    - Context-aware search result processing
    """
    
    def __init__(self, api_key: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Gemini Web Search Provider
        
        Args:
            api_key: Google AI API key
            config: Configuration options
        """
        self.api_key = api_key
        self.config = config or {}
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Model selection for different operations - Use 1.5 models for reliable web search
        self.search_model = self.config.get("search_model", "gemini-1.5-flash-002")  # Fast for search with web grounding
        self.analysis_model = self.config.get("analysis_model", "gemini-1.5-pro-002")  # Deep for analysis
        self.synthesis_model = self.config.get("synthesis_model", "gemini-1.5-flash-002")  # Balanced
        
        # Search configuration
        self.max_search_results = self.config.get("max_search_results", 10)
        self.search_timeout = self.config.get("search_timeout", 30)
        self.cache_duration = self.config.get("cache_duration", 300)  # 5 minutes
        
        # Safety settings
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        # Initialize cache
        self.search_cache = {}
        self.cache_dir = Path("./cache/web_search")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        
        logger.info(f"Gemini Web Search Provider initialized with models: {self.search_model}, {self.analysis_model}")
    
    async def analyze_query_freshness(self, query: str) -> Dict[str, Any]:
        """
        Analyze whether a query requires real-time web search
        
        Args:
            query: User query to analyze
            
        Returns:
            Dict with freshness analysis results
        """
        analysis_prompt = f"""
Analyze this query to determine if it requires real-time web search for current information:

Query: "{query}"

Consider these factors:
1. Time-sensitive keywords (current, now, today, latest, recent, price, news)
2. Financial data, stock prices, cryptocurrency prices
3. Breaking news, current events, weather
4. Real-time status, live information
5. Recent updates, new releases

Return JSON format:
{{
    "requires_web_search": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "explanation",
    "search_type": "real_time|recent|general|none",
    "keywords": ["key", "words", "for", "search"],
    "urgency": "high|medium|low",
    "expected_freshness": "minutes|hours|days|static"
}}
"""
        
        try:
            model = genai.GenerativeModel(self.search_model)
            response = await asyncio.to_thread(
                model.generate_content,
                analysis_prompt,
                safety_settings=self.safety_settings,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=500
                )
            )
            
            # Extract response text and clean it
            response_text = response.text.strip()
            
            # Try to extract JSON from response (sometimes wrapped in code blocks)
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                if json_end > json_start:
                    response_text = response_text[json_start:json_end].strip()
            elif "{" in response_text and "}" in response_text:
                # Extract JSON part from response
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                response_text = response_text[json_start:json_end]
            
            # Parse JSON response
            result = json.loads(response_text)
            
            # Add metadata
            result["analyzed_at"] = datetime.now().isoformat()
            result["query"] = query
            
            logger.info(f"Query freshness analysis: {result['requires_web_search']} ({result['confidence']:.2f})")
            return result
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse freshness analysis JSON: {e}. Response: {response.text[:200]}...")
            return {
                "requires_web_search": True,  # Default to safe side
                "confidence": 0.5,
                "reasoning": "Analysis parsing failed, defaulting to web search",
                "search_type": "general",
                "keywords": query.split(),
                "urgency": "medium",
                "expected_freshness": "hours"
            }
            
        except Exception as e:
            logger.error(f"Query freshness analysis failed: {e}")
            raise GeminiWebSearchError(f"Freshness analysis failed: {e}")
    
    async def perform_web_search(self, query: str, search_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform web search using Gemini's grounding capabilities
        
        Args:
            query: Search query
            search_params: Additional search parameters
            
        Returns:
            Dict with search results and metadata
        """
        # Check cache first
        cache_key = self._generate_cache_key(query, search_params)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            logger.info(f"Using cached search result for: {query}")
            return cached_result
        
        # Rate limiting
        await self._enforce_rate_limit()
        
        try:
            # Prepare search prompt with grounding instructions
            search_prompt = self._build_search_prompt(query, search_params)
            
            # Use Gemini with web grounding - Updated for 2025 API syntax
            logger.info(f"Using google_search tool (2025 syntax) for model: {self.search_model}")
            
            # For Gemini 1.5 models: Use legacy google_search_retrieval with simplified syntax
            # For newer models: Use google_search tool
            if "1.5" in self.search_model:
                # Legacy approach with simplified syntax for backward compatibility
                model = genai.GenerativeModel(
                    self.search_model,
                    tools='google_search_retrieval'  # Simplified syntax for Gemini 1.5
                )
            else:
                # Modern approach for newer models
                model = genai.GenerativeModel(
                    self.search_model,
                    tools='google_search'  # New syntax for Gemini 2.0+
                )
            
            response = await asyncio.to_thread(
                model.generate_content,
                search_prompt,
                safety_settings=self.safety_settings,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.4,
                    max_output_tokens=2048
                )
            )
            
            # Process response and extract sources
            search_result = self._process_search_response(response, query)
            
            # Cache the result
            self._cache_result(cache_key, search_result)
            
            logger.info(f"Web search completed for: {query} ({len(search_result.get('sources', []))} sources)")
            return search_result
            
        except Exception as e:
            logger.error(f"Web search failed for query '{query}': {e}")
            raise GeminiWebSearchError(f"Web search failed: {e}")
    
    def _build_search_prompt(self, query: str, search_params: Optional[Dict[str, Any]] = None) -> str:
        """Build optimized search prompt for Gemini web grounding"""
        
        search_params = search_params or {}
        
        # Base search prompt
        prompt = f"""
Search for current, accurate information about: {query}

Please provide:
1. Current factual information with specific data points
2. Multiple reliable sources with URLs
3. Timestamps or publication dates when available
4. Any conflicting information from different sources

Focus on:
- Accuracy and recency of information
- Authoritative and credible sources
- Specific data, numbers, and facts
- Different perspectives if relevant

Query: {query}
"""
        
        # Add specific search modifiers based on parameters
        if search_params.get("search_type") == "real_time":
            prompt += "\nEmphasize the most recent information available (within hours or minutes if possible)."
        
        if search_params.get("urgency") == "high":
            prompt += "\nThis is time-critical information - prioritize the most current sources."
        
        if search_params.get("keywords"):
            keywords = ", ".join(search_params["keywords"])
            prompt += f"\nKey search terms: {keywords}"
        
        return prompt
    
    def _process_search_response(self, response, original_query: str) -> Dict[str, Any]:
        """Process Gemini search response and extract structured data"""
        
        # Extract main response text
        response_text = response.text if hasattr(response, 'text') else str(response)
        
        # Extract sources from grounding metadata
        sources = []
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'grounding_metadata'):
                grounding_metadata = candidate.grounding_metadata
                if hasattr(grounding_metadata, 'web_search_queries'):
                    for query_metadata in grounding_metadata.web_search_queries:
                        for result in getattr(query_metadata, 'search_results', []):
                            source = {
                                "title": getattr(result, 'title', ''),
                                "url": getattr(result, 'url', ''),
                                "snippet": getattr(result, 'snippet', ''),
                                "source_type": "web_search",
                                "confidence": getattr(result, 'confidence', 0.8),
                                "retrieved_at": datetime.now().isoformat()
                            }
                            sources.append(source)
        
        # If no sources from grounding metadata, try to extract from text
        if not sources:
            sources = self._extract_sources_from_text(response_text)
        
        # Generate search metadata
        search_metadata = {
            "query": original_query,
            "search_time": datetime.now().isoformat(),
            "model_used": self.search_model,
            "response_length": len(response_text),
            "sources_found": len(sources),
            "search_quality": self._assess_search_quality(response_text, sources)
        }
        
        return {
            "query": original_query,
            "response": response_text,
            "sources": sources,
            "metadata": search_metadata,
            "cached": False,
            "search_timestamp": datetime.now().isoformat()
        }
    
    def _extract_sources_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract source URLs and information from response text"""
        sources = []
        
        # Pattern to match URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+[^\s<>"{}|\\^`\[\].,;:!?]'
        urls = re.findall(url_pattern, text)
        
        for i, url in enumerate(urls):
            # Try to extract context around the URL
            url_index = text.find(url)
            context_start = max(0, url_index - 100)
            context_end = min(len(text), url_index + len(url) + 100)
            context = text[context_start:context_end].strip()
            
            source = {
                "title": f"Source {i+1}",
                "url": url,
                "snippet": context,
                "source_type": "extracted",
                "confidence": 0.7,
                "retrieved_at": datetime.now().isoformat()
            }
            sources.append(source)
        
        return sources
    
    def _assess_search_quality(self, response_text: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the quality of search results"""
        
        quality_score = 0.0
        factors = []
        
        # Check response length (more detailed is usually better)
        if len(response_text) > 500:
            quality_score += 0.2
            factors.append("detailed_response")
        
        # Check number of sources
        if len(sources) >= 3:
            quality_score += 0.3
            factors.append("multiple_sources")
        elif len(sources) >= 1:
            quality_score += 0.1
            factors.append("has_sources")
        
        # Check for specific data (numbers, dates, etc.)
        if re.search(r'\d+', response_text):
            quality_score += 0.2
            factors.append("contains_data")
        
        # Check for recent dates
        current_year = datetime.now().year
        if str(current_year) in response_text:
            quality_score += 0.1
            factors.append("recent_information")
        
        # Check for authoritative language
        if any(term in response_text.lower() for term in ['according to', 'reported', 'official', 'confirmed']):
            quality_score += 0.1
            factors.append("authoritative_language")
        
        # Cap at 1.0
        quality_score = min(1.0, quality_score)
        
        return {
            "overall_score": quality_score,
            "factors": factors,
            "assessment": "high" if quality_score > 0.7 else "medium" if quality_score > 0.4 else "low"
        }
    
    async def synthesize_with_graphrag(self, 
                                     web_results: Dict[str, Any], 
                                     graphrag_context: Dict[str, Any],
                                     original_query: str) -> Dict[str, Any]:
        """
        Synthesize web search results with GraphRAG knowledge
        
        Args:
            web_results: Results from web search
            graphrag_context: Context from GraphRAG system
            original_query: Original user query
            
        Returns:
            Synthesized response combining both sources
        """
        synthesis_prompt = f"""
You are an AI assistant creating a comprehensive response by combining real-time web search data with existing knowledge graph information.

Original Query: {original_query}

WEB SEARCH RESULTS:
{web_results.get('response', '')}

Sources from web search:
{self._format_sources_for_prompt(web_results.get('sources', []))}

KNOWLEDGE GRAPH CONTEXT:
{self._format_graphrag_context(graphrag_context)}

TASK:
Create a comprehensive, accurate response that:
1. Integrates both real-time web data and knowledge graph insights
2. Prioritizes the most current information for time-sensitive queries
3. Uses knowledge graph data to provide deeper context and relationships
4. Clearly attributes information to sources
5. Identifies and resolves any conflicts between sources
6. Provides specific data, numbers, and facts when available

Format your response with:
- Main answer/summary
- Key facts with source attribution
- Additional context from knowledge graph
- Source reliability assessment

Response:
"""
        
        try:
            model = genai.GenerativeModel(self.synthesis_model)
            response = await asyncio.to_thread(
                model.generate_content,
                synthesis_prompt,
                safety_settings=self.safety_settings,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.5,
                    max_output_tokens=3000
                )
            )
            
            synthesized_response = response.text
            
            # Create comprehensive result
            result = {
                "query": original_query,
                "synthesized_response": synthesized_response,
                "web_results": web_results,
                "graphrag_context": graphrag_context,
                "synthesis_metadata": {
                    "model_used": self.synthesis_model,
                    "synthesis_time": datetime.now().isoformat(),
                    "web_sources_count": len(web_results.get('sources', [])),
                    "graphrag_entities_count": len(graphrag_context.get('entities', [])),
                    "response_length": len(synthesized_response),
                    "synthesis_quality": self._assess_synthesis_quality(synthesized_response, web_results, graphrag_context)
                },
                "all_sources": self._combine_all_sources(web_results, graphrag_context),
                "generated_at": datetime.now().isoformat()
            }
            
            logger.info(f"Synthesis completed for query: {original_query}")
            return result
            
        except Exception as e:
            logger.error(f"Synthesis failed for query '{original_query}': {e}")
            raise GeminiWebSearchError(f"Synthesis failed: {e}")
    
    def _format_sources_for_prompt(self, sources: List[Dict[str, Any]]) -> str:
        """Format web sources for inclusion in synthesis prompt"""
        if not sources:
            return "No sources available"
        
        formatted_sources = []
        for i, source in enumerate(sources[:5]):  # Limit to top 5
            formatted_sources.append(f"""
Source {i+1}: {source.get('title', 'Unknown Title')}
URL: {source.get('url', 'No URL')}
Content: {source.get('snippet', 'No content')}
""")
        
        return "\n".join(formatted_sources)
    
    def _format_graphrag_context(self, context: Dict[str, Any]) -> str:
        """Format GraphRAG context for inclusion in synthesis prompt"""
        if not context:
            return "No knowledge graph context available"
        
        formatted_parts = []
        
        # Add entities
        if context.get('entities'):
            entities_text = []
            for entity in context['entities'][:5]:  # Top 5 entities
                entities_text.append(f"- {entity.get('name', 'Unknown')}: {entity.get('description', 'No description')}")
            formatted_parts.append("Key Entities:\n" + "\n".join(entities_text))
        
        # Add relationships
        if context.get('relationships'):
            relationships_text = []
            for rel in context['relationships'][:5]:  # Top 5 relationships
                relationships_text.append(f"- {rel.get('source', '')} → {rel.get('relationship', '')} → {rel.get('target', '')}")
            formatted_parts.append("Key Relationships:\n" + "\n".join(relationships_text))
        
        # Add response if available
        if context.get('response'):
            formatted_parts.append(f"Knowledge Graph Analysis:\n{context['response']}")
        
        return "\n\n".join(formatted_parts) if formatted_parts else "No structured context available"
    
    def _assess_synthesis_quality(self, response: str, web_results: Dict, graphrag_context: Dict) -> Dict[str, Any]:
        """Assess quality of synthesized response"""
        
        quality_score = 0.0
        factors = []
        
        # Check if response integrates both sources
        web_mentioned = any(word in response.lower() for word in ['according to', 'source', 'report', 'website'])
        graphrag_mentioned = any(word in response.lower() for word in ['knowledge', 'context', 'relationship', 'entity'])
        
        if web_mentioned and graphrag_mentioned:
            quality_score += 0.3
            factors.append("integrated_sources")
        elif web_mentioned or graphrag_mentioned:
            quality_score += 0.1
            factors.append("single_source_integration")
        
        # Check for specific data
        if re.search(r'\d+', response):
            quality_score += 0.2
            factors.append("contains_specific_data")
        
        # Check response length and detail
        if len(response) > 1000:
            quality_score += 0.2
            factors.append("comprehensive_response")
        elif len(response) > 500:
            quality_score += 0.1
            factors.append("detailed_response")
        
        # Check for source attribution
        if any(word in response.lower() for word in ['source:', 'according to', 'reported by']):
            quality_score += 0.2
            factors.append("source_attribution")
        
        # Check for current information
        current_year = datetime.now().year
        if str(current_year) in response:
            quality_score += 0.1
            factors.append("current_information")
        
        quality_score = min(1.0, quality_score)
        
        return {
            "overall_score": quality_score,
            "factors": factors,
            "assessment": "high" if quality_score > 0.7 else "medium" if quality_score > 0.4 else "low"
        }
    
    def _combine_all_sources(self, web_results: Dict, graphrag_context: Dict) -> List[Dict[str, Any]]:
        """Combine sources from both web search and GraphRAG"""
        all_sources = []
        
        # Add web sources
        for source in web_results.get('sources', []):
            all_sources.append({
                **source,
                "source_origin": "web_search"
            })
        
        # Add GraphRAG sources (if available)
        if graphrag_context.get('text_units'):
            for unit in graphrag_context['text_units'][:3]:  # Top 3
                all_sources.append({
                    "title": f"Knowledge Graph: {unit.get('source', 'Internal Knowledge')}",
                    "content": unit.get('text', '')[:200] + "...",
                    "source_type": "knowledge_graph",
                    "source_origin": "graphrag",
                    "confidence": unit.get('similarity_score', 0.8)
                })
        
        return all_sources
    
    def _generate_cache_key(self, query: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key for search results"""
        cache_data = {
            "query": query.lower().strip(),
            "params": params or {}
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached search result if still valid"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                # Check if cache is still valid
                cache_time = datetime.fromisoformat(cached_data.get('cached_at', ''))
                if datetime.now() - cache_time < timedelta(seconds=self.cache_duration):
                    cached_data['cached'] = True
                    return cached_data
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"Invalid cache file {cache_key}: {e}")
        
        return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache search result"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            result_to_cache = {
                **result,
                'cached_at': datetime.now().isoformat()
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result_to_cache, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.warning(f"Failed to cache result {cache_key}: {e}")
    
    async def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()

class PerplexityStyleGraphRAG:
    """
    Perplexity-style system combining GraphRAG with real-time web search
    
    Main orchestrator that provides the complete Perplexity experience:
    - Analyzes queries for freshness requirements
    - Performs web search when needed
    - Integrates with existing GraphRAG knowledge
    - Synthesizes comprehensive responses
    """
    
    def __init__(self, 
                 web_search_provider: GeminiWebSearchProvider,
                 graphrag_search_engine,  # LocalSearch, GlobalSearch, or HybridSearch
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize Perplexity-style GraphRAG system
        
        Args:
            web_search_provider: Gemini web search provider
            graphrag_search_engine: GraphRAG search engine (local/global/hybrid)
            config: Configuration options
        """
        self.web_search = web_search_provider
        self.graphrag = graphrag_search_engine
        self.config = config or {}
        
        # Processing configuration
        self.freshness_threshold = self.config.get("freshness_threshold", 0.7)
        self.always_use_graphrag = self.config.get("always_use_graphrag", True)
        self.max_response_time = self.config.get("max_response_time", 30)
        
        logger.info("Perplexity-style GraphRAG system initialized")
    
    async def process_query(self, 
                          user_query: str, 
                          conversation_history: Optional[List[Dict[str, str]]] = None,
                          force_web_search: bool = False) -> Dict[str, Any]:
        """
        Process user query with Perplexity-style intelligence
        
        Args:
            user_query: User's question
            conversation_history: Previous conversation context
            force_web_search: Force web search regardless of freshness analysis
            
        Returns:
            Comprehensive response with web and GraphRAG integration
        """
        start_time = time.time()
        logger.info(f"Processing Perplexity-style query: {user_query}")
        
        try:
            # Step 1: Analyze query freshness
            freshness_analysis = await self.web_search.analyze_query_freshness(user_query)
            
            # Step 2: Determine search strategy
            needs_web_search = (
                force_web_search or 
                freshness_analysis["requires_web_search"] or
                freshness_analysis["confidence"] > self.freshness_threshold
            )
            
            # Step 3: Execute searches in parallel when possible
            tasks = []
            
            # Always use GraphRAG for existing knowledge
            if self.always_use_graphrag:
                graphrag_task = asyncio.create_task(
                    self.graphrag.search(user_query, conversation_history)
                )
                tasks.append(("graphrag", graphrag_task))
            
            # Use web search if needed
            web_results = None
            if needs_web_search:
                search_params = {
                    "search_type": freshness_analysis.get("search_type", "general"),
                    "urgency": freshness_analysis.get("urgency", "medium"),
                    "keywords": freshness_analysis.get("keywords", [])
                }
                
                web_search_task = asyncio.create_task(
                    self.web_search.perform_web_search(user_query, search_params)
                )
                tasks.append(("web_search", web_search_task))
            
            # Execute tasks in parallel
            results = {}
            if tasks:
                completed_tasks = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
                
                for (task_name, _), result in zip(tasks, completed_tasks):
                    if isinstance(result, Exception):
                        logger.error(f"{task_name} failed: {result}")
                        results[task_name] = {"error": str(result)}
                    else:
                        results[task_name] = result
            
            # Step 4: Synthesize results
            graphrag_context = results.get("graphrag", {})
            web_results = results.get("web_search", {})
            
            if web_results and not web_results.get("error"):
                # Synthesize web + GraphRAG
                final_result = await self.web_search.synthesize_with_graphrag(
                    web_results, graphrag_context, user_query
                )
                response_type = "web_graphrag_synthesis"
            elif graphrag_context and not graphrag_context.get("error"):
                # GraphRAG only
                final_result = {
                    "query": user_query,
                    "synthesized_response": graphrag_context.get("response", "No response available"),
                    "web_results": {},
                    "graphrag_context": graphrag_context,
                    "all_sources": self._extract_graphrag_sources(graphrag_context),
                    "generated_at": datetime.now().isoformat()
                }
                response_type = "graphrag_only"
            else:
                # Fallback response
                final_result = {
                    "query": user_query,
                    "synthesized_response": "I apologize, but I'm unable to find sufficient information to answer your query at this time.",
                    "web_results": web_results or {},
                    "graphrag_context": graphrag_context or {},
                    "all_sources": [],
                    "generated_at": datetime.now().isoformat()
                }
                response_type = "fallback"
            
            # Add processing metadata
            processing_time = time.time() - start_time
            final_result["processing_metadata"] = {
                "response_type": response_type,
                "processing_time_seconds": processing_time,
                "freshness_analysis": freshness_analysis,
                "web_search_performed": needs_web_search and not web_results.get("error"),
                "graphrag_used": bool(graphrag_context and not graphrag_context.get("error")),
                "total_sources": len(final_result.get("all_sources", [])),
                "response_quality": self._assess_overall_quality(final_result)
            }
            
            logger.info(f"Query processed successfully in {processing_time:.2f}s ({response_type})")
            return final_result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "query": user_query,
                "synthesized_response": f"I encountered an error while processing your query: {str(e)}",
                "error": str(e),
                "generated_at": datetime.now().isoformat(),
                "processing_metadata": {
                    "response_type": "error",
                    "processing_time_seconds": time.time() - start_time,
                    "error_occurred": True
                }
            }
    
    def _extract_graphrag_sources(self, graphrag_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract source information from GraphRAG context"""
        sources = []
        
        # Add entity sources
        for entity in graphrag_context.get("entities", [])[:3]:  # Top 3 entities
            sources.append({
                "title": f"Knowledge Graph Entity: {entity.get('name', 'Unknown')}",
                "content": entity.get("description", "No description"),
                "source_type": "knowledge_graph_entity",
                "source_origin": "graphrag",
                "confidence": entity.get("relevance_score", 0.8)
            })
        
        # Add text unit sources
        for unit in graphrag_context.get("text_units", [])[:3]:  # Top 3
            sources.append({
                "title": f"Knowledge Base: {unit.get('source', 'Internal Knowledge')}",
                "content": unit.get("text", "")[:200] + "...",
                "source_type": "knowledge_graph_text",
                "source_origin": "graphrag",
                "confidence": unit.get("similarity_score", 0.8)
            })
        
        return sources
    
    def _assess_overall_quality(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall quality of the final response"""
        
        quality_score = 0.0
        factors = []
        
        response = result.get("synthesized_response", "")
        sources = result.get("all_sources", [])
        
        # Check response length and detail
        if len(response) > 1000:
            quality_score += 0.2
            factors.append("comprehensive")
        elif len(response) > 500:
            quality_score += 0.1
            factors.append("detailed")
        
        # Check source diversity
        web_sources = len([s for s in sources if s.get("source_origin") == "web_search"])
        graphrag_sources = len([s for s in sources if s.get("source_origin") == "graphrag"])
        
        if web_sources > 0 and graphrag_sources > 0:
            quality_score += 0.3
            factors.append("multi_source_integration")
        elif web_sources > 0 or graphrag_sources > 0:
            quality_score += 0.1
            factors.append("single_source")
        
        # Check for specific data
        if re.search(r'\d+', response):
            quality_score += 0.2
            factors.append("contains_data")
        
        # Check for current information
        current_year = datetime.now().year
        if str(current_year) in response:
            quality_score += 0.1
            factors.append("current_info")
        
        # Check for source attribution
        if any(word in response.lower() for word in ['according to', 'source', 'reported']):
            quality_score += 0.2
            factors.append("source_attribution")
        
        quality_score = min(1.0, quality_score)
        
        return {
            "overall_score": quality_score,
            "factors": factors,
            "assessment": "high" if quality_score > 0.7 else "medium" if quality_score > 0.4 else "low",
            "web_sources": web_sources,
            "graphrag_sources": graphrag_sources,
            "total_sources": len(sources)
        }

# Factory functions
def create_gemini_web_search(api_key: str, config: Optional[Dict[str, Any]] = None) -> GeminiWebSearchProvider:
    """Create Gemini Web Search Provider"""
    return GeminiWebSearchProvider(api_key, config)

def create_perplexity_graphrag(web_search_provider: GeminiWebSearchProvider,
                              graphrag_search_engine,
                              config: Optional[Dict[str, Any]] = None) -> PerplexityStyleGraphRAG:
    """Create complete Perplexity-style GraphRAG system"""
    return PerplexityStyleGraphRAG(web_search_provider, graphrag_search_engine, config)

# CLI interface for testing
async def main():
    """CLI interface for testing Perplexity-style system"""
    import argparse
    from gemini_llm_provider import create_gemini_llm
    from graphrag_search import create_search_engines
    
    parser = argparse.ArgumentParser(description="Perplexity-Style GraphRAG Test")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--data-dir", default="./output", help="GraphRAG data directory")
    parser.add_argument("--force-web", action="store_true", help="Force web search")
    parser.add_argument("--config", help="Config file path")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Create components
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable required")
    
    # Create web search provider
    web_search = create_gemini_web_search(api_key, config.get("web_search", {}))
    
    # Create GraphRAG search engines
    gemini_provider = create_gemini_llm({"api_key": api_key, "model": "gemini-2.0-flash-exp"})
    _, _, hybrid_search = create_search_engines(args.data_dir, gemini_provider)
    
    # Create Perplexity-style system
    perplexity_system = create_perplexity_graphrag(
        web_search, hybrid_search, config.get("perplexity", {})
    )
    
    # Process query
    print(f"\n🔍 Processing Query: {args.query}")
    print("=" * 60)
    
    result = await perplexity_system.process_query(args.query, force_web_search=args.force_web)
    
    # Display results
    print(f"\n📝 Response:")
    print(result["synthesized_response"])
    
    print(f"\n📊 Metadata:")
    metadata = result.get("processing_metadata", {})
    print(f"  • Response Type: {metadata.get('response_type', 'unknown')}")
    print(f"  • Processing Time: {metadata.get('processing_time_seconds', 0):.2f}s")
    print(f"  • Web Search Used: {metadata.get('web_search_performed', False)}")
    print(f"  • GraphRAG Used: {metadata.get('graphrag_used', False)}")
    print(f"  • Total Sources: {metadata.get('total_sources', 0)}")
    
    quality = metadata.get("response_quality", {})
    print(f"  • Quality Score: {quality.get('overall_score', 0):.2f} ({quality.get('assessment', 'unknown')})")
    
    print(f"\n📚 Sources ({len(result.get('all_sources', []))}):")
    for i, source in enumerate(result.get("all_sources", [])[:5]):  # Show top 5
        print(f"  {i+1}. {source.get('title', 'Unknown Title')}")
        if source.get('url'):
            print(f"     URL: {source['url']}")
        print(f"     Type: {source.get('source_type', 'unknown')} | Origin: {source.get('source_origin', 'unknown')}")

if __name__ == "__main__":
    asyncio.run(main())
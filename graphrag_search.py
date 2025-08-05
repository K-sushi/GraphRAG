#!/usr/bin/env python3
"""
Microsoft GraphRAG Search Engines Implementation
SuperClaude Wave Orchestration - Phase 2

Local, Global, and Hybrid search implementations for GraphRAG
"""

import os
import json
import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import faiss

from gemini_llm_provider import GeminiLLMProvider
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphRAGSearchBase:
    """Base class for GraphRAG search engines"""
    
    def __init__(self, 
                 data_dir: str,
                 gemini_provider: GeminiLLMProvider,
                 embedding_model: Optional[SentenceTransformer] = None):
        self.data_dir = Path(data_dir)
        self.gemini_provider = gemini_provider
        
        # Initialize embedding model
        if embedding_model is None:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            self.embedding_model = embedding_model
        
        # Load data
        self.entities = []
        self.relationships = []
        self.text_units = []
        self.communities = []
        self.community_reports = []
        self.documents = []
        
        # Vector store
        self.vector_store = None
        
        self._load_data()
    
    def _load_data(self):
        """Load GraphRAG data from JSON files"""
        data_files = {
            "entities": "entities.json",
            "relationships": "relationships.json", 
            "text_units": "text_units.json",
            "communities": "communities.json",
            "community_reports": "community_reports.json",
            "documents": "documents.json",
        }
        
        for attr_name, filename in data_files.items():
            file_path = self.data_dir / filename
            
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    setattr(self, attr_name, data)
                    logger.info(f"Loaded {len(data)} {attr_name}")
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
                    setattr(self, attr_name, [])
            else:
                logger.warning(f"File not found: {filename}")
                setattr(self, attr_name, [])
        
        # Load vector store
        self._load_vector_store()
    
    def _load_vector_store(self):
        """Load FAISS vector store"""
        vector_store_path = self.data_dir.parent / "cache" / "vector_store.faiss"
        
        if vector_store_path.exists():
            try:
                self.vector_store = faiss.read_index(str(vector_store_path))
                logger.info(f"Loaded vector store with {self.vector_store.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to load vector store: {e}")
                self.vector_store = None
        else:
            logger.warning("Vector store not found")
    
    def _get_entity_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Find entity by name"""
        for entity in self.entities:
            if entity["name"].lower() == name.lower():
                return entity
        return None
    
    def _get_relationships_for_entity(self, entity_name: str) -> List[Dict[str, Any]]:
        """Get all relationships involving an entity"""
        relationships = []
        for rel in self.relationships:
            if rel["source"].lower() == entity_name.lower() or rel["target"].lower() == entity_name.lower():
                relationships.append(rel)
        return relationships
    
    def _get_community_for_entity(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """Find community containing an entity"""
        for community in self.communities:
            if entity_name in community.get("entities", []):
                return community
        return None

class LocalSearch(GraphRAGSearchBase):
    """Local search implementation focusing on specific entities and their neighborhoods"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Local search configuration
        self.text_unit_prop = 0.5
        self.community_prop = 0.1
        self.top_k_entities = 10
        self.top_k_relationships = 10
        self.max_tokens = 12000
        
        logger.info("Local search engine initialized")
    
    async def search(self, 
                    query: str,
                    conversation_history: Optional[List[Dict[str, str]]] = None,
                    **kwargs) -> Dict[str, Any]:
        """Perform local search"""
        logger.info(f"Performing local search for: {query}")
        
        try:
            # Step 1: Extract entities from query
            query_entities = await self._extract_query_entities(query)
            
            # Step 2: Find relevant entities in knowledge graph
            relevant_entities = self._find_relevant_entities(query, query_entities)
            
            # Step 3: Get local context (neighbors, relationships)
            local_context = self._build_local_context(relevant_entities)
            
            # Step 4: Retrieve relevant text units
            relevant_text_units = await self._retrieve_relevant_text_units(query)
            
            # Step 5: Generate response using local context
            response = await self._generate_local_response(
                query, local_context, relevant_text_units, conversation_history
            )
            
            return {
                "response": response,
                "entities": relevant_entities[:self.top_k_entities],
                "relationships": local_context.get("relationships", [])[:self.top_k_relationships],
                "text_units": relevant_text_units[:10],
                "search_type": "local",
                "query_entities": query_entities,
                "context_stats": {
                    "entities_found": len(relevant_entities),
                    "relationships_found": len(local_context.get("relationships", [])),
                    "text_units_retrieved": len(relevant_text_units),
                }
            }
            
        except Exception as e:
            logger.error(f"Local search failed: {e}")
            raise
    
    async def _extract_query_entities(self, query: str) -> List[str]:
        """Extract entities mentioned in the query"""
        entity_extraction_prompt = f"""
Extract named entities from this query. Return only the entity names as a JSON list.

Query: {query}

Return format: ["Entity1", "Entity2", "Entity3"]
"""
        
        try:
            response = await self.gemini_provider.generate([
                {"role": "user", "content": entity_extraction_prompt}
            ])
            
            # Parse JSON response
            query_entities = json.loads(response)
            return query_entities if isinstance(query_entities, list) else []
            
        except Exception as e:
            logger.warning(f"Failed to extract query entities: {e}")
            return []
    
    def _find_relevant_entities(self, query: str, query_entities: List[str]) -> List[Dict[str, Any]]:
        """Find entities relevant to the query"""
        relevant_entities = []
        entity_scores = {}
        
        # 1. Direct matches with query entities
        for query_entity in query_entities:
            for entity in self.entities:
                if query_entity.lower() in entity["name"].lower():
                    entity_scores[entity["id"]] = entity_scores.get(entity["id"], 0) + 3
        
        # 2. Keyword matching in entity names and descriptions
        query_words = query.lower().split()
        for entity in self.entities:
            score = 0
            entity_text = f"{entity['name']} {entity.get('description', '')}".lower()
            
            for word in query_words:
                if word in entity_text:
                    score += 1
            
            if score > 0:
                entity_scores[entity["id"]] = entity_scores.get(entity["id"], 0) + score
        
        # 3. Sort by relevance score
        scored_entities = []
        for entity in self.entities:
            if entity["id"] in entity_scores:
                entity_copy = entity.copy()
                entity_copy["relevance_score"] = entity_scores[entity["id"]]
                scored_entities.append(entity_copy)
        
        # Sort by score and importance
        scored_entities.sort(key=lambda x: (x["relevance_score"], x.get("importance", 5)), reverse=True)
        
        return scored_entities
    
    def _build_local_context(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build local context around relevant entities"""
        context = {
            "entities": entities,
            "relationships": [],
            "communities": [],
        }
        
        # Get relationships between relevant entities
        entity_names = [e["name"] for e in entities[:self.top_k_entities]]
        
        for rel in self.relationships:
            if rel["source"] in entity_names or rel["target"] in entity_names:
                context["relationships"].append(rel)
        
        # Get communities containing these entities
        seen_communities = set()
        for entity in entities[:self.top_k_entities]:
            community = self._get_community_for_entity(entity["name"])
            if community and community["id"] not in seen_communities:
                context["communities"].append(community)
                seen_communities.add(community["id"])
        
        return context
    
    async def _retrieve_relevant_text_units(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve text units relevant to the query using vector similarity"""
        if not self.vector_store or not self.text_units:
            return []
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query])
            query_embedding = query_embedding.astype('float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search
            k = min(20, self.vector_store.ntotal)  # Top 20 similar text units
            scores, indices = self.vector_store.search(query_embedding, k)
            
            # Get relevant text units
            relevant_units = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.text_units):
                    unit = self.text_units[idx].copy()
                    unit["similarity_score"] = float(score)
                    unit["rank"] = i + 1
                    relevant_units.append(unit)
            
            return relevant_units
            
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []
    
    async def _generate_local_response(self, 
                                     query: str,
                                     context: Dict[str, Any],
                                     text_units: List[Dict[str, Any]],
                                     conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate response using local context"""
        
        # Build context information
        context_parts = []
        
        # Add entity information
        if context["entities"]:
            context_parts.append("## Relevant Entities:")
            for entity in context["entities"][:5]:  # Top 5 entities
                context_parts.append(f"- **{entity['name']}** ({entity.get('type', 'unknown')}): {entity.get('description', 'No description')}")
        
        # Add relationship information
        if context["relationships"]:
            context_parts.append("\n## Key Relationships:")
            for rel in context["relationships"][:5]:  # Top 5 relationships
                context_parts.append(f"- {rel['source']} → {rel['relationship']} → {rel['target']}: {rel.get('description', '')}")
        
        # Add relevant text excerpts
        if text_units:
            context_parts.append("\n## Relevant Context:")
            for unit in text_units[:3]:  # Top 3 text units
                context_parts.append(f"- {unit['text'][:200]}{'...' if len(unit['text']) > 200 else ''}")
        
        context_text = "\n".join(context_parts)
        
        # Build conversation history
        history_text = ""
        if conversation_history:
            history_parts = []
            for msg in conversation_history[-5:]:  # Last 5 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history_parts.append(f"{role}: {content}")
            history_text = "\n".join(history_parts)
        
        # Generate response prompt
        response_prompt = f"""
You are a helpful AI assistant with access to a knowledge graph. Use the provided context to answer the user's question accurately and comprehensively.

User Question: {query}

Knowledge Graph Context:
{context_text}

{f"Previous Conversation:{history_text}" if history_text else ""}

Instructions:
1. Answer the question using the provided context
2. Be specific and cite relevant entities and relationships
3. If the context doesn't fully answer the question, say so
4. Provide a clear, well-structured response

Answer:
"""
        
        try:
            response = await self.gemini_provider.generate([
                {"role": "user", "content": response_prompt}
            ])
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I apologize, but I encountered an error while generating a response to your question."

class GlobalSearch(GraphRAGSearchBase):
    """Global search implementation for comprehensive community-based answers"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Global search configuration
        self.max_tokens = 12000
        self.data_max_tokens = 12000
        self.map_max_tokens = 1000
        self.reduce_max_tokens = 2000
        self.concurrency = 8
        
        logger.info("Global search engine initialized")
    
    async def search(self, 
                    query: str,
                    conversation_history: Optional[List[Dict[str, str]]] = None,
                    **kwargs) -> Dict[str, Any]:
        """Perform global search using community reports"""
        logger.info(f"Performing global search for: {query}")
        
        try:
            # Step 1: Select relevant community reports
            relevant_reports = await self._select_relevant_reports(query)
            
            # Step 2: Map phase - analyze each community report
            mapped_responses = await self._map_community_reports(query, relevant_reports)
            
            # Step 3: Reduce phase - synthesize final response
            final_response = await self._reduce_responses(query, mapped_responses, conversation_history)
            
            return {
                "response": final_response,
                "communities": [report["community_id"] for report in relevant_reports],
                "community_reports": relevant_reports,
                "mapped_responses": mapped_responses,
                "search_type": "global",
                "context_stats": {
                    "communities_analyzed": len(relevant_reports),
                    "mapped_responses": len(mapped_responses),
                }
            }
            
        except Exception as e:
            logger.error(f"Global search failed: {e}")
            raise
    
    async def _select_relevant_reports(self, query: str) -> List[Dict[str, Any]]:
        """Select community reports relevant to the query"""
        if not self.community_reports:
            return []
        
        # Score reports based on content relevance
        scored_reports = []
        query_words = query.lower().split()
        
        for report in self.community_reports:
            score = 0
            report_text = f"{report.get('title', '')} {report.get('content', '')}".lower()
            
            # Keyword matching
            for word in query_words:
                if word in report_text:
                    score += report_text.count(word)
            
            # Entity matching
            report_entities = report.get("entities", [])
            for entity_name in report_entities:
                if any(word in entity_name.lower() for word in query_words):
                    score += 2
            
            if score > 0:
                report_copy = report.copy()
                report_copy["relevance_score"] = score
                scored_reports.append(report_copy)
        
        # Sort by relevance and rank
        scored_reports.sort(key=lambda x: (x["relevance_score"], x.get("rank", 0)), reverse=True)
        
        # Return top reports (limit based on token budget)
        return scored_reports[:10]  # Top 10 reports
    
    async def _map_community_reports(self, query: str, reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map phase: analyze each community report for relevance to the query"""
        
        map_prompt = """
Analyze the following community report and determine how it relates to the user's question.

User Question: {query}

Community Report:
Title: {title}
Content: {content}

Instructions:
1. Identify key information that helps answer the user's question
2. Extract relevant facts, relationships, and insights
3. Assign a relevance score (1-10) for how well this community addresses the question
4. Provide a brief summary of the relevant information

If this community is not relevant to the question, return "NOT_RELEVANT".

Response format:
Relevance Score: [1-10]
Summary: [Brief summary of relevant information]
Key Points:
- [Key point 1]
- [Key point 2]
- [Key point 3]
"""
        
        mapped_responses = []
        
        # Process reports concurrently with limited concurrency
        semaphore = asyncio.Semaphore(self.concurrency)
        
        async def process_report(report):
            async with semaphore:
                try:
                    prompt = map_prompt.format(
                        query=query,
                        title=report.get("title", "Untitled"),
                        content=report.get("content", "")[:self.map_max_tokens * 4]  # Rough token limit
                    )
                    
                    response = await self.gemini_provider.generate([
                        {"role": "user", "content": prompt}
                    ])
                    
                    if "NOT_RELEVANT" not in response:
                        return {
                            "community_id": report["community_id"],
                            "title": report.get("title", ""),
                            "mapped_content": response,
                            "original_report": report,
                        }
                    
                except Exception as e:
                    logger.warning(f"Failed to map report {report.get('id', 'unknown')}: {e}")
                
                return None
        
        # Execute mapping tasks
        tasks = [process_report(report) for report in reports]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        for result in results:
            if result is not None and not isinstance(result, Exception):
                mapped_responses.append(result)
        
        return mapped_responses
    
    async def _reduce_responses(self, 
                              query: str,
                              mapped_responses: List[Dict[str, Any]],
                              conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Reduce phase: synthesize final comprehensive response"""
        
        if not mapped_responses:
            return "I don't have enough information in my knowledge base to answer your question comprehensively."
        
        # Combine mapped responses
        combined_insights = []
        for i, response in enumerate(mapped_responses):
            combined_insights.append(f"## Community {i+1}: {response['title']}\n{response['mapped_content']}")
        
        insights_text = "\n\n".join(combined_insights)
        
        # Build conversation history
        history_text = ""
        if conversation_history:
            history_parts = []
            for msg in conversation_history[-3:]:  # Last 3 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history_parts.append(f"{role}: {content}")
            history_text = "\n".join(history_parts)
        
        reduce_prompt = f"""
You are a helpful AI assistant tasked with providing a comprehensive answer to the user's question based on analysis of multiple community reports from a knowledge graph.

User Question: {query}

Community Analysis Results:
{insights_text}

{f"Previous Conversation:{history_text}" if history_text else ""}

Instructions:
1. Synthesize the information from all community analyses
2. Provide a comprehensive, well-structured answer
3. Include specific details and examples from the communities
4. Address different aspects of the question if multiple perspectives exist
5. If there are conflicting information, acknowledge and explain
6. Be clear about the scope and limitations of your answer

Generate a comprehensive response:
"""
        
        try:
            response = await self.gemini_provider.generate([
                {"role": "user", "content": reduce_prompt}
            ])
            return response
            
        except Exception as e:
            logger.error(f"Response reduction failed: {e}")
            return "I apologize, but I encountered an error while synthesizing the comprehensive response."

class HybridSearch:
    """Hybrid search combining local and global search strategies"""
    
    def __init__(self, local_search: LocalSearch, global_search: GlobalSearch):
        self.local_search = local_search
        self.global_search = global_search
        
        logger.info("Hybrid search engine initialized")
    
    async def search(self, 
                    query: str,
                    mode: str = "auto",
                    conversation_history: Optional[List[Dict[str, str]]] = None,
                    **kwargs) -> Dict[str, Any]:
        """Perform hybrid search"""
        logger.info(f"Performing hybrid search for: {query} (mode: {mode})")
        
        try:
            if mode == "auto":
                # Determine best search strategy
                search_strategy = await self._determine_search_strategy(query)
            else:
                search_strategy = mode
            
            if search_strategy == "local":
                result = await self.local_search.search(query, conversation_history, **kwargs)
                result["search_strategy"] = "local_only"
                
            elif search_strategy == "global":
                result = await self.global_search.search(query, conversation_history, **kwargs)
                result["search_strategy"] = "global_only"
                
            else:  # hybrid
                # Run both searches in parallel
                local_task = asyncio.create_task(
                    self.local_search.search(query, conversation_history, **kwargs)
                )
                global_task = asyncio.create_task(
                    self.global_search.search(query, conversation_history, **kwargs)
                )
                
                local_result, global_result = await asyncio.gather(local_task, global_task)
                
                # Combine results
                result = await self._combine_search_results(query, local_result, global_result, conversation_history)
                result["search_strategy"] = "hybrid"
                result["local_result"] = local_result
                result["global_result"] = global_result
            
            return result
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise
    
    async def _determine_search_strategy(self, query: str) -> str:
        """Determine the best search strategy for the query"""
        
        strategy_prompt = f"""
Analyze this query and determine the best search strategy:

Query: {query}

Search Strategies:
1. LOCAL: Best for specific questions about particular entities, relationships, or localized information
2. GLOBAL: Best for broad questions requiring comprehensive overview or synthesis across multiple topics
3. HYBRID: Best for complex questions that need both specific details and broad context

Consider:
- Query scope (specific vs. broad)
- Information needs (detailed vs. comprehensive)
- Question complexity

Return only one word: LOCAL, GLOBAL, or HYBRID
"""
        
        try:
            response = await self.local_search.gemini_provider.generate([
                {"role": "user", "content": strategy_prompt}
            ])
            
            strategy = response.strip().upper()
            if strategy in ["LOCAL", "GLOBAL", "HYBRID"]:
                return strategy.lower()
            else:
                return "hybrid"  # Default
                
        except Exception as e:
            logger.warning(f"Failed to determine search strategy: {e}")
            return "hybrid"  # Default fallback
    
    async def _combine_search_results(self, 
                                    query: str,
                                    local_result: Dict[str, Any],
                                    global_result: Dict[str, Any],
                                    conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Combine local and global search results"""
        
        combine_prompt = f"""
You have two different search results for the user's question. Combine them into a comprehensive, well-structured answer.

User Question: {query}

Local Search Result (specific entities and relationships):
{local_result.get('response', '')}

Global Search Result (comprehensive community analysis):
{global_result.get('response', '')}

Instructions:
1. Create a unified response that leverages both perspectives
2. Start with specific details from local search if relevant
3. Expand with broader context from global search
4. Eliminate redundancy while preserving unique insights from both
5. Maintain a logical flow and structure
6. Acknowledge when the two approaches provide complementary information

Generate a comprehensive combined response:
"""
        
        try:
            combined_response = await self.local_search.gemini_provider.generate([
                {"role": "user", "content": combine_prompt}
            ])
            
            return {
                "response": combined_response,
                "entities": local_result.get("entities", []),
                "relationships": local_result.get("relationships", []),
                "communities": global_result.get("communities", []),
                "community_reports": global_result.get("community_reports", []),
                "text_units": local_result.get("text_units", []),
                "search_type": "hybrid",
                "context_stats": {
                    "local_entities": len(local_result.get("entities", [])),
                    "local_relationships": len(local_result.get("relationships", [])),
                    "global_communities": len(global_result.get("communities", [])),
                    "combined_response_generated": True,
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to combine search results: {e}")
            # Fallback: return local result with global communities
            result = local_result.copy()
            result["communities"] = global_result.get("communities", [])
            result["search_type"] = "hybrid_fallback"
            return result

# Factory functions
def create_search_engines(data_dir: str, gemini_provider: GeminiLLMProvider) -> Tuple[LocalSearch, GlobalSearch, HybridSearch]:
    """Create all search engines"""
    local_search = LocalSearch(data_dir, gemini_provider)
    global_search = GlobalSearch(data_dir, gemini_provider)
    hybrid_search = HybridSearch(local_search, global_search)
    
    return local_search, global_search, hybrid_search

# CLI interface for testing
async def main():
    """CLI interface for testing search engines"""
    import argparse
    from gemini_llm_provider import create_gemini_llm
    
    parser = argparse.ArgumentParser(description="GraphRAG Search Test")
    parser.add_argument("--data-dir", default="./output", help="Data directory")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--mode", choices=["local", "global", "hybrid"], default="hybrid", help="Search mode")
    
    args = parser.parse_args()
    
    # Create Gemini provider
    gemini_provider = create_gemini_llm({
        "api_key": os.getenv("GEMINI_API_KEY"),
        "model": "gemini-2.0-flash-exp",  # Fast model for search
    })
    
    # Create search engines
    local_search, global_search, hybrid_search = create_search_engines(args.data_dir, gemini_provider)
    
    # Perform search
    if args.mode == "local":
        result = await local_search.search(args.query)
    elif args.mode == "global":
        result = await global_search.search(args.query)
    else:
        result = await hybrid_search.search(args.query)
    
    # Display results
    print(f"\n=== {args.mode.upper()} SEARCH RESULTS ===")
    print(f"Query: {args.query}")
    print(f"Response: {result['response']}")
    print(f"Search Strategy: {result.get('search_strategy', args.mode)}")
    print(f"Context Stats: {result.get('context_stats', {})}")

if __name__ == "__main__":
    asyncio.run(main())
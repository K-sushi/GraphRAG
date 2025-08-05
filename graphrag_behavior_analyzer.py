#!/usr/bin/env python3
"""
GraphRAG Behavior Analysis Tool
Deep inspection of GraphRAG internal processes and decision-making
"""

import os
import sys
import asyncio
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import traceback
from dataclasses import dataclass, asdict
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Load environment variables
def load_env():
    env_file = ".env"
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env()

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('graphrag_behavior.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessStep:
    """Represents a single step in the GraphRAG process"""
    step_name: str
    start_time: float
    end_time: Optional[float] = None
    input_data: Any = None
    output_data: Any = None
    metadata: Dict[str, Any] = None
    error: Optional[str] = None
    
    @property
    def duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return 0.0

@dataclass
class GraphRAGTrace:
    """Complete trace of GraphRAG execution"""
    trace_id: str
    query: str
    start_time: float
    end_time: Optional[float] = None
    steps: List[ProcessStep] = None
    final_result: Any = None
    performance_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.steps is None:
            self.steps = []

class GraphRAGBehaviorAnalyzer:
    """Deep analysis tool for GraphRAG behavior"""
    
    def __init__(self):
        self.traces: List[GraphRAGTrace] = []
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.current_trace: Optional[GraphRAGTrace] = None
        
        # Analysis configurations
        self.detailed_logging = True
        self.save_intermediate_results = True
        self.visualize_graphs = True
        
        # Create analysis directories
        self.analysis_dir = Path("./analysis")
        self.analysis_dir.mkdir(exist_ok=True)
        (self.analysis_dir / "traces").mkdir(exist_ok=True)
        (self.analysis_dir / "graphs").mkdir(exist_ok=True)
        (self.analysis_dir / "metrics").mkdir(exist_ok=True)
    
    def start_trace(self, query: str) -> str:
        """Start a new execution trace"""
        trace_id = f"trace_{int(time.time())}_{hash(query) % 10000}"
        
        self.current_trace = GraphRAGTrace(
            trace_id=trace_id,
            query=query,
            start_time=time.time(),
            steps=[],
            performance_metrics={}
        )
        
        logger.info(f"[TRACE] Starting trace: {trace_id}")
        logger.info(f"[QUERY] Query: {query}")
        
        return trace_id
    
    def log_step(self, step_name: str, input_data: Any = None, metadata: Dict[str, Any] = None):
        """Log the start of a processing step"""
        if not self.current_trace:
            return
        
        step = ProcessStep(
            step_name=step_name,
            start_time=time.time(),
            input_data=self._serialize_data(input_data),
            metadata=metadata or {}
        )
        
        self.current_trace.steps.append(step)
        logger.debug(f"[STEP] Step started: {step_name}")
        
        if input_data and self.detailed_logging:
            logger.debug(f"[INPUT] Input: {str(input_data)[:200]}...")
    
    def complete_step(self, output_data: Any = None, error: str = None):
        """Complete the current processing step"""
        if not self.current_trace or not self.current_trace.steps:
            return
        
        current_step = self.current_trace.steps[-1]
        current_step.end_time = time.time()
        current_step.output_data = self._serialize_data(output_data)
        current_step.error = error
        
        duration = current_step.duration
        
        if error:
            logger.error(f"[ERROR] Step failed: {current_step.step_name} ({duration:.2f}s) - {error}")
        else:
            logger.debug(f"[OK] Step completed: {current_step.step_name} ({duration:.2f}s)")
            
            if output_data and self.detailed_logging:
                logger.debug(f"[OUTPUT] Output: {str(output_data)[:200]}...")
    
    def complete_trace(self, final_result: Any = None):
        """Complete the current execution trace"""
        if not self.current_trace:
            return
        
        self.current_trace.end_time = time.time()
        self.current_trace.final_result = self._serialize_data(final_result)
        
        # Calculate performance metrics
        total_duration = self.current_trace.end_time - self.current_trace.start_time
        step_durations = {step.step_name: step.duration for step in self.current_trace.steps}
        
        self.current_trace.performance_metrics = {
            "total_duration": total_duration,
            "step_count": len(self.current_trace.steps),
            "step_durations": step_durations,
            "average_step_duration": sum(step_durations.values()) / len(step_durations) if step_durations else 0,
            "longest_step": max(step_durations.items(), key=lambda x: x[1]) if step_durations else None,
            "failed_steps": sum(1 for step in self.current_trace.steps if step.error),
        }
        
        # Save trace
        self.traces.append(self.current_trace)
        self._save_trace(self.current_trace)
        
        logger.info(f"[COMPLETE] Trace completed: {self.current_trace.trace_id}")
        logger.info(f"[TIME] Total duration: {total_duration:.2f}s")
        logger.info(f"[STATS] Steps: {len(self.current_trace.steps)} ({self.current_trace.performance_metrics['failed_steps']} failed)")
        
        self.current_trace = None
    
    def _serialize_data(self, data: Any) -> Any:
        """Serialize data for storage"""
        if data is None:
            return None
        
        try:
            # Handle common data types
            if isinstance(data, (str, int, float, bool)):
                return data
            elif isinstance(data, (list, tuple)):
                return [self._serialize_data(item) for item in data[:10]]  # Limit to first 10 items
            elif isinstance(data, dict):
                return {k: self._serialize_data(v) for k, v in list(data.items())[:20]}  # Limit to first 20 items
            else:
                return str(data)[:500]  # Limit string length
        except Exception as e:
            return f"<Serialization error: {str(e)}>"
    
    def _save_trace(self, trace: GraphRAGTrace):
        """Save trace to file"""
        if not self.save_intermediate_results:
            return
        
        try:
            trace_file = self.analysis_dir / "traces" / f"{trace.trace_id}.json"
            with open(trace_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(trace), f, indent=2, ensure_ascii=False)
            
            logger.debug(f"[SAVED] Trace saved: {trace_file}")
        except Exception as e:
            logger.error(f"Failed to save trace: {e}")
    
    async def analyze_document_processing(self, document_path: str) -> Dict[str, Any]:
        """Analyze document processing behavior"""
        trace_id = self.start_trace(f"Document processing: {document_path}")
        
        try:
            # Step 1: Document loading
            self.log_step("document_loading", {"path": document_path})
            
            if not os.path.exists(document_path):
                self.complete_step(error="Document not found")
                return {}
            
            with open(document_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            doc_stats = {
                "length": len(content),
                "word_count": len(content.split()),
                "line_count": len(content.splitlines())
            }
            
            self.complete_step(doc_stats)
            
            # Step 2: Text chunking analysis
            self.log_step("text_chunking", {"content_length": len(content)})
            
            chunks = self._analyze_chunking(content)
            chunk_stats = {
                "chunk_count": len(chunks),
                "avg_chunk_size": sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0,
                "chunk_sizes": [len(chunk) for chunk in chunks]
            }
            
            self.complete_step(chunk_stats)
            
            # Step 3: Entity extraction analysis
            self.log_step("entity_extraction", {"chunk_count": len(chunks)})
            
            all_entities = []
            for i, chunk in enumerate(chunks):
                entities = await self._analyze_entity_extraction(chunk, i)
                all_entities.extend(entities)
            
            entity_stats = {
                "total_entities": len(all_entities),
                "unique_entities": len(set(e["name"] for e in all_entities)),
                "entity_types": {},
                "entities_per_chunk": len(all_entities) / len(chunks) if chunks else 0
            }
            
            # Count entity types
            for entity in all_entities:
                entity_type = entity.get("type", "unknown")
                entity_stats["entity_types"][entity_type] = entity_stats["entity_types"].get(entity_type, 0) + 1
            
            self.complete_step(entity_stats)
            
            # Step 4: Relationship analysis
            self.log_step("relationship_analysis", {"entity_count": len(all_entities)})
            
            relationships = await self._analyze_relationships(chunks, all_entities)
            
            relationship_stats = {
                "total_relationships": len(relationships),
                "relationship_types": {},
                "avg_relationships_per_entity": len(relationships) / len(all_entities) if all_entities else 0
            }
            
            for rel in relationships:
                rel_type = rel.get("relationship", "unknown")
                relationship_stats["relationship_types"][rel_type] = relationship_stats["relationship_types"].get(rel_type, 0) + 1
            
            self.complete_step(relationship_stats)
            
            # Step 5: Graph construction analysis
            self.log_step("graph_construction", {
                "entities": len(all_entities),
                "relationships": len(relationships)
            })
            
            graph_stats = self._analyze_graph_construction(all_entities, relationships)
            self.complete_step(graph_stats)
            
            # Final results
            final_result = {
                "document_stats": doc_stats,
                "chunk_stats": chunk_stats,
                "entity_stats": entity_stats,
                "relationship_stats": relationship_stats,
                "graph_stats": graph_stats,
                "entities": all_entities,
                "relationships": relationships
            }
            
            self.complete_trace(final_result)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Document processing analysis failed: {e}")
            logger.error(traceback.format_exc())
            self.complete_step(error=str(e))
            self.complete_trace()
            return {}
    
    def _analyze_chunking(self, content: str) -> List[str]:
        """Analyze text chunking behavior"""
        # Simple chunking for analysis
        chunk_size = 1200
        overlap = 100
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk = content[start:end]
            chunks.append(chunk)
            
            if end >= len(content):
                break
            
            start = end - overlap
        
        logger.debug(f"[CHUNKS] Created {len(chunks)} chunks with {chunk_size} chars and {overlap} overlap")
        
        return chunks
    
    async def _analyze_entity_extraction(self, chunk: str, chunk_id: int) -> List[Dict[str, Any]]:
        """Analyze entity extraction from a single chunk"""
        if not self.api_key:
            # Fallback to simple extraction
            return self._simple_entity_extraction(chunk, chunk_id)
        
        try:
            import aiohttp
            
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }
            
            prompt = f"""
            Extract entities from this text chunk. For each entity, provide:
            1. Name of the entity
            2. Type (person, organization, location, event, concept, technology)
            3. Brief description
            4. Importance score (1-10)
            
            Text chunk {chunk_id}: {chunk}
            
            Return ONLY a JSON array:
            [{{"name": "Entity Name", "type": "entity_type", "description": "brief desc", "importance": 7}}]
            """
            
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 1000
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "candidates" in data and len(data["candidates"]) > 0:
                            content = data["candidates"][0]["content"]["parts"][0]["text"]
                            
                            try:
                                entities = json.loads(content.strip())
                                if isinstance(entities, list):
                                    # Add chunk metadata
                                    for entity in entities:
                                        entity["chunk_id"] = chunk_id
                                        entity["extraction_method"] = "gemini"
                                    
                                    logger.debug(f"[GEMINI] Extracted {len(entities)} entities from chunk {chunk_id}")
                                    return entities
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse Gemini response for chunk {chunk_id}")
            
        except Exception as e:
            logger.warning(f"Gemini entity extraction failed for chunk {chunk_id}: {e}")
        
        # Fallback to simple extraction
        return self._simple_entity_extraction(chunk, chunk_id)
    
    def _simple_entity_extraction(self, chunk: str, chunk_id: int) -> List[Dict[str, Any]]:
        """Simple fallback entity extraction"""
        import re
        
        # Find capitalized words/phrases
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(pattern, chunk)
        
        # Filter common words
        stop_words = {
            "The", "This", "That", "These", "Those", "And", "But", "Or", 
            "For", "With", "By", "In", "On", "At", "To", "From", "Of",
            "It", "Is", "Are", "Was", "Were", "Be", "Been", "Being"
        }
        
        entities = []
        for match in matches:
            if match not in stop_words:
                entities.append({
                    "name": match,
                    "type": "unknown",
                    "description": f"Entity found in chunk {chunk_id}",
                    "importance": 5,
                    "chunk_id": chunk_id,
                    "extraction_method": "simple"
                })
        
        # Remove duplicates
        unique_entities = []
        seen = set()
        for entity in entities:
            if entity["name"] not in seen:
                seen.add(entity["name"])
                unique_entities.append(entity)
        
        logger.debug(f"[SIMPLE] Simple extraction: {len(unique_entities)} entities from chunk {chunk_id}")
        return unique_entities[:10]  # Limit to top 10
    
    async def _analyze_relationships(self, chunks: List[str], entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze relationship extraction"""
        relationships = []
        
        # Group entities by chunk
        chunk_entities = {}
        for entity in entities:
            chunk_id = entity.get("chunk_id", 0)
            if chunk_id not in chunk_entities:
                chunk_entities[chunk_id] = []
            chunk_entities[chunk_id].append(entity)
        
        # Analyze relationships within each chunk
        for chunk_id, chunk_ents in chunk_entities.items():
            if len(chunk_ents) >= 2:  # Need at least 2 entities
                chunk_text = chunks[chunk_id] if chunk_id < len(chunks) else ""
                chunk_rels = await self._extract_chunk_relationships(chunk_text, chunk_ents, chunk_id)
                relationships.extend(chunk_rels)
        
        return relationships
    
    async def _extract_chunk_relationships(self, chunk_text: str, entities: List[Dict[str, Any]], chunk_id: int) -> List[Dict[str, Any]]:
        """Extract relationships from a single chunk"""
        if len(entities) < 2:
            return []
        
        # Simple co-occurrence based relationships
        relationships = []
        entity_names = [e["name"] for e in entities]
        
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                # Check if both entities appear in the text
                if entity1["name"] in chunk_text and entity2["name"] in chunk_text:
                    relationships.append({
                        "source": entity1["name"],
                        "target": entity2["name"],
                        "relationship": "co_occurs",
                        "description": f"Co-occur in chunk {chunk_id}",
                        "strength": 0.5,
                        "chunk_id": chunk_id,
                        "extraction_method": "co_occurrence"
                    })
        
        logger.debug(f"[RELATIONS] Found {len(relationships)} relationships in chunk {chunk_id}")
        return relationships
    
    def _analyze_graph_construction(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze graph construction process"""
        try:
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add nodes (entities)
            for entity in entities:
                G.add_node(entity["name"], **entity)
            
            # Add edges (relationships)
            for rel in relationships:
                if rel["source"] in G.nodes and rel["target"] in G.nodes:
                    G.add_edge(rel["source"], rel["target"], **rel)
            
            # Analyze graph properties
            graph_stats = {
                "node_count": G.number_of_nodes(),
                "edge_count": G.number_of_edges(),
                "density": nx.density(G),
                "connected_components": nx.number_connected_components(G),
                "largest_component_size": len(max(nx.connected_components(G), key=len)) if G.nodes else 0,
                "average_degree": sum(dict(G.degree()).values()) / G.number_of_nodes() if G.nodes else 0,
                "clustering_coefficient": nx.average_clustering(G) if G.nodes else 0
            }
            
            # Community detection if possible
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(G)
                communities = len(set(partition.values()))
                graph_stats["communities_detected"] = communities
                graph_stats["modularity"] = community_louvain.modularity(partition, G)
            except ImportError:
                # Use connected components as fallback
                graph_stats["communities_detected"] = nx.number_connected_components(G)
                graph_stats["modularity"] = None
            
            # Visualize graph if requested
            if self.visualize_graphs and G.nodes:
                self._visualize_graph(G)
            
            logger.debug(f"[GRAPH] Graph stats: {graph_stats['node_count']} nodes, {graph_stats['edge_count']} edges, {graph_stats['communities_detected']} communities")
            
            return graph_stats
            
        except Exception as e:
            logger.error(f"Graph analysis failed: {e}")
            return {
                "node_count": len(entities),
                "edge_count": len(relationships),
                "error": str(e)
            }
    
    def _visualize_graph(self, G: nx.Graph):
        """Create graph visualization"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Use spring layout for better visualization
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Draw nodes with different colors for different entity types
            node_colors = []
            for node in G.nodes():
                node_data = G.nodes[node]
                entity_type = node_data.get("type", "unknown")
                color_map = {
                    "person": "lightblue",
                    "organization": "lightgreen",
                    "location": "lightcoral",
                    "technology": "lightyellow",
                    "concept": "lightpink",
                    "unknown": "lightgray"
                }
                node_colors.append(color_map.get(entity_type, "lightgray"))
            
            # Draw the graph
            nx.draw(G, pos, 
                   node_color=node_colors,
                   node_size=300,
                   with_labels=True,
                   font_size=8,
                   font_weight="bold",
                   edge_color="gray",
                   alpha=0.7)
            
            plt.title("GraphRAG Knowledge Graph Structure", fontsize=16, fontweight="bold")
            plt.axis('off')
            plt.tight_layout()
            
            # Save visualization
            viz_file = self.analysis_dir / "graphs" / f"graph_{int(time.time())}.png"
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"[VIZ] Graph visualization saved: {viz_file}")
            
        except Exception as e:
            logger.error(f"Graph visualization failed: {e}")
    
    async def analyze_query_processing(self, query: str) -> Dict[str, Any]:
        """Analyze query processing behavior"""
        trace_id = self.start_trace(f"Query processing: {query}")
        
        try:
            # Step 1: Query analysis
            self.log_step("query_analysis", {"query": query})
            
            query_stats = {
                "length": len(query),
                "word_count": len(query.split()),
                "question_words": [w for w in query.lower().split() if w in ["what", "how", "why", "when", "where", "who"]],
                "complexity_score": min(len(query.split()) / 10, 1.0)
            }
            
            self.complete_step(query_stats)
            
            # Step 2: Entity extraction from query
            self.log_step("query_entity_extraction")
            
            query_entities = await self._analyze_entity_extraction(query, -1)
            self.complete_step({"entities_found": len(query_entities), "entities": query_entities})
            
            # Step 3: Document relevance analysis
            self.log_step("document_relevance")
            
            documents = self._load_documents()
            relevance_scores = self._analyze_document_relevance(query, documents)
            
            self.complete_step({
                "documents_analyzed": len(documents),
                "relevant_documents": sum(1 for score in relevance_scores.values() if score > 0),
                "relevance_scores": relevance_scores
            })
            
            # Step 4: Context building
            self.log_step("context_building")
            
            context = self._build_query_context(query, documents, relevance_scores)
            
            self.complete_step({
                "context_length": len(context),
                "context_sources": len([d for d, score in relevance_scores.items() if score > 0])
            })
            
            # Step 5: Response generation
            self.log_step("response_generation", {"context_length": len(context)})
            
            if self.api_key:
                response = await self._generate_response(query, context)
            else:
                response = "Mock response - Gemini API not available"
            
            response_stats = {
                "response_length": len(response),
                "response_word_count": len(response.split())
            }
            
            self.complete_step(response_stats)
            
            # Final results
            final_result = {
                "query_stats": query_stats,
                "query_entities": query_entities,
                "document_relevance": relevance_scores,
                "context": context,
                "response": response,
                "response_stats": response_stats
            }
            
            self.complete_trace(final_result)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Query processing analysis failed: {e}")
            logger.error(traceback.format_exc())
            self.complete_step(error=str(e))
            self.complete_trace()
            return {}
    
    def _load_documents(self) -> List[Dict[str, Any]]:
        """Load documents for analysis"""
        documents = []
        input_dir = Path("./input")
        
        if input_dir.exists():
            for file_path in input_dir.glob("*.txt"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    documents.append({
                        "id": file_path.stem,
                        "title": file_path.name,
                        "content": content,
                        "path": str(file_path),
                        "length": len(content)
                    })
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
        
        return documents
    
    def _analyze_document_relevance(self, query: str, documents: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze document relevance to query"""
        query_words = set(query.lower().split())
        relevance_scores = {}
        
        for doc in documents:
            content_words = set(doc["content"].lower().split())
            
            # Simple word overlap scoring
            overlap = len(query_words.intersection(content_words))
            total_query_words = len(query_words)
            
            relevance_score = overlap / total_query_words if total_query_words > 0 else 0
            relevance_scores[doc["id"]] = relevance_score
        
        return relevance_scores
    
    def _build_query_context(self, query: str, documents: List[Dict[str, Any]], relevance_scores: Dict[str, float]) -> str:
        """Build context for query"""
        # Get top 3 relevant documents
        sorted_docs = sorted(documents, key=lambda d: relevance_scores.get(d["id"], 0), reverse=True)
        top_docs = sorted_docs[:3]
        
        context_parts = []
        for doc in top_docs:
            if relevance_scores.get(doc["id"], 0) > 0:
                excerpt = doc["content"][:300] + "..." if len(doc["content"]) > 300 else doc["content"]
                context_parts.append(f"From {doc['title']}: {excerpt}")
        
        return "\n\n".join(context_parts)
    
    async def _generate_response(self, query: str, context: str) -> str:
        """Generate response using Gemini"""
        try:
            import aiohttp
            
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-002:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }
            
            prompt = f"""
            Context: {context}
            
            User Query: {query}
            
            Please provide a comprehensive answer based on the context provided.
            """
            
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 1500
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "candidates" in data and len(data["candidates"]) > 0:
                            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            
            return "Error: Failed to generate response"
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"Error: {str(e)}"
    
    def generate_behavior_report(self) -> Dict[str, Any]:
        """Generate comprehensive behavior analysis report"""
        if not self.traces:
            return {"error": "No traces available"}
        
        report = {
            "summary": {
                "total_traces": len(self.traces),
                "successful_traces": sum(1 for t in self.traces if not any(s.error for s in t.steps)),
                "failed_traces": sum(1 for t in self.traces if any(s.error for s in t.steps)),
                "average_duration": sum(t.end_time - t.start_time for t in self.traces if t.end_time) / len(self.traces),
            },
            "step_analysis": {},
            "performance_patterns": {},
            "error_analysis": {},
            "recommendations": []
        }
        
        # Analyze steps
        all_steps = [step for trace in self.traces for step in trace.steps]
        step_names = set(step.step_name for step in all_steps)
        
        for step_name in step_names:
            step_instances = [s for s in all_steps if s.step_name == step_name]
            successful_steps = [s for s in step_instances if not s.error]
            
            report["step_analysis"][step_name] = {
                "total_instances": len(step_instances),
                "successful_instances": len(successful_steps),
                "failure_rate": (len(step_instances) - len(successful_steps)) / len(step_instances) * 100,
                "average_duration": sum(s.duration for s in successful_steps) / len(successful_steps) if successful_steps else 0,
                "common_errors": [s.error for s in step_instances if s.error]
            }
        
        # Save report
        report_file = self.analysis_dir / "behavior_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[REPORT] Behavior report saved: {report_file}")
        
        return report

# CLI interface
async def main():
    """Main analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GraphRAG Behavior Analyzer")
    parser.add_argument("--mode", choices=[
        "document", "query", "full", "report"
    ], default="full", help="Analysis mode")
    parser.add_argument("--document", help="Document to analyze")
    parser.add_argument("--query", help="Query to analyze")
    parser.add_argument("--output", help="Output directory")
    
    args = parser.parse_args()
    
    analyzer = GraphRAGBehaviorAnalyzer()
    
    if args.output:
        analyzer.analysis_dir = Path(args.output)
        analyzer.analysis_dir.mkdir(exist_ok=True)
    
    print("GraphRAG Behavior Analyzer")
    print("=" * 60)
    
    try:
        if args.mode == "document":
            if not args.document:
                print("Error: --document required for document analysis")
                return 1
            
            print(f"[ANALYZE] Analyzing document: {args.document}")
            result = await analyzer.analyze_document_processing(args.document)
            print(f"[OK] Analysis complete. Results saved to {analyzer.analysis_dir}")
            
        elif args.mode == "query":
            if not args.query:
                print("Error: --query required for query analysis")
                return 1
            
            print(f"[ANALYZE] Analyzing query: {args.query}")
            result = await analyzer.analyze_query_processing(args.query)
            print(f"[OK] Analysis complete. Results saved to {analyzer.analysis_dir}")
            
        elif args.mode == "full":
            print("[ANALYZE] Running full analysis...")
            
            # Analyze all documents
            input_dir = Path("./input")
            if input_dir.exists():
                for doc_path in input_dir.glob("*.txt"):
                    print(f"[DOC] Analyzing document: {doc_path.name}")
                    await analyzer.analyze_document_processing(str(doc_path))
            
            # Analyze sample queries
            sample_queries = [
                "What is artificial intelligence?",
                "How does blockchain technology work?",  
                "Explain quantum computing applications"
            ]
            
            for query in sample_queries:
                print(f"[QUERY] Analyzing query: {query}")
                await analyzer.analyze_query_processing(query)
            
            print("[OK] Full analysis complete")
            
        elif args.mode == "report":
            print("[REPORT] Generating behavior report...")
            
        # Generate final report
        report = analyzer.generate_behavior_report()
        
        print("\n[SUMMARY] ANALYSIS SUMMARY")
        print("-" * 40)
        print(f"Total traces: {report['summary']['total_traces']}")
        print(f"Successful: {report['summary']['successful_traces']}")
        print(f"Failed: {report['summary']['failed_traces']}")
        print(f"Average duration: {report['summary']['average_duration']:.2f}s")
        
        print(f"\n[SAVED] All results saved to: {analyzer.analysis_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        return 1
    except Exception as e:
        print(f"Analysis failed: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
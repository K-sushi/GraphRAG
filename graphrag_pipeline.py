#!/usr/bin/env python3
"""
Microsoft GraphRAG Complete Pipeline Implementation
SuperClaude Wave Orchestration - Phase 2

Full GraphRAG indexing and search pipeline with Gemini integration
"""

import os
import sys
import asyncio
import logging
import pandas as pd
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from datetime import datetime
import json
import yaml
from dataclasses import dataclass, asdict
import tiktoken

# Vector and ML imports
import faiss
from sentence_transformers import SentenceTransformer

# GraphRAG core imports
try:
    from graphrag.index.workflows.v1.create_base_text_units import (
        create_base_text_units,
        CreateBaseTextUnitsConfiguration,
    )
    from graphrag.index.workflows.v1.create_base_extracted_entities import (
        create_base_extracted_entities,
        CreateBaseExtractedEntitiesConfiguration,
    )
    from graphrag.index.workflows.v1.create_final_entities import (
        create_final_entities,
    )
    from graphrag.index.workflows.v1.create_final_relationships import (
        create_final_relationships,
    )
    from graphrag.index.workflows.v1.create_final_communities import (
        create_final_communities,
    )
except ImportError:
    # Fallback to manual implementation if GraphRAG not available
    logging.warning("GraphRAG modules not available, using manual implementation")

from gemini_llm_provider import GeminiLLMProvider, create_gemini_llm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GraphRAGConfig:
    """GraphRAG pipeline configuration"""
    # Data paths
    input_dir: str = "./input"
    output_dir: str = "./output"
    cache_dir: str = "./cache"
    
    # Chunking configuration
    chunk_size: int = 1200
    chunk_overlap: int = 100
    
    # Entity extraction
    entity_extraction_prompt: str = "prompts/entity_extraction.txt"
    entity_types: List[str] = None
    max_gleanings: int = 1
    
    # Community detection
    community_level: int = 2
    use_community_summary: bool = True
    
    # Search configuration
    local_search_text_unit_prop: float = 0.5
    local_search_community_prop: float = 0.1
    local_search_top_k_entities: int = 10
    local_search_top_k_relationships: int = 10
    local_search_max_tokens: int = 12000
    
    # Global search
    global_search_max_tokens: int = 12000
    global_search_data_max_tokens: int = 12000
    global_search_map_max_tokens: int = 1000
    global_search_reduce_max_tokens: int = 2000
    global_search_concurrency: int = 32
    
    # Performance settings
    max_workers: int = 4
    batch_size: int = 10
    
    def __post_init__(self):
        if self.entity_types is None:
            self.entity_types = ["organization", "person", "geo", "event"]

class GraphRAGPipeline:
    """Complete GraphRAG implementation with Gemini integration"""
    
    def __init__(self, config: GraphRAGConfig, gemini_provider: GeminiLLMProvider):
        self.config = config
        self.gemini_provider = gemini_provider
        
        # Initialize paths
        self.input_path = Path(config.input_dir)
        self.output_path = Path(config.output_dir)
        self.cache_path = Path(config.cache_dir)
        
        # Create directories
        self.input_path.mkdir(exist_ok=True)
        self.output_path.mkdir(exist_ok=True)
        self.cache_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.embedding_model = None
        self.vector_store = None
        
        # Data storage
        self.documents = []
        self.text_units = []
        self.entities = []
        self.relationships = []
        self.communities = []
        self.community_reports = []
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
        logger.info("GraphRAG Pipeline initialized")
    
    def _initialize_embedding_model(self):
        """Initialize embedding model"""
        try:
            # Use a fast, efficient embedding model
            model_name = "all-MiniLM-L6-v2"  # Fast and good quality
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Initialized embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    async def run_complete_pipeline(self, documents: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run the complete GraphRAG pipeline"""
        logger.info("Starting complete GraphRAG pipeline...")
        
        start_time = datetime.utcnow()
        
        try:
            # Step 1: Load or use provided documents
            if documents:
                self.documents = documents
            else:
                self.documents = await self._load_documents()
            
            logger.info(f"Loaded {len(self.documents)} documents")
            
            # Step 2: Create text units (chunking)
            self.text_units = await self._create_text_units()
            logger.info(f"Created {len(self.text_units)} text units")
            
            # Step 3: Extract entities
            self.entities = await self._extract_entities()
            logger.info(f"Extracted {len(self.entities)} entities")
            
            # Step 4: Extract relationships
            self.relationships = await self._extract_relationships()
            logger.info(f"Extracted {len(self.relationships)} relationships")
            
            # Step 5: Detect communities
            self.communities = await self._detect_communities()
            logger.info(f"Detected {len(self.communities)} communities")
            
            # Step 6: Generate community reports
            self.community_reports = await self._generate_community_reports()
            logger.info(f"Generated {len(self.community_reports)} community reports")
            
            # Step 7: Create vector embeddings
            await self._create_embeddings()
            logger.info("Created vector embeddings")
            
            # Step 8: Initialize vector store
            await self._initialize_vector_store()
            logger.info("Initialized vector store")
            
            # Step 9: Save results
            await self._save_results()
            logger.info("Saved all results")
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "success": True,
                "processing_time": processing_time,
                "documents_processed": len(self.documents),
                "text_units_created": len(self.text_units),
                "entities_extracted": len(self.entities),
                "relationships_extracted": len(self.relationships),
                "communities_detected": len(self.communities),
                "community_reports_generated": len(self.community_reports),
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    async def _load_documents(self) -> List[Dict[str, Any]]:
        """Load documents from input directory"""
        documents = []
        
        for file_path in self.input_path.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if content:
                    documents.append({
                        "id": file_path.stem,
                        "title": file_path.name,
                        "content": content,
                        "source": str(file_path),
                        "length": len(content),
                        "tokens": len(self.tokenizer.encode(content)),
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        return documents
    
    async def _create_text_units(self) -> List[Dict[str, Any]]:
        """Create text units (chunks) from documents"""
        text_units = []
        unit_id = 0
        
        for doc in self.documents:
            content = doc["content"]
            
            # Simple chunking strategy
            chunks = self._chunk_text(content, self.config.chunk_size, self.config.chunk_overlap)
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    text_units.append({
                        "id": f"unit_{unit_id}",
                        "text": chunk,
                        "document_id": doc["id"],
                        "chunk_index": i,
                        "n_tokens": len(self.tokenizer.encode(chunk)),
                        "document_ids": [doc["id"]],  # GraphRAG format
                    })
                    unit_id += 1
        
        return text_units
    
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Chunk text into overlapping segments"""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            if end >= len(tokens):
                break
            
            start = end - overlap
        
        return chunks
    
    async def _extract_entities(self) -> List[Dict[str, Any]]:
        """Extract entities from text units using Gemini"""
        entities = []
        entity_id = 0
        
        # Entity extraction prompt
        extraction_prompt = """
Extract entities from the following text. For each entity, identify:
1. Name of the entity
2. Type (person, organization, location, event, concept)
3. Description (brief explanation of what this entity is)
4. Importance (1-10 scale)

Text: {text}

Return the results in JSON format:
[
  {
    "name": "Entity Name",
    "type": "entity_type",
    "description": "Brief description",
    "importance": 7
  }
]
"""
        
        # Process text units in batches
        for i in range(0, len(self.text_units), self.config.batch_size):
            batch = self.text_units[i:i + self.config.batch_size]
            
            for unit in batch:
                try:
                    # Generate entity extraction
                    prompt = extraction_prompt.format(text=unit["text"])
                    response = await self.gemini_provider.generate([
                        {"role": "user", "content": prompt}
                    ])
                    
                    # Parse JSON response
                    try:
                        extracted_entities = json.loads(response)
                        
                        for entity_data in extracted_entities:
                            if isinstance(entity_data, dict) and "name" in entity_data:
                                entities.append({
                                    "id": f"entity_{entity_id}",
                                    "name": entity_data.get("name", "").strip(),
                                    "type": entity_data.get("type", "unknown"),
                                    "description": entity_data.get("description", ""),
                                    "importance": entity_data.get("importance", 5),
                                    "text_unit_ids": [unit["id"]],
                                    "document_ids": [unit["document_id"]],
                                    "human_readable_id": entity_id,
                                })
                                entity_id += 1
                    
                    except json.JSONDecodeError:
                        # Fallback: simple named entity recognition
                        logger.warning(f"Failed to parse JSON for unit {unit['id']}, using fallback")
                        simple_entities = self._simple_entity_extraction(unit["text"])
                        
                        for entity_name in simple_entities:
                            entities.append({
                                "id": f"entity_{entity_id}",
                                "name": entity_name,
                                "type": "unknown",
                                "description": f"Entity extracted from {unit['document_id']}",
                                "importance": 5,
                                "text_unit_ids": [unit["id"]],
                                "document_ids": [unit["document_id"]],
                                "human_readable_id": entity_id,
                            })
                            entity_id += 1
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Entity extraction failed for unit {unit['id']}: {e}")
        
        # Deduplicate entities by name
        entities = self._deduplicate_entities(entities)
        
        return entities
    
    def _simple_entity_extraction(self, text: str) -> List[str]:
        """Simple fallback entity extraction"""
        # Very basic extraction - look for capitalized words/phrases
        import re
        
        # Find capitalized words (potential proper nouns)
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(pattern, text)
        
        # Filter out common words
        stop_words = {"The", "This", "That", "These", "Those", "And", "But", "Or", "For", "With", "By"}
        entities = [match for match in matches if match not in stop_words]
        
        return list(set(entities))  # Remove duplicates
    
    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate entities by name (case-insensitive)"""
        seen_names = {}
        deduplicated = []
        
        for entity in entities:
            name_lower = entity["name"].lower()
            if name_lower not in seen_names:
                seen_names[name_lower] = entity
                deduplicated.append(entity)
            else:
                # Merge text_unit_ids and document_ids
                existing = seen_names[name_lower]
                existing["text_unit_ids"].extend(entity["text_unit_ids"])
                existing["document_ids"].extend(entity["document_ids"])
                existing["text_unit_ids"] = list(set(existing["text_unit_ids"]))
                existing["document_ids"] = list(set(existing["document_ids"]))
        
        return deduplicated
    
    async def _extract_relationships(self) -> List[Dict[str, Any]]:
        """Extract relationships between entities using Gemini"""
        relationships = []
        relationship_id = 0
        
        # Create entity name lookup
        entity_names = [entity["name"] for entity in self.entities]
        
        relationship_prompt = """
Analyze the following text and identify relationships between entities.

Entities in the text: {entities}

Text: {text}

For each relationship you find, provide:
1. Source entity name (must be from the entity list)
2. Target entity name (must be from the entity list)  
3. Relationship type (e.g., "works_for", "located_in", "collaborates_with", "part_of")
4. Description of the relationship
5. Strength (1-10 scale)

Return the results in JSON format:
[
  {
    "source": "Source Entity Name",
    "target": "Target Entity Name", 
    "relationship": "relationship_type",
    "description": "Brief description of the relationship",
    "strength": 8
  }
]
"""
        
        # Process text units that contain multiple entities
        for unit in self.text_units:
            # Find entities that appear in this text unit
            unit_entities = [e for e in self.entities if unit["id"] in e["text_unit_ids"]]
            
            if len(unit_entities) >= 2:  # Need at least 2 entities for relationships
                try:
                    entity_names_in_unit = [e["name"] for e in unit_entities]
                    
                    prompt = relationship_prompt.format(
                        entities=", ".join(entity_names_in_unit),
                        text=unit["text"]
                    )
                    
                    response = await self.gemini_provider.generate([
                        {"role": "user", "content": prompt}
                    ])
                    
                    # Parse JSON response
                    try:
                        extracted_relationships = json.loads(response)
                        
                        for rel_data in extracted_relationships:
                            if isinstance(rel_data, dict) and "source" in rel_data and "target" in rel_data:
                                # Validate entities exist
                                source_name = rel_data.get("source", "").strip()
                                target_name = rel_data.get("target", "").strip()
                                
                                if source_name in entity_names_in_unit and target_name in entity_names_in_unit:
                                    relationships.append({
                                        "id": f"relationship_{relationship_id}",
                                        "source": source_name,
                                        "target": target_name,
                                        "relationship": rel_data.get("relationship", "related_to"),
                                        "description": rel_data.get("description", ""),
                                        "weight": rel_data.get("strength", 5) / 10.0,  # Normalize to 0-1
                                        "text_unit_ids": [unit["id"]],
                                        "document_ids": [unit["document_id"]],
                                        "human_readable_id": relationship_id,
                                    })
                                    relationship_id += 1
                    
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse relationships for unit {unit['id']}")
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Relationship extraction failed for unit {unit['id']}: {e}")
        
        return relationships
    
    async def _detect_communities(self) -> List[Dict[str, Any]]:
        """Detect communities using network analysis"""
        if not self.entities or not self.relationships:
            return []
        
        try:
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add nodes (entities)
            for entity in self.entities:
                G.add_node(entity["name"], **entity)
            
            # Add edges (relationships)
            for rel in self.relationships:
                if rel["source"] in G.nodes and rel["target"] in G.nodes:
                    G.add_edge(rel["source"], rel["target"], weight=rel["weight"], **rel)
            
            # Detect communities using Louvain algorithm
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(G)
            except ImportError:
                # Fallback to connected components
                logger.warning("Community detection library not available, using connected components")
                components = list(nx.connected_components(G))
                partition = {}
                for i, component in enumerate(components):
                    for node in component:
                        partition[node] = i
            
            # Group entities by community
            communities_dict = {}
            for entity_name, community_id in partition.items():
                if community_id not in communities_dict:
                    communities_dict[community_id] = []
                communities_dict[community_id].append(entity_name)
            
            # Create community objects
            communities = []
            for community_id, entity_names in communities_dict.items():
                if len(entity_names) >= 2:  # Only include communities with multiple entities
                    communities.append({
                        "id": f"community_{community_id}",
                        "title": f"Community {community_id}",
                        "entities": entity_names,
                        "size": len(entity_names),
                        "level": 0,  # Single level for now
                        "human_readable_id": community_id,
                    })
            
            return communities
            
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            return []
    
    async def _generate_community_reports(self) -> List[Dict[str, Any]]:
        """Generate community reports using Gemini"""
        community_reports = []
        
        community_report_prompt = """
Generate a comprehensive report for a community of related entities.

Community Entities: {entities}

Related Information:
{context}

Please provide:
1. A descriptive title for this community
2. A summary of what this community represents
3. Key themes and topics
4. Important relationships within the community
5. Overall significance and impact

Format the response as a structured report.
"""
        
        for community in self.communities:
            try:
                # Gather context about community entities
                entity_names = community["entities"]
                
                # Find entities and their descriptions
                community_entities = [e for e in self.entities if e["name"] in entity_names]
                
                # Find relationships within the community
                community_relationships = [
                    r for r in self.relationships 
                    if r["source"] in entity_names and r["target"] in entity_names
                ]
                
                # Build context
                context_parts = []
                
                # Add entity descriptions
                for entity in community_entities:
                    if entity["description"]:
                        context_parts.append(f"- {entity['name']}: {entity['description']}")
                
                # Add relationship descriptions
                for rel in community_relationships:
                    if rel["description"]:
                        context_parts.append(f"- {rel['source']} {rel['relationship']} {rel['target']}: {rel['description']}")
                
                context_text = "\n".join(context_parts) if context_parts else "No additional context available."
                
                # Generate report
                prompt = community_report_prompt.format(
                    entities=", ".join(entity_names),
                    context=context_text
                )
                
                report_text = await self.gemini_provider.generate([
                    {"role": "user", "content": prompt}
                ])
                
                community_reports.append({
                    "id": f"report_{community['id']}",
                    "community_id": community["id"],
                    "title": f"Community Report: {community['title']}",
                    "content": report_text,
                    "entities": entity_names,
                    "relationships_count": len(community_relationships),
                    "rank": community["size"],  # Rank by community size
                    "human_readable_id": community["human_readable_id"],
                })
                
                # Rate limiting
                await asyncio.sleep(0.2)
                
            except Exception as e:
                logger.warning(f"Failed to generate report for community {community['id']}: {e}")
        
        return community_reports
    
    async def _create_embeddings(self):
        """Create vector embeddings for entities and text units"""
        logger.info("Creating embeddings...")
        
        # Create embeddings for text units
        text_unit_texts = [unit["text"] for unit in self.text_units]
        if text_unit_texts:
            text_unit_embeddings = self.embedding_model.encode(text_unit_texts)
            
            for i, unit in enumerate(self.text_units):
                unit["embedding"] = text_unit_embeddings[i].tolist()
        
        # Create embeddings for entities (using their descriptions)
        entity_texts = []
        for entity in self.entities:
            # Combine name and description for better embeddings
            text = f"{entity['name']}: {entity.get('description', entity['name'])}"
            entity_texts.append(text)
        
        if entity_texts:
            entity_embeddings = self.embedding_model.encode(entity_texts)
            
            for i, entity in enumerate(self.entities):
                entity["embedding"] = entity_embeddings[i].tolist()
        
        logger.info("Embeddings created successfully")
    
    async def _initialize_vector_store(self):
        """Initialize FAISS vector store"""
        if not self.text_units or "embedding" not in self.text_units[0]:
            logger.warning("No embeddings available for vector store")
            return
        
        try:
            # Get embedding dimension
            embedding_dim = len(self.text_units[0]["embedding"])
            
            # Create FAISS index
            self.vector_store = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine similarity)
            
            # Add text unit embeddings
            embeddings = np.array([unit["embedding"] for unit in self.text_units]).astype('float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            
            self.vector_store.add(embeddings)
            
            logger.info(f"Vector store initialized with {self.vector_store.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
    
    async def _save_results(self):
        """Save all results to output directory"""
        # Save as JSON files
        results = {
            "documents": self.documents,
            "text_units": self.text_units,
            "entities": self.entities,
            "relationships": self.relationships,
            "communities": self.communities,
            "community_reports": self.community_reports,
        }
        
        for name, data in results.items():
            file_path = self.output_path / f"{name}.json"
            
            # Remove embeddings for JSON serialization
            serializable_data = []
            for item in data:
                item_copy = item.copy()
                if "embedding" in item_copy:
                    del item_copy["embedding"]  # Too large for JSON
                serializable_data.append(item_copy)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        # Save vector store
        if self.vector_store:
            vector_store_path = self.cache_path / "vector_store.faiss"
            faiss.write_index(self.vector_store, str(vector_store_path))
        
        # Save pipeline metadata
        metadata = {
            "pipeline_version": "1.0.0",
            "created_at": datetime.utcnow().isoformat(),
            "config": asdict(self.config),
            "statistics": {
                "total_documents": len(self.documents),
                "total_text_units": len(self.text_units),
                "total_entities": len(self.entities),
                "total_relationships": len(self.relationships),
                "total_communities": len(self.communities),
                "total_community_reports": len(self.community_reports),
            }
        }
        
        metadata_path = self.output_path / "pipeline_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("All results saved successfully")

# Factory function for easy initialization
def create_graphrag_pipeline(config_dict: Optional[Dict[str, Any]] = None) -> GraphRAGPipeline:
    """Create GraphRAG pipeline with configuration"""
    # Load configuration
    if config_dict is None:
        config_dict = {}
    
    config = GraphRAGConfig(**config_dict)
    
    # Create Gemini provider
    gemini_provider = create_gemini_llm({
        "api_key": os.getenv("GEMINI_API_KEY"),
        "model": os.getenv("GRAPHRAG_LLM_MODEL", "gemini-1.5-pro-002"),
        "requests_per_minute": int(os.getenv("GRAPHRAG_REQUESTS_PER_MINUTE", "10000")),
        "tokens_per_minute": int(os.getenv("GRAPHRAG_TOKENS_PER_MINUTE", "150000")),
        "concurrent_requests": int(os.getenv("GRAPHRAG_CONCURRENT_REQUESTS", "25")),
    })
    
    return GraphRAGPipeline(config, gemini_provider)

# CLI interface for testing
async def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GraphRAG Pipeline")
    parser.add_argument("--input-dir", default="./input", help="Input directory")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Chunk size")
    
    args = parser.parse_args()
    
    # Create configuration
    config = GraphRAGConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
    )
    
    # Create and run pipeline
    pipeline = create_graphrag_pipeline(asdict(config))
    
    try:
        result = await pipeline.run_complete_pipeline()
        print(f"Pipeline completed successfully: {result}")
    except Exception as e:
        print(f"Pipeline failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
#!/usr/bin/env python3
"""
Create sample graph data for GraphRAG visualization demo
GraphRAG視覚化デモ用のサンプルグラフデータ作成
"""

import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
import json

def create_sample_graphrag_data(output_dir="./output"):
    """Create sample GraphRAG-style data for demonstration"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create sample knowledge graph
    G = nx.Graph()
    
    # Sample entities with different types
    entities = [
        {"id": "person_1", "name": "Alice Johnson", "type": "Person", "description": "Data scientist at TechCorp"},
        {"id": "person_2", "name": "Bob Smith", "type": "Person", "description": "Machine learning engineer"},
        {"id": "person_3", "name": "Carol Davis", "type": "Person", "description": "AI researcher"},
        {"id": "company_1", "name": "TechCorp", "type": "Organization", "description": "Technology company specializing in AI"},
        {"id": "company_2", "name": "DataInc", "type": "Organization", "description": "Data analytics company"},
        {"id": "project_1", "name": "GraphRAG System", "type": "Project", "description": "Knowledge graph-based RAG system"},
        {"id": "project_2", "name": "Bitcoin Analysis", "type": "Project", "description": "Cryptocurrency price prediction"},
        {"id": "tech_1", "name": "Machine Learning", "type": "Technology", "description": "AI technique for pattern recognition"},
        {"id": "tech_2", "name": "Knowledge Graphs", "type": "Technology", "description": "Graph-based knowledge representation"},
        {"id": "tech_3", "name": "Natural Language Processing", "type": "Technology", "description": "AI for text processing"},
        {"id": "concept_1", "name": "Retrieval Augmented Generation", "type": "Concept", "description": "RAG methodology"},
        {"id": "concept_2", "name": "Large Language Models", "type": "Concept", "description": "LLM technology"},
    ]
    
    # Sample relationships
    relationships = [
        {"source": "person_1", "target": "company_1", "description": "works at", "weight": 1.0},
        {"source": "person_2", "target": "company_2", "description": "employed by", "weight": 1.0},
        {"source": "person_3", "target": "project_1", "description": "leads", "weight": 0.9},
        {"source": "person_1", "target": "project_1", "description": "contributes to", "weight": 0.8},
        {"source": "person_2", "target": "project_2", "description": "developed", "weight": 0.9},
        {"source": "project_1", "target": "tech_2", "description": "uses", "weight": 0.8},
        {"source": "project_1", "target": "concept_1", "description": "implements", "weight": 0.9},
        {"source": "project_2", "target": "tech_1", "description": "applies", "weight": 0.7},
        {"source": "tech_1", "target": "concept_2", "description": "related to", "weight": 0.6},
        {"source": "tech_3", "target": "concept_2", "description": "component of", "weight": 0.8},
        {"source": "concept_1", "target": "concept_2", "description": "utilizes", "weight": 0.9},
        {"source": "company_1", "target": "tech_2", "description": "researches", "weight": 0.7},
        {"source": "person_1", "target": "person_3", "description": "collaborates with", "weight": 0.8},
        {"source": "person_2", "target": "person_3", "description": "knows", "weight": 0.6},
    ]
    
    # Add nodes to graph
    for entity in entities:
        G.add_node(entity["id"], **entity)
    
    # Add edges to graph
    for rel in relationships:
        G.add_edge(rel["source"], rel["target"], 
                  relationship=rel["description"], 
                  weight=rel["weight"])
    
    # Create entities DataFrame
    entities_df = pd.DataFrame(entities)
    entities_df['degree'] = [G.degree[eid] for eid in entities_df['id']]
    entities_df['community'] = 0  # Simple community assignment
    
    # Assign communities based on type
    type_to_community = {
        "Person": 0,
        "Organization": 1, 
        "Project": 2,
        "Technology": 3,
        "Concept": 4
    }
    entities_df['community'] = entities_df['type'].map(type_to_community)
    
    # Create relationships DataFrame
    relationships_df = pd.DataFrame(relationships)
    relationships_df['rank'] = range(len(relationships_df))
    
    # Save as parquet files
    entities_df.to_parquet(output_path / "create_final_entities.parquet")
    relationships_df.to_parquet(output_path / "create_final_relationships.parquet")
    
    # Create communities DataFrame
    communities = []
    for community_id, community_name in enumerate(["People", "Organizations", "Projects", "Technologies", "Concepts"]):
        communities.append({
            "id": community_id,
            "title": community_name,
            "size": len(entities_df[entities_df['community'] == community_id]),
            "level": 0
        })
    
    communities_df = pd.DataFrame(communities)
    communities_df.to_parquet(output_path / "create_final_communities.parquet")
    
    # Save as GraphML
    nx.write_graphml(G, output_path / "graph.graphml")
    
    print(f"Sample GraphRAG data created:")
    print(f"  Entities: {len(entities)}")
    print(f"  Relationships: {len(relationships)}")
    print(f"  Communities: {len(communities)}")
    print(f"  Files saved to: {output_path}")
    
    return G

if __name__ == "__main__":
    create_sample_graphrag_data()
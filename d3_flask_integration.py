#!/usr/bin/env python3
"""
GraphRAG D3.js Integration with Flask
GraphRAG知識グラフのD3.js統合システム

Advanced web-based visualization with:
- Real-time graph interaction
- Dynamic filtering and search
- Multi-level zoom and pan
- Responsive design
"""

from flask import Flask, render_template, jsonify, request
import json
import networkx as nx
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any

app = Flask(__name__)

class D3GraphRAGServer:
    """
    Flask server for D3.js GraphRAG visualization
    D3.js GraphRAG視覚化用Flaskサーバー
    """
    
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.graph = nx.Graph()
        self.entities = {}
        self.relationships = {}
        self.communities = {}
        self.load_graph_data()
    
    def load_graph_data(self):
        """Load GraphRAG data from parquet files"""
        try:
            # Load entities
            entities_path = self.output_dir / "create_final_entities.parquet"
            if entities_path.exists():
                entities_df = pd.read_parquet(entities_path)
                for _, entity in entities_df.iterrows():
                    node_id = entity.get('id', entity.get('name', ''))
                    self.graph.add_node(
                        node_id,
                        name=entity.get('name', ''),
                        type=entity.get('type', ''),
                        description=entity.get('description', ''),
                        degree=entity.get('degree', 0)
                    )
            
            # Load relationships
            relationships_path = self.output_dir / "create_final_relationships.parquet"
            if relationships_path.exists():
                relationships_df = pd.read_parquet(relationships_path)
                for _, rel in relationships_df.iterrows():
                    source = rel.get('source', '')
                    target = rel.get('target', '')
                    if source and target:
                        self.graph.add_edge(
                            source,
                            target,
                            relationship=rel.get('description', ''),
                            weight=rel.get('weight', 1.0)
                        )
        
        except Exception as e:
            print(f"Error loading graph data: {e}")
    
    def get_graph_json(self) -> Dict[str, Any]:
        """Convert NetworkX graph to D3.js compatible JSON"""
        nodes = []
        links = []
        
        # Create node list
        for node_id, node_data in self.graph.nodes(data=True):
            nodes.append({
                'id': node_id,
                'name': node_data.get('name', str(node_id)),
                'type': node_data.get('type', 'Unknown'),
                'description': node_data.get('description', ''),
                'degree': self.graph.degree[node_id],
                'group': hash(node_data.get('type', 'Unknown')) % 10  # Color group
            })
        
        # Create edge list
        for source, target, edge_data in self.graph.edges(data=True):
            links.append({
                'source': source,
                'target': target,
                'relationship': edge_data.get('relationship', ''),
                'weight': edge_data.get('weight', 1.0)
            })
        
        return {
            'nodes': nodes,
            'links': links,
            'stats': {
                'node_count': len(nodes),
                'edge_count': len(links),
                'density': nx.density(self.graph) if len(nodes) > 1 else 0
            }
        }

# Global server instance
graph_server = D3GraphRAGServer()

@app.route('/')
def index():
    """Main visualization page"""
    return render_template('graph_visualization.html')

@app.route('/api/graph')
def get_graph():
    """API endpoint for graph data"""
    return jsonify(graph_server.get_graph_json())

@app.route('/api/search')
def search_entities():
    """Search entities by name or type"""
    query = request.args.get('q', '').lower()
    entity_type = request.args.get('type', '')
    
    results = []
    for node_id, node_data in graph_server.graph.nodes(data=True):
        name = node_data.get('name', str(node_id)).lower()
        node_type = node_data.get('type', '').lower()
        
        # Match query against name or description
        if query in name or query in node_data.get('description', '').lower():
            if not entity_type or entity_type.lower() in node_type:
                results.append({
                    'id': node_id,
                    'name': node_data.get('name', str(node_id)),
                    'type': node_data.get('type', 'Unknown'),
                    'description': node_data.get('description', ''),
                    'degree': graph_server.graph.degree[node_id]
                })
    
    return jsonify(results[:50])  # Limit to 50 results

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
#!/usr/bin/env python3
"""
Quick GraphRAG Visualization Guide
即座に使えるGraphRAG視覚化ガイド

Copy-paste ready code for immediate use
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pyvis.network import Network
import pandas as pd
from pathlib import Path

# =====================================
# 1. QUICK START: Static Visualization
# =====================================

def quick_static_visualization(graph_file, output="graph.png"):
    """
    Most basic GraphRAG visualization - copy and use immediately
    最も基本的なGraphRAG視覚化 - すぐにコピペで使用可能
    """
    # Load GraphML file
    G = nx.read_graphml(graph_file)
    
    # Simple force-directed layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Create figure
    plt.figure(figsize=(15, 12))
    
    # Draw nodes with size based on degree
    degrees = dict(G.degree())
    node_sizes = [degrees[node] * 100 for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, 
                          node_size=node_sizes,
                          node_color='lightblue',
                          alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, 
                          alpha=0.5,
                          edge_color='gray')
    
    # Draw labels for important nodes only
    important_nodes = [node for node, degree in degrees.items() if degree > 2]
    labels = {node: G.nodes[node].get('name', str(node)) for node in important_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=10)
    
    plt.title(f"GraphRAG Knowledge Graph ({len(G.nodes())} entities)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Static visualization saved: {output}")

# =====================================
# 2. INTERACTIVE WEB VISUALIZATION
# =====================================

def quick_interactive_visualization(graph_file, output="graph.html"):
    """
    Interactive web visualization - ready to deploy
    インタラクティブWeb視覚化 - デプロイ即可能
    """
    # Load graph
    G = nx.read_graphml(graph_file)
    
    # Create Pyvis network
    net = Network(height="600px", width="100%", bgcolor="#ffffff")
    
    # Configure physics for better layout
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100}
      }
    }
    """)
    
    # Add nodes
    for node in G.nodes():
        node_data = G.nodes[node]
        name = node_data.get('name', str(node))
        node_type = node_data.get('type', 'Unknown')
        
        net.add_node(node, 
                    label=name,
                    title=f"Type: {node_type}\\nConnections: {G.degree[node]}",
                    size=G.degree[node] * 5 + 10)
    
    # Add edges
    for source, target in G.edges():
        net.add_edge(source, target)
    
    # Save HTML file
    net.save_graph(output)
    print(f"Interactive visualization saved: {output}")

# =====================================
# 3. COMPREHENSIVE ANALYSIS
# =====================================

def comprehensive_graph_analysis(graph_file):
    """
    Complete analysis with multiple visualizations
    複数視覚化による包括的分析
    """
    G = nx.read_graphml(graph_file)
    
    print("=" * 50)
    print("GRAPHRAG KNOWLEDGE GRAPH ANALYSIS")
    print("=" * 50)
    
    # Basic statistics
    print(f"Nodes: {len(G.nodes())}")
    print(f"Edges: {len(G.edges())}")
    print(f"Density: {nx.density(G):.4f}")
    print(f"Connected Components: {nx.number_connected_components(G)}")
    
    # Centrality analysis
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    
    print("\nTOP 5 MOST CONNECTED ENTITIES:")
    top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    for node, score in top_degree:
        name = G.nodes[node].get('name', str(node))
        print(f"  {name}: {score:.3f}")
    
    print("\nTOP 5 MOST INFLUENTIAL ENTITIES:")
    top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    for node, score in top_betweenness:
        name = G.nodes[node].get('name', str(node))
        print(f"  {name}: {score:.3f}")
    
    # Entity type distribution
    types = {}
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'Unknown')
        types[node_type] = types.get(node_type, 0) + 1
    
    print("\nENTITY TYPE DISTRIBUTION:")
    for entity_type, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {entity_type}: {count}")

# =====================================
# 4. PRODUCTION-READY DASHBOARD
# =====================================

def create_dashboard_data(graph_file, output_dir="dashboard_data"):
    """
    Create JSON data for production dashboard
    本格ダッシュボード用JSONデータ作成
    """
    G = nx.read_graphml(graph_file)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Node data
    nodes = []
    for node in G.nodes():
        node_data = G.nodes[node]
        nodes.append({
            'id': node,
            'name': node_data.get('name', str(node)),
            'type': node_data.get('type', 'Unknown'),
            'description': node_data.get('description', ''),
            'degree': G.degree[node],
            'degree_centrality': nx.degree_centrality(G)[node],
            'betweenness_centrality': nx.betweenness_centrality(G).get(node, 0)
        })
    
    # Edge data
    edges = []
    for source, target in G.edges():
        edge_data = G.edges[source, target]
        edges.append({
            'source': source,
            'target': target,
            'relationship': edge_data.get('relationship', 'connected'),
            'weight': edge_data.get('weight', 1.0)
        })
    
    # Statistics
    stats = {
        'node_count': len(G.nodes()),
        'edge_count': len(G.edges()),
        'density': nx.density(G),
        'connected_components': nx.number_connected_components(G)
    }
    
    # Save JSON files
    import json
    
    with open(f"{output_dir}/nodes.json", 'w', encoding='utf-8') as f:
        json.dump(nodes, f, indent=2, ensure_ascii=False)
    
    with open(f"{output_dir}/edges.json", 'w', encoding='utf-8') as f:
        json.dump(edges, f, indent=2, ensure_ascii=False)
    
    with open(f"{output_dir}/stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"Dashboard data saved to: {output_dir}/")

# =====================================
# 5. USAGE EXAMPLES
# =====================================

if __name__ == "__main__":
    # Assume you have a GraphML file from GraphRAG
    graph_file = "./output/graph.graphml"
    
    if Path(graph_file).exists():
        print("Running GraphRAG visualization examples...")
        
        # 1. Quick static image
        quick_static_visualization(graph_file, "quick_graph.png")
        
        # 2. Interactive HTML
        quick_interactive_visualization(graph_file, "quick_interactive.html")
        
        # 3. Complete analysis
        comprehensive_graph_analysis(graph_file)
        
        # 4. Dashboard data
        create_dashboard_data(graph_file, "dashboard_data")
        
        print("\n" + "="*50)
        print("ALL VISUALIZATIONS COMPLETED!")
        print("="*50)
        print("Generated files:")
        print("  * quick_graph.png - Static visualization")
        print("  * quick_interactive.html - Web visualization")
        print("  * dashboard_data/ - JSON data for custom apps")
        
    else:
        print(f"GraphML file not found: {graph_file}")
        print("Please ensure you have GraphRAG output data.")

# =====================================
# 6. QUICK REFERENCE COMMANDS
# =====================================

"""
COPY-PASTE READY COMMANDS:

# Basic static visualization:
python -c "
import networkx as nx
import matplotlib.pyplot as plt
G = nx.read_graphml('output/graph.graphml')
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=300, font_size=8)
plt.savefig('graph.png', dpi=300, bbox_inches='tight')
print('Graph saved as graph.png')
"

# Interactive visualization:
python -c "
import networkx as nx
from pyvis.network import Network
G = nx.read_graphml('output/graph.graphml')
net = Network()
net.from_nx(G)
net.show('graph.html')
print('Interactive graph saved as graph.html')
"

# Graph statistics:
python -c "
import networkx as nx
G = nx.read_graphml('output/graph.graphml')
print(f'Nodes: {len(G.nodes())}, Edges: {len(G.edges())}')
print(f'Density: {nx.density(G):.4f}')
"
"""
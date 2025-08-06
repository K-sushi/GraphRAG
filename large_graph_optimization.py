#!/usr/bin/env python3
"""
Large Graph Visualization Optimization
大規模グラフ視覚化最適化システム

Advanced techniques for handling large knowledge graphs:
- Level of Detail (LOD) rendering
- Graph clustering and hierarchical views
- Streaming and progressive loading
- Performance optimization strategies
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import json
from pathlib import Path
import logging

# Community detection for clustering
try:
    import community as community_louvain
    COMMUNITY_AVAILABLE = True
except ImportError:
    COMMUNITY_AVAILABLE = False

logger = logging.getLogger(__name__)

class LargeGraphOptimizer:
    """
    大規模グラフ視覚化最適化システム
    Optimization system for large graph visualization
    """
    
    def __init__(self, graph: nx.Graph, max_display_nodes: int = 500):
        self.graph = graph
        self.max_display_nodes = max_display_nodes
        self.hierarchical_levels = {}
        self.node_importance = {}
        self.clusters = {}
        
        logger.info(f"Large graph optimizer initialized: {len(graph.nodes())} nodes")
    
    def calculate_node_importance(self) -> Dict[str, float]:
        """
        Calculate node importance using multiple centrality measures
        複数の中心性指標を使用したノード重要度計算
        """
        try:
            # Calculate multiple centrality measures
            degree_centrality = nx.degree_centrality(self.graph)
            betweenness_centrality = nx.betweenness_centrality(
                self.graph, k=min(100, len(self.graph.nodes()))
            )
            closeness_centrality = nx.closeness_centrality(self.graph)
            
            # PageRank for directed importance
            pagerank = nx.pagerank(self.graph)
            
            # Combine centralities with weights
            importance_scores = {}
            for node in self.graph.nodes():
                score = (
                    0.3 * degree_centrality.get(node, 0) +
                    0.3 * betweenness_centrality.get(node, 0) +
                    0.2 * closeness_centrality.get(node, 0) +
                    0.2 * pagerank.get(node, 0)
                )
                importance_scores[node] = score
            
            self.node_importance = importance_scores
            logger.info(f"Node importance calculated for {len(importance_scores)} nodes")
            return importance_scores
            
        except Exception as e:
            logger.error(f"Node importance calculation failed: {e}")
            # Fallback to degree centrality only
            self.node_importance = nx.degree_centrality(self.graph)
            return self.node_importance
    
    def create_hierarchical_clustering(self, levels: int = 3) -> Dict[int, Dict]:
        """
        Create hierarchical clustering for multi-level visualization
        マルチレベル視覚化のための階層クラスタリング
        """
        if not COMMUNITY_AVAILABLE:
            logger.warning("Community detection not available")
            return {}
        
        try:
            current_graph = self.graph.copy()
            hierarchical_data = {}
            
            for level in range(levels):
                # Detect communities at current level
                partition = community_louvain.best_partition(current_graph)
                communities = defaultdict(list)
                
                for node, community_id in partition.items():
                    communities[community_id].append(node)
                
                # Store level data
                hierarchical_data[level] = {
                    'communities': dict(communities),
                    'partition': partition,
                    'modularity': community_louvain.modularity(partition, current_graph),
                    'node_count': len(current_graph.nodes()),
                    'community_count': len(communities)
                }
                
                # Create next level graph (community graph)
                if level < levels - 1:
                    next_graph = nx.Graph()
                    
                    # Add community nodes
                    for community_id, nodes in communities.items():
                        # Calculate community properties
                        community_size = len(nodes)
                        total_degree = sum(current_graph.degree[node] for node in nodes)
                        avg_importance = np.mean([
                            self.node_importance.get(node, 0) for node in nodes
                        ])
                        
                        next_graph.add_node(
                            f"cluster_{community_id}",
                            size=community_size,
                            total_degree=total_degree,
                            avg_importance=avg_importance,
                            member_nodes=nodes
                        )
                    
                    # Add edges between communities
                    community_edges = defaultdict(int)
                    for edge in current_graph.edges():
                        source_community = partition[edge[0]]
                        target_community = partition[edge[1]]
                        
                        if source_community != target_community:
                            key = tuple(sorted([source_community, target_community]))
                            community_edges[key] += 1
                    
                    for (comm1, comm2), weight in community_edges.items():
                        next_graph.add_edge(
                            f"cluster_{comm1}",
                            f"cluster_{comm2}",
                            weight=weight
                        )
                    
                    current_graph = next_graph
            
            self.hierarchical_levels = hierarchical_data
            logger.info(f"Hierarchical clustering created with {levels} levels")
            return hierarchical_data
            
        except Exception as e:
            logger.error(f"Hierarchical clustering failed: {e}")
            return {}
    
    def generate_lod_subgraph(self, 
                             zoom_level: float = 1.0, 
                             focus_node: Optional[str] = None,
                             max_nodes: Optional[int] = None) -> nx.Graph:
        """
        Generate Level of Detail subgraph based on zoom level and focus
        ズームレベルと焦点に基づくLevel of Detailサブグラフ生成
        """
        if not max_nodes:
            max_nodes = self.max_display_nodes
        
        if not self.node_importance:
            self.calculate_node_importance()
        
        # Determine node selection strategy based on zoom level
        if zoom_level >= 2.0:
            # High zoom: show more nodes around focus
            selected_nodes = self._select_focused_nodes(focus_node, max_nodes)
        elif zoom_level >= 1.0:
            # Medium zoom: show important nodes
            selected_nodes = self._select_important_nodes(max_nodes)
        else:
            # Low zoom: show only most important nodes
            selected_nodes = self._select_top_nodes(max_nodes // 2)
        
        # Create subgraph
        subgraph = self.graph.subgraph(selected_nodes).copy()
        
        # Add importance scores to nodes
        for node in subgraph.nodes():
            subgraph.nodes[node]['importance'] = self.node_importance.get(node, 0)
        
        logger.info(f"LOD subgraph generated: {len(subgraph.nodes())} nodes at zoom {zoom_level}")
        return subgraph
    
    def _select_focused_nodes(self, focus_node: Optional[str], max_nodes: int) -> List[str]:
        """Select nodes around a focus node"""
        if not focus_node or focus_node not in self.graph:
            return self._select_important_nodes(max_nodes)
        
        # Use BFS to find nodes around focus
        distances = nx.single_source_shortest_path_length(
            self.graph, focus_node, cutoff=3
        )
        
        # Sort by distance and importance
        candidates = [
            (node, dist, self.node_importance.get(node, 0))
            for node, dist in distances.items()
        ]
        
        # Prioritize by distance (closer is better) and importance
        candidates.sort(key=lambda x: (x[1], -x[2]))
        
        return [node for node, _, _ in candidates[:max_nodes]]
    
    def _select_important_nodes(self, max_nodes: int) -> List[str]:
        """Select most important nodes"""
        sorted_nodes = sorted(
            self.node_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [node for node, _ in sorted_nodes[:max_nodes]]
    
    def _select_top_nodes(self, max_nodes: int) -> List[str]:
        """Select top nodes by combined metrics"""
        # Combine importance and degree for top-level view
        node_scores = {}
        degrees = dict(self.graph.degree())
        
        for node in self.graph.nodes():
            importance = self.node_importance.get(node, 0)
            degree = degrees.get(node, 0)
            # Normalize degree by maximum degree
            max_degree = max(degrees.values()) if degrees else 1
            normalized_degree = degree / max_degree
            
            node_scores[node] = 0.7 * importance + 0.3 * normalized_degree
        
        sorted_nodes = sorted(
            node_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [node for node, _ in sorted_nodes[:max_nodes]]
    
    def create_cluster_overview(self) -> Dict[str, Any]:
        """
        Create cluster-based overview for very large graphs
        非常に大きなグラフのためのクラスター概要作成
        """
        if not self.hierarchical_levels:
            self.create_hierarchical_clustering()
        
        if not self.hierarchical_levels:
            return {}
        
        # Use the top level (most aggregated)
        top_level = max(self.hierarchical_levels.keys())
        level_data = self.hierarchical_levels[top_level]
        
        cluster_graph = nx.Graph()
        cluster_info = {}
        
        for community_id, member_nodes in level_data['communities'].items():
            # Calculate cluster properties
            cluster_size = len(member_nodes)
            
            # Get internal edges (within cluster)
            subgraph = self.graph.subgraph(member_nodes)
            internal_edges = len(subgraph.edges())
            
            # Calculate average importance
            avg_importance = np.mean([
                self.node_importance.get(node, 0) for node in member_nodes
            ])
            
            # Find most important node in cluster
            cluster_nodes_importance = [
                (node, self.node_importance.get(node, 0))
                for node in member_nodes
            ]
            representative_node, max_importance = max(
                cluster_nodes_importance, key=lambda x: x[1]
            )
            
            # Add cluster node
            cluster_id = f"cluster_{community_id}"
            cluster_graph.add_node(
                cluster_id,
                size=cluster_size,
                internal_edges=internal_edges,
                avg_importance=avg_importance,
                representative=representative_node,
                member_count=cluster_size
            )
            
            cluster_info[cluster_id] = {
                'id': cluster_id,
                'size': cluster_size,
                'internal_edges': internal_edges,
                'avg_importance': avg_importance,
                'representative': representative_node,
                'members': member_nodes[:10]  # Show top 10 members
            }
        
        # Add edges between clusters
        partition = level_data['partition']
        cluster_edges = defaultdict(int)
        
        for edge in self.graph.edges():
            source_community = partition[edge[0]]
            target_community = partition[edge[1]]
            
            if source_community != target_community:
                source_cluster = f"cluster_{source_community}"
                target_cluster = f"cluster_{target_community}"
                key = tuple(sorted([source_cluster, target_cluster]))
                cluster_edges[key] += 1
        
        for (cluster1, cluster2), weight in cluster_edges.items():
            cluster_graph.add_edge(cluster1, cluster2, weight=weight)
        
        return {
            'graph': cluster_graph,
            'cluster_info': cluster_info,
            'stats': {
                'total_clusters': len(cluster_info),
                'total_nodes': len(self.graph.nodes()),
                'total_edges': len(self.graph.edges()),
                'modularity': level_data['modularity']
            }
        }
    
    def export_optimized_format(self, output_path: str) -> bool:
        """
        Export optimized format for web visualization
        Web視覚化用の最適化形式でエクスポート
        """
        try:
            # Calculate importance scores
            if not self.node_importance:
                self.calculate_node_importance()
            
            # Create hierarchical structure
            if not self.hierarchical_levels:
                self.create_hierarchical_clustering()
            
            # Generate different LOD levels
            lod_levels = {
                'high': self.generate_lod_subgraph(2.0, max_nodes=1000),
                'medium': self.generate_lod_subgraph(1.0, max_nodes=500),
                'low': self.generate_lod_subgraph(0.5, max_nodes=100)
            }
            
            # Create cluster overview
            cluster_overview = self.create_cluster_overview()
            
            # Format for JSON export
            export_data = {
                'metadata': {
                    'total_nodes': len(self.graph.nodes()),
                    'total_edges': len(self.graph.edges()),
                    'optimization_levels': list(lod_levels.keys()),
                    'hierarchical_levels': len(self.hierarchical_levels)
                },
                'lod_levels': {},
                'hierarchical_data': self.hierarchical_levels,
                'cluster_overview': {
                    'stats': cluster_overview.get('stats', {}),
                    'cluster_info': cluster_overview.get('cluster_info', {})
                },
                'node_importance': self.node_importance
            }
            
            # Convert NetworkX graphs to JSON format
            for level_name, subgraph in lod_levels.items():
                nodes = []
                edges = []
                
                for node_id, node_data in subgraph.nodes(data=True):
                    nodes.append({
                        'id': node_id,
                        'name': node_data.get('name', str(node_id)),
                        'type': node_data.get('type', 'Unknown'),
                        'importance': node_data.get('importance', 0),
                        'degree': subgraph.degree[node_id]
                    })
                
                for source, target, edge_data in subgraph.edges(data=True):
                    edges.append({
                        'source': source,
                        'target': target,
                        'weight': edge_data.get('weight', 1.0)
                    })
                
                export_data['lod_levels'][level_name] = {
                    'nodes': nodes,
                    'edges': edges,
                    'stats': {
                        'node_count': len(nodes),
                        'edge_count': len(edges)
                    }
                }
            
            # Add cluster overview graph
            if cluster_overview.get('graph'):
                cluster_graph = cluster_overview['graph']
                cluster_nodes = []
                cluster_edges = []
                
                for node_id, node_data in cluster_graph.nodes(data=True):
                    cluster_nodes.append({
                        'id': node_id,
                        'size': node_data.get('size', 1),
                        'avg_importance': node_data.get('avg_importance', 0),
                        'representative': node_data.get('representative', ''),
                        'member_count': node_data.get('member_count', 0)
                    })
                
                for source, target, edge_data in cluster_graph.edges(data=True):
                    cluster_edges.append({
                        'source': source,
                        'target': target,
                        'weight': edge_data.get('weight', 1.0)
                    })
                
                export_data['cluster_overview']['graph'] = {
                    'nodes': cluster_nodes,
                    'edges': cluster_edges
                }
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Optimized format exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

def optimize_large_graph(graph: nx.Graph, 
                        output_dir: str = "./output",
                        max_display_nodes: int = 500) -> bool:
    """
    Optimize large graph for visualization
    大規模グラフの視覚化最適化
    """
    print("🔧 Large Graph Visualization Optimization")
    print("=" * 50)
    
    try:
        optimizer = LargeGraphOptimizer(graph, max_display_nodes)
        
        print(f"📊 Original graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
        
        # Step 1: Calculate importance
        print("🧠 [1/4] Calculating node importance...")
        optimizer.calculate_node_importance()
        
        # Step 2: Create hierarchical clustering
        print("🏗️ [2/4] Creating hierarchical clustering...")
        hierarchical_data = optimizer.create_hierarchical_clustering()
        
        if hierarchical_data:
            for level, data in hierarchical_data.items():
                print(f"   Level {level}: {data['community_count']} communities, modularity: {data['modularity']:.3f}")
        
        # Step 3: Generate LOD levels
        print("🎯 [3/4] Generating Level of Detail views...")
        lod_high = optimizer.generate_lod_subgraph(2.0, max_nodes=1000)
        lod_medium = optimizer.generate_lod_subgraph(1.0, max_nodes=500)
        lod_low = optimizer.generate_lod_subgraph(0.5, max_nodes=100)
        
        print(f"   High detail: {len(lod_high.nodes())} nodes")
        print(f"   Medium detail: {len(lod_medium.nodes())} nodes")
        print(f"   Low detail: {len(lod_low.nodes())} nodes")
        
        # Step 4: Export optimized format
        print("💾 [4/4] Exporting optimized format...")
        output_path = Path(output_dir) / "optimized_graph.json"
        if optimizer.export_optimized_format(str(output_path)):
            print(f"   ✅ Exported to: {output_path}")
        
        # Create cluster overview
        cluster_overview = optimizer.create_cluster_overview()
        if cluster_overview:
            stats = cluster_overview['stats']
            print(f"\n📋 Cluster Overview:")
            print(f"   Clusters: {stats['total_clusters']}")
            print(f"   Modularity: {stats['modularity']:.3f}")
        
        print("\n🎉 Optimization completed!")
        return True
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return False

if __name__ == "__main__":
    # Example usage with synthetic graph
    print("Creating sample large graph for optimization...")
    
    # Create a large sample graph
    G = nx.barabasi_albert_graph(2000, 5)  # 2000 nodes, preferential attachment
    
    # Add some attributes
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['name'] = f"Entity_{node}"
        G.nodes[node]['type'] = f"Type_{i % 10}"
        G.nodes[node]['description'] = f"Sample entity {node} description"
    
    # Optimize
    optimize_large_graph(G)
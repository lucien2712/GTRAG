"""
Graph persistence utilities for gtrag.

Handles saving and loading of NetworkX graphs with metadata preservation.
"""

import pickle
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import networkx as nx

logger = logging.getLogger(__name__)


class GraphPersistence:
    """Utilities for saving and loading gtrag knowledge graphs."""
    
    @staticmethod
    def save_graph(graph: nx.Graph, filepath: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Save NetworkX graph to disk with optional metadata.
        
        Args:
            graph: NetworkX graph to save
            filepath: Path to save the graph
            metadata: Optional metadata to save alongside graph
        """
        filepath = Path(filepath)
        
        try:
            # Save graph using pickle (preserves all NetworkX features)
            with open(filepath, 'wb') as f:
                pickle.dump(graph, f)
            
            # Save metadata separately if provided
            if metadata:
                metadata_path = filepath.with_suffix('.metadata.json')
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                    
            logger.info(f"Graph saved successfully to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save graph to {filepath}: {e}")
            raise
    
    @staticmethod
    def load_graph(filepath: str) -> tuple[nx.Graph, Optional[Dict[str, Any]]]:
        """
        Load NetworkX graph from disk with optional metadata.
        
        Args:
            filepath: Path to load the graph from
            
        Returns:
            Tuple of (graph, metadata)
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Graph file not found: {filepath}")
        
        try:
            # Load graph
            with open(filepath, 'rb') as f:
                graph = pickle.load(f)
            
            # Load metadata if exists
            metadata = None
            metadata_path = filepath.with_suffix('.metadata.json')
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            logger.info(f"Graph loaded successfully from {filepath}")
            return graph, metadata
            
        except Exception as e:
            logger.error(f"Failed to load graph from {filepath}: {e}")
            raise
    
    @staticmethod
    def export_graph_gexf(graph: nx.Graph, filepath: str):
        """
        Export graph to GEXF format for visualization tools like Gephi.
        
        Args:
            graph: NetworkX graph to export
            filepath: Output filepath (.gexf extension will be added if missing)
        """
        filepath = Path(filepath)
        if filepath.suffix != '.gexf':
            filepath = filepath.with_suffix('.gexf')
        
        try:
            nx.write_gexf(graph, filepath)
            logger.info(f"Graph exported to GEXF format: {filepath}")
        except Exception as e:
            logger.error(f"Failed to export graph to GEXF: {e}")
            raise
    
    @staticmethod
    def export_graph_graphml(graph: nx.Graph, filepath: str):
        """
        Export graph to GraphML format for broader tool compatibility.
        
        Args:
            graph: NetworkX graph to export
            filepath: Output filepath (.graphml extension will be added if missing)
        """
        filepath = Path(filepath)
        if filepath.suffix != '.graphml':
            filepath = filepath.with_suffix('.graphml')
        
        try:
            nx.write_graphml(graph, filepath)
            logger.info(f"Graph exported to GraphML format: {filepath}")
        except Exception as e:
            logger.error(f"Failed to export graph to GraphML: {e}")
            raise
    
    @staticmethod
    def get_graph_stats(graph: nx.Graph) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the graph.
        
        Args:
            graph: NetworkX graph to analyze
            
        Returns:
            Dictionary with graph statistics
        """
        if len(graph) == 0:
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "is_connected": False,
                "avg_degree": 0,
                "density": 0,
                "temporal_nodes": 0,
                "entity_nodes": 0,
                "temporal_edges": 0
            }
        
        # Basic stats
        num_nodes = len(graph.nodes())
        num_edges = len(graph.edges())
        is_connected = nx.is_connected(graph)
        avg_degree = sum(dict(graph.degree()).values()) / num_nodes if num_nodes > 0 else 0
        density = nx.density(graph)
        
        # gtrag-specific stats
        temporal_nodes = sum(1 for n, data in graph.nodes(data=True) 
                           if data.get('node_type') == 'time')
        entity_nodes = sum(1 for n, data in graph.nodes(data=True) 
                         if data.get('node_type') == 'entity')
        temporal_edges = sum(1 for u, v, data in graph.edges(data=True) 
                           if data.get('relation_type') == 'temporal_evolution')
        
        # Node type distribution
        node_types = {}
        for n, data in graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "is_connected": is_connected,
            "avg_degree": round(avg_degree, 2),
            "density": round(density, 4),
            "temporal_nodes": temporal_nodes,
            "entity_nodes": entity_nodes,
            "temporal_edges": temporal_edges,
            "node_type_distribution": node_types
        }
    
    @staticmethod
    def validate_graph(graph: nx.Graph) -> Dict[str, Any]:
        """
        Validate gtrag graph structure and return validation results.
        
        Args:
            graph: NetworkX graph to validate
            
        Returns:
            Dictionary with validation results
        """
        issues = []
        warnings = []
        
        # Check for required node attributes
        required_node_attrs = ['type', 'name']
        for node, data in graph.nodes(data=True):
            for attr in required_node_attrs:
                if attr not in data:
                    issues.append(f"Node {node} missing required attribute: {attr}")
        
        # Check for temporal consistency
        temporal_nodes = [n for n, data in graph.nodes(data=True) 
                         if data.get('node_type') == 'time']
        if len(temporal_nodes) == 0:
            warnings.append("No temporal nodes found - temporal functionality may be limited")
        
        # Check for isolated nodes
        isolated_nodes = list(nx.isolates(graph))
        if isolated_nodes:
            warnings.append(f"Found {len(isolated_nodes)} isolated nodes")
        
        # Check for self-loops
        self_loops = list(nx.selfloop_edges(graph))
        if self_loops:
            warnings.append(f"Found {len(self_loops)} self-loops")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "validation_summary": f"Graph validation: {len(issues)} issues, {len(warnings)} warnings"
        }
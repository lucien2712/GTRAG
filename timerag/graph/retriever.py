"""
TimeRAG graph retrieval module.

This module contains the core `GraphRetriever` class, which is responsible for
retrieving the most relevant information from a constructed knowledge graph
based on user queries.

The main retrieval process includes:
1. **Query Understanding**: Use LLM to convert natural language questions into structured retrieval keywords
2. **Layered Retrieval**: Use different levels of keywords to search from both node and edge dimensions
3. **Multi-hop Expansion**: Starting from initially retrieved nodes, explore the graph outward to discover more implicit relevant information
"""

import logging
import json
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# Try to import required libraries
try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    cosine_similarity = None
    SKLEARN_AVAILABLE = False

from .builder import GraphBuilder
from ..extractors.llm_extractor import LLMExtractor
from ..config.settings import ModelConfig
from ..config.prompts import PromptConfig

logger = logging.getLogger(__name__)


class GraphRetriever:
    """Graph retriever implementing various strategies for searching information from the graph."""
    
    def __init__(self, graph_builder: GraphBuilder, extractor: LLMExtractor):
        """
        Initialize GraphRetriever.

        Args:
            graph_builder: GraphBuilder instance containing the knowledge graph
            extractor: LLMExtractor instance for calling LLM
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. Using fallback similarity calculation.")
        
        self.graph_builder = graph_builder
        self.graph = graph_builder.graph
        self.extractor = extractor
    
    def understand_query(self, query: str) -> Dict[str, Any]:
        """
        Use LLM to analyze user query and generate structured intent and keywords.
        
        Args:
            query: User's natural language question
            
        Returns:
            Dictionary containing query understanding results
        """
        return self.extractor.understand_query(query)

    def search(self, intent: Dict[str, Any], top_k: int = 10, similarity_threshold: float = 0.3) -> Tuple[List[Dict], List[Dict]]:
        """
        Retrieve entities and relationships from the graph based on query intent.

        Args:
            intent: Structured query intent from understand_query
            top_k: Maximum number of results to return
            similarity_threshold: Semantic similarity threshold

        Returns:
            Tuple of (entities list, relations list)
        """
        high_level_keys = intent.get("high_level_keywords", [])
        low_level_keys = intent.get("low_level_keywords", [])
        
        # Use vector store if available for more efficient search
        if self.graph_builder.vector_store:
            entities = self._search_with_vector_store(high_level_keys + low_level_keys, top_k, similarity_threshold)
        else:
            # Fallback to traditional graph search
            entities = self._search_entities_traditional(low_level_keys, similarity_threshold, top_k)
        
        # Search relationships
        relations = self._search_relations(high_level_keys, similarity_threshold, top_k)
        
        # Multi-hop expansion to find connected information
        expanded_entities, expanded_relations = self._multi_hop_expansion(
            entities, relations, max_hops=2, max_neighbors_per_hop=3
        )
        
        # Combine and rank results
        all_entities = self._merge_and_rank_entities(entities + expanded_entities, top_k)
        all_relations = self._merge_and_rank_relations(relations + expanded_relations, top_k)

        return all_entities, all_relations
    
    def _search_with_vector_store(self, keywords: List[str], top_k: int, threshold: float) -> List[Dict[str, Any]]:
        """Search entities using vector store for efficiency."""
        if not keywords:
            return []
        
        query_text = " ".join(keywords)
        similar_entities = self.graph_builder.search_similar_entities(
            query_text, top_k=top_k * 2  # Get more candidates for filtering
        )
        
        # Convert to standard format and filter by threshold
        entities = []
        for entity in similar_entities:
            if entity['similarity_score'] <= threshold:  # Lower score = higher similarity in some metrics
                entities.append({
                    'name': entity['name'],
                    'type': entity['type'],
                    'description': entity['description'],
                    'score': 1 - entity['similarity_score'],  # Convert to similarity score
                    'metadata': {'quarter': entity.get('quarter'), 'node_id': entity['node_id']}
                })
        
        return entities[:top_k]
    
    def _search_entities_traditional(self, keywords: List[str], threshold: float, top_k: int) -> List[Dict[str, Any]]:
        """Search entities using traditional graph traversal and similarity calculation."""
        if not keywords:
            return []
        
        query_text = " ".join(keywords)
        query_embedding = self.graph_builder.encode(query_text)
        
        if len(query_embedding) == 0:
            # Fallback to text matching
            return self._search_entities_text_match(keywords, top_k)
        
        candidates = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get('node_type') != 'entity':
                continue
                
            node_embedding = data.get('embedding')
            if node_embedding is not None and len(node_embedding) > 0:
                similarity = self._calculate_similarity(query_embedding, node_embedding)
                if similarity >= threshold:
                    candidates.append({
                        'name': data.get('name', node_id),
                        'type': data.get('type', 'unknown'),
                        'description': data.get('description', ''),
                        'score': similarity,
                        'metadata': {
                            'quarter': data.get('quarter'),
                            'node_id': node_id,
                            'chunk_id': data.get('chunk_id')
                        }
                    })
        
        # Sort by score and return top-k
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:top_k]
    
    def _search_entities_text_match(self, keywords: List[str], top_k: int) -> List[Dict[str, Any]]:
        """Fallback text-based entity search when embeddings are not available."""
        candidates = []
        keywords_lower = [k.lower() for k in keywords]
        
        for node_id, data in self.graph.nodes(data=True):
            if data.get('node_type') != 'entity':
                continue
            
            # Calculate text match score
            text_content = f"{data.get('name', '')} {data.get('description', '')}".lower()
            match_score = sum(1 for keyword in keywords_lower if keyword in text_content)
            
            if match_score > 0:
                candidates.append({
                    'name': data.get('name', node_id),
                    'type': data.get('type', 'unknown'),
                    'description': data.get('description', ''),
                    'score': match_score / len(keywords_lower),  # Normalize score
                    'metadata': {
                        'quarter': data.get('quarter'),
                        'node_id': node_id,
                        'chunk_id': data.get('chunk_id')
                    }
                })
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:top_k]

    def _search_relations(self, keywords: List[str], threshold: float, top_k: int) -> List[Dict[str, Any]]:
        """Search relationships using semantic similarity or text matching."""
        if not keywords:
            return []

        query_text = " ".join(keywords)
        query_embedding = self.graph_builder.encode(query_text)
        
        candidates = []
        for u, v, data in self.graph.edges(data=True):
            if data.get('relation_type') == 'temporal_evolution':
                continue  # Skip temporal edges in content search
            
            if len(query_embedding) > 0:
                # Use embedding similarity
                relation_text = f"{data.get('relation_type', '')} {data.get('description', '')}"
                relation_embedding = self.graph_builder.encode(relation_text)
                
                if len(relation_embedding) > 0:
                    similarity = self._calculate_similarity(query_embedding, relation_embedding)
                    if similarity >= threshold:
                        candidates.append({
                            'source': self.graph.nodes[u].get('name', u),
                            'target': self.graph.nodes[v].get('name', v),
                            'type': data.get('relation_type', 'unknown'),
                            'description': data.get('description', ''),
                            'score': similarity,
                            'metadata': {
                                'quarter': data.get('quarter'),
                                'source_id': u,
                                'target_id': v,
                                'chunk_id': data.get('chunk_id')
                            }
                        })
            else:
                # Fallback to text matching
                relation_text = f"{data.get('relation_type', '')} {data.get('description', '')}".lower()
                match_score = sum(1 for keyword in keywords if keyword.lower() in relation_text)
                
                if match_score > 0:
                    candidates.append({
                        'source': self.graph.nodes[u].get('name', u),
                        'target': self.graph.nodes[v].get('name', v),
                        'type': data.get('relation_type', 'unknown'),
                        'description': data.get('description', ''),
                        'score': match_score / len(keywords),
                        'metadata': {
                            'quarter': data.get('quarter'),
                            'source_id': u,
                            'target_id': v,
                            'chunk_id': data.get('chunk_id')
                        }
                    })

        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:top_k]

    def _multi_hop_expansion(self, entities: List[Dict], relations: List[Dict], 
                           max_hops: int = 2, max_neighbors_per_hop: int = 3) -> Tuple[List[Dict], List[Dict]]:
        """Perform multi-hop expansion to discover connected information."""
        if not entities:
            return [], []
        
        visited_nodes = set()
        expanded_entities = []
        expanded_relations = []
        
        # Start from high-scoring entities
        seed_nodes = [entity['metadata']['node_id'] for entity in entities[:3] 
                     if 'node_id' in entity.get('metadata', {})]
        
        current_nodes = seed_nodes
        visited_nodes.update(current_nodes)
        
        for hop in range(max_hops):
            next_nodes = []
            
            for node_id in current_nodes:
                if not self.graph.has_node(node_id):
                    continue
                
                # Get neighbors with importance scoring
                neighbors = list(self.graph.neighbors(node_id))[:max_neighbors_per_hop]
                
                for neighbor_id in neighbors:
                    if neighbor_id in visited_nodes:
                        continue
                    
                    neighbor_data = self.graph.nodes[neighbor_id]
                    if neighbor_data.get('node_type') == 'entity':
                        # Add expanded entity
                        expanded_entities.append({
                            'name': neighbor_data.get('name', neighbor_id),
                            'type': neighbor_data.get('type', 'unknown'),
                            'description': neighbor_data.get('description', ''),
                            'score': 0.5 - (hop * 0.1),  # Decay score by hop distance
                            'metadata': {
                                'quarter': neighbor_data.get('quarter'),
                                'node_id': neighbor_id,
                                'chunk_id': neighbor_data.get('chunk_id'),
                                'hop_distance': hop + 1
                            }
                        })
                    
                    # Add connecting relation
                    if self.graph.has_edge(node_id, neighbor_id):
                        edge_data = self.graph.get_edge_data(node_id, neighbor_id)
                        for key, relation_data in edge_data.items():
                            expanded_relations.append({
                                'source': self.graph.nodes[node_id].get('name', node_id),
                                'target': neighbor_data.get('name', neighbor_id),
                                'type': relation_data.get('relation_type', 'unknown'),
                                'description': relation_data.get('description', ''),
                                'score': 0.4 - (hop * 0.1),
                                'metadata': {
                                    'quarter': relation_data.get('quarter'),
                                    'source_id': node_id,
                                    'target_id': neighbor_id,
                                    'chunk_id': relation_data.get('chunk_id'),
                                    'hop_distance': hop + 1
                                }
                            })
                    
                    next_nodes.append(neighbor_id)
                    visited_nodes.add(neighbor_id)
            
            current_nodes = next_nodes
            if not current_nodes:
                break
        
        return expanded_entities, expanded_relations

    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        if SKLEARN_AVAILABLE and cosine_similarity:
            return cosine_similarity([embedding1], [embedding2])[0][0]
        else:
            # Fallback implementation
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

    def _merge_and_rank_entities(self, entities: List[Dict], top_k: int) -> List[Dict]:
        """Merge duplicate entities and rank by score."""
        # Remove duplicates based on node_id
        seen = set()
        unique_entities = []
        
        for entity in entities:
            node_id = entity.get('metadata', {}).get('node_id')
            if node_id and node_id not in seen:
                seen.add(node_id)
                unique_entities.append(entity)
            elif not node_id:  # Keep entities without node_id
                unique_entities.append(entity)
        
        # Sort by score and return top-k
        unique_entities.sort(key=lambda x: x['score'], reverse=True)
        return unique_entities[:top_k]

    def _merge_and_rank_relations(self, relations: List[Dict], top_k: int) -> List[Dict]:
        """Merge duplicate relations and rank by score."""
        # Remove duplicates based on source-target-type combination
        seen = set()
        unique_relations = []
        
        for relation in relations:
            key = (
                relation.get('metadata', {}).get('source_id'),
                relation.get('metadata', {}).get('target_id'),
                relation.get('type')
            )
            if key not in seen:
                seen.add(key)
                unique_relations.append(relation)
        
        # Sort by score and return top-k
        unique_relations.sort(key=lambda x: x['score'], reverse=True)
        return unique_relations[:top_k]

    def get_temporal_evolution(self, entity_name: str) -> List[Dict[str, Any]]:
        """Get temporal evolution path for a specific entity."""
        temporal_nodes = []
        
        # Find all nodes for this entity across time
        for node_id, data in self.graph.nodes(data=True):
            if (data.get('node_type') == 'entity' and 
                data.get('name', '').lower() == entity_name.lower()):
                temporal_nodes.append({
                    'node_id': node_id,
                    'quarter': data.get('quarter', ''),
                    'description': data.get('description', ''),
                    'data': data
                })
        
        # Sort by quarter
        temporal_nodes.sort(key=lambda x: x['quarter'])
        
        return temporal_nodes
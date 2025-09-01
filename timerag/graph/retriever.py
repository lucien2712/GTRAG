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
            relation_keywords = data.get('relation_keywords', [])
            if 'temporal_evolution' in relation_keywords:
                continue  # Skip temporal edges in content search
            
            if len(query_embedding) > 0:
                # Use embedding similarity - description already contains relation types
                relation_text = data.get('description', '')
                relation_embedding = self.graph_builder.encode(relation_text)
                
                if len(relation_embedding) > 0:
                    similarity = self._calculate_similarity(query_embedding, relation_embedding)
                    if similarity >= threshold:
                        candidates.append({
                            'source': self.graph.nodes[u].get('name', u),
                            'target': self.graph.nodes[v].get('name', v),
                            'type': ', '.join(relation_keywords) if relation_keywords else 'unknown',
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
                # Fallback to text matching - description already contains relation types
                relation_text = data.get('description', '').lower()
                match_score = sum(1 for keyword in keywords if keyword.lower() in relation_text)
                
                if match_score > 0:
                    relation_keywords = data.get('relation_keywords', [])
                    candidates.append({
                        'source': self.graph.nodes[u].get('name', u),
                        'target': self.graph.nodes[v].get('name', v),
                        'type': ', '.join(relation_keywords) if relation_keywords else 'unknown',
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
        """
        Perform multi-hop expansion with comprehensive scoring considering:
        1. Node similarity (semantic similarity to query)
        2. Edge similarity (edge relevance to query)
        3. Node centrality (degree, betweenness, pagerank)
        4. Edge centrality (edge betweenness)
        """
        if not entities:
            return [], []
        
        # Pre-calculate centrality measures for efficient lookup
        centrality_cache = self._calculate_centrality_measures()
        
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
            hop_candidates = []  # Store candidates for this hop to rank them
            
            for node_id in current_nodes:
                if not self.graph.has_node(node_id):
                    continue
                
                # Get all neighbors and calculate comprehensive scores
                neighbors = list(self.graph.neighbors(node_id))
                
                for neighbor_id in neighbors:
                    if neighbor_id in visited_nodes:
                        continue
                    
                    neighbor_data = self.graph.nodes[neighbor_id]
                    if neighbor_data.get('node_type') != 'entity':
                        continue
                    
                    # Calculate comprehensive neighbor score
                    neighbor_score = self._calculate_neighbor_importance(
                        node_id, neighbor_id, neighbor_data, centrality_cache, hop
                    )
                    
                    hop_candidates.append({
                        'neighbor_id': neighbor_id,
                        'neighbor_data': neighbor_data,
                        'source_id': node_id,
                        'score': neighbor_score,
                        'hop_distance': hop + 1
                    })
            
            # Sort candidates by comprehensive score and select top ones
            hop_candidates.sort(key=lambda x: x['score'], reverse=True)
            selected_candidates = hop_candidates[:max_neighbors_per_hop * len(current_nodes)]
            
            for candidate in selected_candidates:
                neighbor_id = candidate['neighbor_id']
                neighbor_data = candidate['neighbor_data']
                source_id = candidate['source_id']
                
                # Add expanded entity
                expanded_entities.append({
                    'name': neighbor_data.get('name', neighbor_id),
                    'type': neighbor_data.get('type', 'unknown'),
                    'description': neighbor_data.get('description', ''),
                    'score': candidate['score'],
                    'metadata': {
                        'quarter': neighbor_data.get('quarter'),
                        'node_id': neighbor_id,
                        'chunk_id': neighbor_data.get('chunk_id'),
                        'hop_distance': candidate['hop_distance'],
                        'centrality_score': centrality_cache['nodes'].get(neighbor_id, 0.0)
                    }
                })
                
                # Add connecting relation with edge scoring
                if self.graph.has_edge(source_id, neighbor_id):
                    edge_data = self.graph.get_edge_data(source_id, neighbor_id)
                    for key, relation_data in edge_data.items():
                        edge_score = self._calculate_edge_importance(
                            source_id, neighbor_id, relation_data, centrality_cache, hop
                        )
                        
                        relation_keywords = relation_data.get('relation_keywords', [])
                        expanded_relations.append({
                            'source': self.graph.nodes[source_id].get('name', source_id),
                            'target': neighbor_data.get('name', neighbor_id),
                            'type': ', '.join(relation_keywords) if relation_keywords else 'unknown',
                            'description': relation_data.get('description', ''),
                            'score': edge_score,
                            'metadata': {
                                'quarter': relation_data.get('quarter'),
                                'source_id': source_id,
                                'target_id': neighbor_id,
                                'chunk_id': relation_data.get('chunk_id'),
                                'hop_distance': candidate['hop_distance'],
                                'edge_centrality': centrality_cache['edges'].get((source_id, neighbor_id), 0.0)
                            }
                        })
                
                next_nodes.append(neighbor_id)
                visited_nodes.add(neighbor_id)
            
            current_nodes = next_nodes
            if not current_nodes:
                break
        
        return expanded_entities, expanded_relations

    def _calculate_centrality_measures(self) -> Dict[str, Dict]:
        """
        Calculate centrality measures for nodes and edges.
        Returns cached centrality scores for efficient lookup.
        """
        try:
            import networkx as nx
        except ImportError:
            logger.warning("NetworkX not available for centrality calculations. Using fallback scoring.")
            return {'nodes': {}, 'edges': {}}
        
        centrality_cache = {'nodes': {}, 'edges': {}}
        
        # Node centrality measures
        try:
            degree_centrality = nx.degree_centrality(self.graph)
            betweenness_centrality = nx.betweenness_centrality(self.graph, k=min(100, self.graph.number_of_nodes()))
            pagerank = nx.pagerank(self.graph, max_iter=50)
            
            # Combine centrality measures with weights from config
            from ..config.settings import ScoreWeights
            weights = ScoreWeights()
            
            for node_id in self.graph.nodes():
                combined_centrality = (
                    degree_centrality.get(node_id, 0.0) * weights.DEGREE_WEIGHT +
                    betweenness_centrality.get(node_id, 0.0) * weights.EDGE_WEIGHT +
                    pagerank.get(node_id, 0.0) * weights.SEMANTIC_WEIGHT
                )
                centrality_cache['nodes'][node_id] = combined_centrality
            
        except Exception as e:
            logger.warning(f"Node centrality calculation failed: {e}")
            # Fallback: use simple degree centrality
            for node_id in self.graph.nodes():
                degree = self.graph.degree(node_id)
                centrality_cache['nodes'][node_id] = degree / max(self.graph.number_of_nodes(), 1)
        
        # Edge centrality measures (edge betweenness)
        try:
            edge_betweenness = nx.edge_betweenness_centrality(self.graph, k=min(50, self.graph.number_of_nodes()))
            for edge, centrality in edge_betweenness.items():
                centrality_cache['edges'][edge] = centrality
                # Also add reverse edge for undirected behavior
                centrality_cache['edges'][(edge[1], edge[0])] = centrality
                
        except Exception as e:
            logger.warning(f"Edge centrality calculation failed: {e}")
            # Fallback: use constant edge weight
            for edge in self.graph.edges():
                centrality_cache['edges'][edge] = 0.1
        
        return centrality_cache

    def _calculate_neighbor_importance(self, source_id: str, neighbor_id: str, 
                                     neighbor_data: Dict, centrality_cache: Dict, hop: int) -> float:
        """
        Calculate comprehensive neighbor importance score considering:
        1. Node similarity (semantic similarity)
        2. Node centrality (combined centrality measures)
        3. Hop distance decay
        4. Temporal relevance
        """
        from ..config.settings import ScoreWeights
        weights = ScoreWeights()
        
        # 1. Node similarity score (semantic)
        neighbor_embedding = neighbor_data.get('embedding')
        if neighbor_embedding is not None and len(neighbor_embedding) > 0:
            # Calculate similarity to source node
            source_embedding = self.graph.nodes[source_id].get('embedding')
            if source_embedding is not None and len(source_embedding) > 0:
                semantic_score = self._calculate_similarity(
                    np.array(neighbor_embedding), np.array(source_embedding)
                )
            else:
                semantic_score = 0.3  # Default semantic score
        else:
            semantic_score = 0.3
        
        # 2. Node centrality score
        centrality_score = centrality_cache['nodes'].get(neighbor_id, 0.0)
        
        # 3. Hop distance decay
        hop_penalty = 0.8 ** hop  # Exponential decay
        
        # 4. Temporal relevance (bonus for temporal evolution edges)
        temporal_bonus = 1.0
        if self.graph.has_edge(source_id, neighbor_id):
            edge_data = self.graph.get_edge_data(source_id, neighbor_id)
            for key, relation_data in edge_data.items():
                relation_keywords = relation_data.get('relation_keywords', [])
                if 'temporal_evolution' in relation_keywords:
                    temporal_bonus = 1.2  # 20% bonus for temporal connections
                    break
        
        # Combine all factors
        importance_score = (
            semantic_score * weights.SEMANTIC_WEIGHT +
            centrality_score * weights.DEGREE_WEIGHT +
            0.2 * weights.EDGE_WEIGHT  # Base connectivity score
        ) * hop_penalty * temporal_bonus
        
        return min(importance_score, 1.0)  # Cap at 1.0

    def _calculate_edge_importance(self, source_id: str, target_id: str, 
                                 edge_data: Dict, centrality_cache: Dict, hop: int) -> float:
        """
        Calculate comprehensive edge importance score considering:
        1. Edge similarity (relation semantic relevance)
        2. Edge centrality (edge betweenness centrality)
        3. Edge type importance
        4. Hop distance decay
        """
        from ..config.settings import ScoreWeights
        weights = ScoreWeights()
        
        # 1. Edge semantic relevance (based on concatenated relations embedding similarity)  
        relation_keywords = edge_data.get('relation_keywords', [])
        relation_desc = edge_data.get('description', '')
        
        # For concatenated relations, the description already includes relation keywords
        # So we can directly use the description which contains all relation information
        relation_text = relation_desc if relation_desc else ""
        
        if len(relation_text) > 0:
            # Calculate relation embedding and compare with query context
            relation_embedding = self.graph_builder.encode(relation_text)
            if len(relation_embedding) > 0:
                # Compare with source and target node embeddings to get contextual relevance
                source_embedding = self.graph.nodes[source_id].get('embedding')
                target_embedding = self.graph.nodes[target_id].get('embedding')
                
                semantic_scores = []
                if source_embedding is not None and len(source_embedding) > 0:
                    source_sim = self._calculate_similarity(
                        np.array(relation_embedding), np.array(source_embedding)
                    )
                    semantic_scores.append(source_sim)
                
                if target_embedding is not None and len(target_embedding) > 0:
                    target_sim = self._calculate_similarity(
                        np.array(relation_embedding), np.array(target_embedding)
                    )
                    semantic_scores.append(target_sim)
                
                # Use average similarity if we have embeddings, otherwise fallback
                semantic_score = np.mean(semantic_scores) if semantic_scores else 0.3
            else:
                semantic_score = 0.3  # Fallback when encoding fails
        else:
            semantic_score = 0.2  # Very low score for empty descriptions
        
        # 2. Edge centrality score
        edge_centrality = centrality_cache['edges'].get((source_id, target_id), 0.0)
        
        # 3. No relation type importance weighting - treat all relations equally
        type_importance = 0.5  # Neutral importance for all relation types
        
        # 4. Hop distance decay
        hop_penalty = 0.7 ** hop
        
        # Combine all factors
        edge_score = (
            semantic_score * weights.SEMANTIC_WEIGHT +
            edge_centrality * weights.EDGE_WEIGHT +
            type_importance * weights.DEGREE_WEIGHT
        ) * hop_penalty
        
        return min(edge_score, 1.0)  # Cap at 1.0

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
"""
TimeRAG knowledge graph construction module.

This module contains the core `GraphBuilder` class, which uses `networkx` to build,
manage and store knowledge graphs. The graph consists of nodes representing "entities"
and edges representing "relationships".

Main features include:
- Adding extracted entities and relationships to the graph
- Generating semantic embedding vectors for entity and relationship descriptions
- Building "temporal edges" across different time periods to capture temporal information
- Providing graph save and load functionality
"""

import json
import logging
from typing import List, Dict, Any, Optional, Callable
import networkx as nx
import numpy as np

# Try to import sentence_transformers, set to None if failed
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from ..extractors.llm_extractor import Entity, Relation
from ..config.settings import ModelConfig
from ..storage.vector_store import VectorStore, FAISSVectorStore, InMemoryVectorStore

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Responsible for building and managing the knowledge graph.
    
    This class encapsulates all graph operation logic, including adding nodes, adding edges,
    building temporal associations, and serialization (save/load).
    """
    
    def __init__(self, 
                 embedding_model_name: Optional[str] = None, 
                 embedding_func: Optional[Callable[[str], np.ndarray]] = None,
                 use_vector_store: bool = True,
                 vector_store: Optional[VectorStore] = None):
        """
        Initialize GraphBuilder.

        Supports two embedding vector generation modes:
        1. `embedding_model_name` mode: Use `sentence-transformers` library to load pre-trained models
        2. `embedding_func` mode: Use a user-defined function to generate embedding vectors

        Args:
            embedding_model_name: SentenceTransformer model name (optional)
            embedding_func: Custom embedding vector generation function (optional)
            use_vector_store: Whether to use vector database for embeddings
            vector_store: Custom vector store instance (optional)
        """
        self.graph = nx.MultiDiGraph()  # Use multi-directed graph, allowing multiple relationship types between nodes
        
        # Initialize embedding function
        if embedding_func:
            self.encode = embedding_func
        elif SENTENCE_TRANSFORMERS_AVAILABLE and (embedding_model_name or ModelConfig.EMBEDDING_MODEL):
            model = embedding_model_name or ModelConfig.EMBEDDING_MODEL
            self.encoder = SentenceTransformer(model)
            self.encode = self._default_encode
        else:
            logger.warning("No embedding function available. Embeddings will be disabled.")
            self.encode = lambda x: np.array([])  # Dummy function
        
        # Initialize vector store for efficient similarity search
        self.use_vector_store = use_vector_store
        if use_vector_store:
            if vector_store:
                self.vector_store = vector_store
            else:
                # Try to use FAISS, fallback to in-memory store
                try:
                    # Determine embedding dimension
                    test_embedding = self.encode("test")
                    dimension = len(test_embedding) if len(test_embedding) > 0 else 384
                    self.vector_store = FAISSVectorStore(dimension=dimension, metric="cosine")
                except Exception as e:
                    logger.warning(f"Failed to initialize FAISS vector store: {e}. Using in-memory store.")
                    dimension = 384  # Default dimension
                    self.vector_store = InMemoryVectorStore(dimension=dimension, metric="cosine")
        else:
            self.vector_store = None

    def _default_encode(self, text: str) -> np.ndarray:
        """Use default SentenceTransformer for encoding."""
        return self.encoder.encode(text)
        
    def add_entities(self, entities: List[Entity]):
        """Add a batch of entities as nodes to the graph. If node exists, merge its attributes."""
        vectors_to_add = []
        metadata_to_add = []
        ids_to_add = []
        
        for entity in entities:
            node_id = f"{entity.name}__{entity.metadata.get('quarter', 'Q_UNKNOWN')}"
            
            if self.graph.has_node(node_id):
                # Node exists, perform merge
                existing_data = self.graph.nodes[node_id]
                
                # 1. Merge descriptions
                new_description = (existing_data.get('description', '') + 
                                 "\n---\n" + 
                                 entity.description)
                
                # 2. Merge source documents (ensure it's a list)
                source_docs = existing_data.get('source_doc_id', [])
                if not isinstance(source_docs, list):
                    source_docs = [source_docs]
                if entity.source_doc_id not in source_docs:
                    source_docs.append(entity.source_doc_id)

                # 3. Recalculate embedding for merged description
                # Combine entity name and description for richer embedding
                entity_text = f"{entity.name}\n{new_description}"
                new_embedding = self.encode(entity_text)

                # 4. Update node attributes
                self.graph.nodes[node_id]['description'] = new_description
                if len(new_embedding) > 0:
                    self.graph.nodes[node_id]['embedding'] = new_embedding
                self.graph.nodes[node_id]['source_doc_id'] = source_docs
                
                # Update vector store
                if self.vector_store and len(new_embedding) > 0:
                    # Remove old embedding and add new one
                    self.vector_store.delete([node_id])
                    vectors_to_add.append(new_embedding.reshape(1, -1))
                    metadata_to_add.append({
                        'node_id': node_id,
                        'type': 'entity',
                        'entity_type': entity.type,
                        'quarter': entity.metadata.get('quarter', 'Q_UNKNOWN')
                    })
                    ids_to_add.append(node_id)
                
                logger.info(f"Merged entity node: {node_id}")

            else:
                # Node doesn't exist, add normally
                # Combine entity name and description for richer embedding
                entity_text = f"{entity.name}\n{entity.description}"
                embedding = self.encode(entity_text)
                
                self.graph.add_node(
                    node_id,
                    node_type="entity",
                    name=entity.name,
                    type=entity.type,
                    description=entity.description,
                    embedding=embedding if len(embedding) > 0 else None,
                    source_doc_id=[entity.source_doc_id],  # Initialize as list
                    **entity.metadata
                )
                
                # Add to vector store
                if self.vector_store and len(embedding) > 0:
                    vectors_to_add.append(embedding.reshape(1, -1))
                    metadata_to_add.append({
                        'node_id': node_id,
                        'type': 'entity',
                        'entity_type': entity.type,
                        'quarter': entity.metadata.get('quarter', 'Q_UNKNOWN')
                    })
                    ids_to_add.append(node_id)
        
        # Batch add to vector store
        if self.vector_store and vectors_to_add:
            combined_vectors = np.vstack(vectors_to_add)
            self.vector_store.add_vectors(combined_vectors, metadata_to_add, ids_to_add)
            logger.info(f"Added {len(vectors_to_add)} entity embeddings to vector store")
    
    def add_relations(self, relations: List[Relation]):
        """Add a batch of relationships as edges to the graph. If relationship exists, merge its attributes."""
        for relation in relations:
            source_id = f"{relation.source}__{relation.metadata.get('quarter', 'Q_UNKNOWN')}"
            target_id = f"{relation.target}__{relation.metadata.get('quarter', 'Q_UNKNOWN')}"
            
            # Use a single key for all relations between two nodes
            key = "relation"

            if self.graph.has_node(source_id) and self.graph.has_node(target_id):
                if self.graph.has_edge(source_id, target_id, key=key):
                    # Edge exists, concatenate multiple relations
                    existing_data = self.graph.get_edge_data(source_id, target_id, key=key)
                    
                    # 1. Concatenate relation keywords  
                    existing_keywords = existing_data.get('relation_keywords', [])
                    if relation.keywords not in existing_keywords:
                        existing_keywords.append(relation.keywords)
                    
                    # 2. Concatenate descriptions with relation keywords prefix
                    existing_desc = existing_data.get('description', '')
                    new_relation_desc = f"{relation.keywords}: {relation.description}"
                    new_description = f"{existing_desc}\n{new_relation_desc}" if existing_desc else new_relation_desc
                    
                    # 3. Concatenate evidence
                    existing_evidence = existing_data.get('evidence', '')
                    new_evidence = f"{existing_evidence}\n{relation.evidence}" if existing_evidence else relation.evidence

                    # 4. Update edge attributes
                    self.graph[source_id][target_id][key]['relation_keywords'] = existing_keywords
                    self.graph[source_id][target_id][key]['description'] = new_description
                    self.graph[source_id][target_id][key]['evidence'] = new_evidence
                    logger.info(f"Concatenated relation edge: {source_id} -> {target_id} (added {relation.keywords})")

                else:
                    # Edge doesn't exist, create new edge
                    self.graph.add_edge(
                        source_id,
                        target_id,
                        key=key,
                        relation_keywords=[relation.keywords],  # Store as list for future concatenation
                        description=f"{relation.keywords}: {relation.description}",
                        evidence=relation.evidence,
                        source_doc_id=relation.source_doc_id,
                        **relation.metadata
                    )
                    logger.debug(f"Added relation edge: {source_id} -> {target_id} ({relation.keywords})")
            else:
                missing_nodes = []
                if not self.graph.has_node(source_id):
                    missing_nodes.append(source_id)
                if not self.graph.has_node(target_id):
                    missing_nodes.append(target_id)
                logger.warning(f"Cannot add relation {relation.type}: missing nodes {missing_nodes}")
    
    def build_temporal_connections(self):
        """Build cross-temporal connection edges - this is one of TimeRAG's core features."""
        entity_nodes = {}
        
        # Group all nodes by entity name
        for node_id, data in self.graph.nodes(data=True):
            if data.get("node_type") == "entity":
                entity_name = data.get("name")
                if entity_name not in entity_nodes:
                    entity_nodes[entity_name] = []
                # Record node ID and its quarter information
                entity_nodes[entity_name].append((data.get("quarter", ""), node_id))
        
        # Build temporal evolution edges between different quarter nodes of the same entity
        temporal_edges_added = 0
        for entity_name, nodes in entity_nodes.items():
            # Sort by quarter to ensure correct temporal order
            sorted_nodes = sorted(nodes, key=lambda x: x[0])
            
            for i in range(len(sorted_nodes) - 1):
                source_q, source_id = sorted_nodes[i]
                target_q, target_id = sorted_nodes[i+1]
                
                self.graph.add_edge(
                    source_id,
                    target_id,
                    key="temporal_evolution",
                    relation_keywords=["temporal_evolution"],
                    description=f"temporal_evolution: {entity_name} evolution from {source_q} to {target_q}"
                )
                temporal_edges_added += 1
        
        logger.info(f"Built {temporal_edges_added} temporal connections")
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        node_types = {}
        relation_types = {}
        
        # Count node types
        for node_id, data in self.graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # Count relation keywords across all edges
        for u, v, data in self.graph.edges(data=True):
            edge_relation_keywords = data.get('relation_keywords', ['unknown'])
            for keywords in edge_relation_keywords:
                relation_types[keywords] = relation_types.get(keywords, 0) + 1
        
        stats = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "node_types": node_types,
            "relation_types": relation_types,
            "has_vector_store": self.vector_store is not None
        }
        
        if self.vector_store:
            stats.update(self.vector_store.get_stats())
        
        return stats
    
    def search_similar_entities(self, query_text: str, top_k: int = 10, 
                               filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar entities using vector similarity."""
        if not self.vector_store:
            logger.warning("Vector store not available for similarity search")
            return []
        
        query_embedding = self.encode(query_text)
        if len(query_embedding) == 0:
            return []
        
        results = self.vector_store.search(query_embedding, top_k, filter_metadata)
        
        # Enrich results with graph data
        enriched_results = []
        for result in results:
            node_id = result['id']
            if self.graph.has_node(node_id):
                node_data = self.graph.nodes[node_id]
                enriched_results.append({
                    'node_id': node_id,
                    'similarity_score': result['score'],
                    'name': node_data.get('name'),
                    'type': node_data.get('type'),
                    'description': node_data.get('description'),
                    'quarter': node_data.get('quarter')
                })
        
        return enriched_results
    
    def save(self, filepath: str):
        """Serialize graph data and save as JSON file."""
        # networkx node_link_data format is convenient for JSON serialization
        # But need to manually handle numpy arrays, convert them to lists
        graph_data = nx.node_link_data(self.graph)
        for node in graph_data["nodes"]:
            if 'embedding' in node and isinstance(node['embedding'], np.ndarray):
                node['embedding'] = node['embedding'].tolist()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Graph saved to {filepath}")
        # Note: Vector store is now saved separately by TimeRAGSystem
    
    def load(self, filepath: str):
        """Load graph data from JSON file and rebuild graph."""
        with open(filepath, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        # Convert embedding from list back to numpy array
        for node in graph_data["nodes"]:
            if 'embedding' in node and isinstance(node['embedding'], list):
                node['embedding'] = np.array(node['embedding'])
        
        self.graph = nx.node_link_graph(graph_data)
        
        # Initialize vector store if needed (but don't load - that's handled by TimeRAGSystem)
        if self.use_vector_store and not self.vector_store:
            try:
                test_embedding = self.encode("test")
                dimension = len(test_embedding) if len(test_embedding) > 0 else 384
                
                if FAISSVectorStore:
                    self.vector_store = FAISSVectorStore(dimension=dimension, metric="cosine")
                else:
                    self.vector_store = InMemoryVectorStore(dimension=dimension, metric="cosine")
                logger.info(f"Initialized empty vector store (dimension={dimension})")
            except Exception as e:
                logger.warning(f"Failed to initialize vector store: {e}")
                self.vector_store = None
        
        logger.info(f"Graph loaded from {filepath}")
        # Note: Vector store loading is now handled by TimeRAGSystem
    
    def get_neighbors(self, node_id: str, relation_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get neighbors of a node with optional relation type filtering."""
        if not self.graph.has_node(node_id):
            return []
        
        neighbors = []
        for neighbor_id in self.graph.neighbors(node_id):
            edge_data = self.graph.get_edge_data(node_id, neighbor_id)
            for key, data in edge_data.items():
                if relation_types is None or data.get('relation_type') in relation_types:
                    neighbor_data = self.graph.nodes[neighbor_id]
                    neighbors.append({
                        'node_id': neighbor_id,
                        'name': neighbor_data.get('name'),
                        'type': neighbor_data.get('type'),
                        'relation_type': data.get('relation_type'),
                        'relation_description': data.get('description')
                    })
        
        return neighbors
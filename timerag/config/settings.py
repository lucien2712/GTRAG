"""
TimeRAG system core configuration settings.

This file defines configuration dataclasses for all TimeRAG modules,
making it easy to manage and customize system behavior.
"""
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """
    Model and algorithm configuration settings.
    
    Defines core parameters for language models, embedding models,
    and retrieval algorithms.
    """
    # OpenAI language model settings
    DEFAULT_MODEL: str = "gpt-4"
    TEMPERATURE: float = 0.1

    # Semantic embedding model settings  
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Retrieval algorithm settings
    DEFAULT_TOP_K: int = 10  # Default number of retrieved nodes
    SIMILARITY_THRESHOLD: float = 0.3  # Semantic similarity threshold
    MAX_NEIGHBORS_PER_HOP: int = 5  # Max neighbors to explore per hop in graph
    MAX_HOPS: int = 3  # Maximum number of hops in graph exploration

    # Importance score thresholds
    HIGH_IMPORTANCE_THRESHOLD: float = 0.7
    MEDIUM_IMPORTANCE_THRESHOLD: float = 0.5

    # Graph settings
    MAX_CHAIN_LENGTH: int = 4  # Maximum reasoning chain length
    MAX_REASONING_CHAINS: int = 10  # Maximum number of reasoning chains


@dataclass
class IndexingParams:
    """
    Document indexing stage parameter configuration.
    
    These parameters affect how documents are processed and added to the knowledge graph.
    """
    enable_entity_linking: bool = True  # Enable entity linking (connecting same entities)
    enable_temporal_connections: bool = True  # Enable temporal relation analysis
    entity_similarity_threshold: float = 0.8  # Similarity threshold for entity linking

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary format."""
        return {
            "enable_entity_linking": self.enable_entity_linking,
            "enable_temporal_connections": self.enable_temporal_connections,
            "entity_similarity_threshold": self.entity_similarity_threshold
        }


@dataclass
class QueryParams:
    """
    Query stage parameter configuration.
    
    Users can customize query behavior through these parameters when asking questions.
    """
    # Retrieval parameters
    top_k: int = 10
    max_hops: int = 3
    max_neighbors_per_hop: int = 5
    similarity_threshold: float = 0.3
    enable_multi_hop: bool = True  # Enable multi-hop queries

    # Hybrid retrieval weights
    centrality_weight: float = 0.3  # Graph centrality weight
    similarity_weight: float = 0.7  # Semantic similarity weight

    # Token management parameters
    entity_max_tokens: int = 30000  # Max tokens for entity information
    relation_max_tokens: int = 30000  # Max tokens for relation information
    final_max_tokens: int = 120000  # Total max tokens sent to model

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary format."""
        return {
            "top_k": self.top_k,
            "max_hops": self.max_hops,
            "max_neighbors_per_hop": self.max_neighbors_per_hop,
            "similarity_threshold": self.similarity_threshold,
            "enable_multi_hop": self.enable_multi_hop,
            "centrality_weight": self.centrality_weight,
            "similarity_weight": self.similarity_weight,
            "entity_max_tokens": self.entity_max_tokens,
            "relation_max_tokens": self.relation_max_tokens,
            "final_max_tokens": self.final_max_tokens
        }


@dataclass
class ChunkingConfig:
    """
    Document chunking configuration.
    
    Defines how to split long documents into manageable pieces for model processing.
    """
    # Token limits
    MAX_TOKENS_PER_CHUNK: int = 3000  # Maximum tokens per chunk
    OVERLAP_TOKENS: int = 200  # Overlapping tokens between chunks
    MIN_CHUNK_TOKENS: int = 500  # Minimum tokens per chunk

    # Splitting preferences
    SENTENCE_BOUNDARY: bool = True  # Split on sentence boundaries
    PARAGRAPH_BOUNDARY: bool = True  # Split on paragraph boundaries
    PRESERVE_SECTIONS: bool = True  # Preserve section structure

    # Processing settings
    MERGE_DUPLICATE_ENTITIES: bool = True  # Merge duplicate entities
    MERGE_DUPLICATE_RELATIONS: bool = True  # Merge duplicate relations
    VALIDATE_CROSS_CHUNK_CONSISTENCY: bool = True  # Validate cross-chunk consistency

    # Encoder name (for tiktoken)
    ENCODING_NAME: str = "o200k_base"


@dataclass
class RetrievalWeights:
    """
    Hybrid retrieval weight configuration.
    
    Used to balance the importance of different retrieval strategies.
    """
    TEMPORAL: float = 0.6  # Temporal relevance weight
    SEMANTIC: float = 0.4  # Semantic similarity weight

    # Node importance score weights
    DEGREE_WEIGHT: float = 0.3  # Node degree (connection count) weight
    EDGE_WEIGHT: float = 0.4  # Edge weight
    SEMANTIC_WEIGHT: float = 0.3  # Semantic weight


@dataclass
class VectorDBConfig:
    """
    Vector database configuration for embedding storage.
    """
    provider: str = "faiss"  # Vector DB provider (faiss, chroma, etc.)
    dimension: int = 384  # Embedding dimension
    index_type: str = "flat"  # Index type for vector search
    metric: str = "cosine"  # Distance metric
    
    # Database connection settings
    host: Optional[str] = None
    port: Optional[int] = None
    collection_name: str = "timerag_embeddings"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "provider": self.provider,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "host": self.host,
            "port": self.port,
            "collection_name": self.collection_name
        }
"""
Vector database implementations for TimeRAG.

Provides abstract interface and concrete implementations for vector storage
and similarity search functionality.
"""

import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Install with: pip install faiss-cpu")


class VectorStore(ABC):
    """Abstract base class for vector storage implementations."""
    
    @abstractmethod
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]], ids: List[str]):
        """Add vectors with metadata to the store."""
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 10, 
               filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]):
        """Delete vectors by IDs."""
        pass
    
    @abstractmethod
    def save(self, filepath: str):
        """Save the vector store to disk.""" 
        pass
    
    @abstractmethod
    def load(self, filepath: str):
        """Load the vector store from disk."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        pass


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store implementation."""
    
    def __init__(self, dimension: int, metric: str = "cosine", index_type: str = "flat"):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Vector dimension
            metric: Distance metric (cosine, l2)
            index_type: FAISS index type (flat, ivf)
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required for FAISSVectorStore. Install with: pip install faiss-cpu")
            
        self.dimension = dimension
        self.metric = metric
        self.index_type = index_type
        
        # Initialize FAISS index
        if index_type == "flat":
            if metric == "cosine":
                self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine
            else:
                self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "ivf":
            # IVF index for faster search on large datasets
            nlist = 100  # Number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
            
        # Store metadata and IDs separately
        self.metadata: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self.id_to_idx: Dict[str, int] = {}
        
        logger.info(f"Initialized FAISS vector store: dimension={dimension}, metric={metric}, type={index_type}")
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]], ids: List[str]):
        """Add vectors with metadata to the FAISS index."""
        if vectors.shape[0] != len(metadata) or vectors.shape[0] != len(ids):
            raise ValueError("Number of vectors, metadata entries, and IDs must match")
            
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match store dimension {self.dimension}")
        
        # Normalize vectors for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(vectors)
        
        # Train index if necessary (for IVF)
        if self.index_type == "ivf" and not self.index.is_trained:
            self.index.train(vectors)
        
        # Add vectors to index
        start_idx = len(self.ids)
        self.index.add(vectors)
        
        # Store metadata and IDs
        self.metadata.extend(metadata)
        self.ids.extend(ids)
        
        # Update ID to index mapping
        for i, vec_id in enumerate(ids):
            self.id_to_idx[vec_id] = start_idx + i
            
        logger.info(f"Added {len(vectors)} vectors to FAISS store")
    
    def search(self, query_vector: np.ndarray, top_k: int = 10, 
               filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors in the FAISS index."""
        if query_vector.shape[0] != self.dimension:
            raise ValueError(f"Query vector dimension {query_vector.shape[0]} doesn't match store dimension {self.dimension}")
        
        # Reshape to 2D array and normalize for cosine similarity
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(query_vector)
        
        # Search in FAISS
        distances, indices = self.index.search(query_vector, min(top_k * 2, len(self.ids)))  # Get more results for filtering
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
                
            if idx >= len(self.metadata):
                continue
                
            result = {
                "id": self.ids[idx],
                "score": float(dist),
                "metadata": self.metadata[idx].copy()
            }
            
            # Apply metadata filtering if specified
            if filter_metadata:
                if not self._matches_filter(result["metadata"], filter_metadata):
                    continue
            
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        logger.debug(f"FAISS search returned {len(results)} results")
        return results
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_criteria: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter_criteria.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True
    
    def delete(self, ids: List[str]):
        """Delete vectors by IDs (FAISS doesn't support direct deletion, so we mark as deleted)."""
        # FAISS doesn't support efficient deletion, so we implement soft deletion
        for vec_id in ids:
            if vec_id in self.id_to_idx:
                idx = self.id_to_idx[vec_id]
                # Mark metadata as deleted
                if idx < len(self.metadata):
                    self.metadata[idx]["_deleted"] = True
                logger.debug(f"Marked vector {vec_id} as deleted")
    
    def save(self, filepath: str):
        """Save FAISS index and metadata to disk."""
        filepath = Path(filepath)
        
        # Save FAISS index
        index_path = filepath.with_suffix('.faiss')
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata and IDs
        metadata_path = filepath.with_suffix('.metadata.npy')
        np.save(metadata_path, {
            'metadata': self.metadata,
            'ids': self.ids,
            'id_to_idx': self.id_to_idx,
            'dimension': self.dimension,
            'metric': self.metric,
            'index_type': self.index_type
        })
        
        logger.info(f"Saved FAISS vector store to {filepath}")
    
    def load(self, filepath: str):
        """Load FAISS index and metadata from disk."""
        filepath = Path(filepath)
        
        # Load FAISS index
        index_path = filepath.with_suffix('.faiss')
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")
        
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata and IDs
        metadata_path = filepath.with_suffix('.metadata.npy')
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        data = np.load(metadata_path, allow_pickle=True).item()
        self.metadata = data['metadata']
        self.ids = data['ids']
        self.id_to_idx = data['id_to_idx']
        self.dimension = data['dimension']
        self.metric = data['metric']
        self.index_type = data['index_type']
        
        logger.info(f"Loaded FAISS vector store from {filepath}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the FAISS vector store."""
        return {
            "total_vectors": len(self.ids),
            "dimension": self.dimension,
            "metric": self.metric,
            "index_type": self.index_type,
            "is_trained": getattr(self.index, 'is_trained', True)
        }


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store implementation for testing."""
    
    def __init__(self, dimension: int, metric: str = "cosine"):
        """Initialize in-memory vector store."""
        self.dimension = dimension
        self.metric = metric
        self.vectors: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self.metadata: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self.id_to_idx: Dict[str, int] = {}
        
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]], ids: List[str]):
        """Add vectors to in-memory store."""
        if vectors.shape[0] != len(metadata) or vectors.shape[0] != len(ids):
            raise ValueError("Number of vectors, metadata entries, and IDs must match")
            
        start_idx = len(self.ids)
        
        # Concatenate new vectors
        if len(self.vectors) == 0:
            self.vectors = vectors.astype(np.float32)
        else:
            self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])
        
        # Add metadata and IDs
        self.metadata.extend(metadata)
        self.ids.extend(ids)
        
        # Update ID to index mapping
        for i, vec_id in enumerate(ids):
            self.id_to_idx[vec_id] = start_idx + i
    
    def search(self, query_vector: np.ndarray, top_k: int = 10, 
               filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search using cosine similarity or L2 distance."""
        if len(self.vectors) == 0:
            return []
        
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        
        if self.metric == "cosine":
            # Normalize vectors
            query_norm = query_vector / np.linalg.norm(query_vector)
            vectors_norm = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True)
            similarities = np.dot(vectors_norm, query_norm.T).flatten()
            # Convert to distances (higher similarity = lower distance)
            distances = 1 - similarities
        else:  # L2 distance
            distances = np.linalg.norm(self.vectors - query_vector, axis=1)
        
        # Get top-k indices
        top_indices = np.argsort(distances)[:top_k]
        
        results = []
        for idx in top_indices:
            if filter_metadata:
                if not self._matches_filter(self.metadata[idx], filter_metadata):
                    continue
            
            results.append({
                "id": self.ids[idx],
                "score": float(distances[idx]),
                "metadata": self.metadata[idx].copy()
            })
        
        return results
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_criteria: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter_criteria.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    def delete(self, ids: List[str]):
        """Delete vectors by IDs."""
        indices_to_remove = []
        for vec_id in ids:
            if vec_id in self.id_to_idx:
                indices_to_remove.append(self.id_to_idx[vec_id])
        
        if indices_to_remove:
            # Remove from vectors, metadata, and ids
            mask = np.ones(len(self.vectors), dtype=bool)
            mask[indices_to_remove] = False
            
            self.vectors = self.vectors[mask]
            self.metadata = [m for i, m in enumerate(self.metadata) if i not in indices_to_remove]
            self.ids = [id_ for i, id_ in enumerate(self.ids) if i not in indices_to_remove]
            
            # Rebuild ID mapping
            self.id_to_idx = {id_: i for i, id_ in enumerate(self.ids)}
    
    def save(self, filepath: str):
        """Save to numpy file."""
        np.save(filepath, {
            'vectors': self.vectors,
            'metadata': self.metadata,
            'ids': self.ids,
            'id_to_idx': self.id_to_idx,
            'dimension': self.dimension,
            'metric': self.metric
        })
    
    def load(self, filepath: str):
        """Load from numpy file."""
        data = np.load(filepath + '.npy', allow_pickle=True).item()
        self.vectors = data['vectors']
        self.metadata = data['metadata']
        self.ids = data['ids']
        self.id_to_idx = data['id_to_idx']
        self.dimension = data['dimension']
        self.metric = data['metric']
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            "total_vectors": len(self.ids),
            "dimension": self.dimension,
            "metric": self.metric,
            "memory_usage_mb": self.vectors.nbytes / (1024 * 1024)
        }
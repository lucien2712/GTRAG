"""
TimeRAG: Temporal-aware Retrieval-Augmented Generation System

A graph-based RAG system designed for temporal data analysis across multiple documents.
Specializes in cross-document queries and time-series analysis.
"""

from .core import TimeRAGSystem
from .config import QueryParams, ChunkingConfig, ModelConfig
from .extractors import LLMExtractor
from .graph import GraphBuilder, GraphRetriever
from .processing import DocumentChunker, TokenManager, BatchProcessor

__version__ = "1.0.0"
__all__ = [
    "TimeRAGSystem",
    "QueryParams", 
    "ChunkingConfig",
    "ModelConfig",
    "LLMExtractor",
    "GraphBuilder", 
    "GraphRetriever",
    "DocumentChunker",
    "TokenManager",
    "BatchProcessor"
]
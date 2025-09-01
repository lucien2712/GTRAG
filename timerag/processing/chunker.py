"""
TimeRAG document chunking module.

This module is responsible for splitting long documents into smaller,
manageable text segments (chunks) based on specified token counts,
while preserving overlapping context to ensure no cross-chunk
relational information is lost during subsequent information extraction.
"""

import logging
import re
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Try to import tiktoken, use fallback if failed
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None
    TIKTOKEN_AVAILABLE = False

from ..config.settings import ChunkingConfig

logger = logging.getLogger(__name__)


# --- Data Structure Definitions ---

@dataclass
class DocumentChunk:
    """Define data structure for a document chunk."""
    content: str
    chunk_id: str
    source_doc_id: str
    chunk_index: int
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary format."""
        return {
            "content": self.content,
            "chunk_id": self.chunk_id,
            "source_doc_id": self.source_doc_id,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "metadata": self.metadata
        }


# --- Core Document Chunking Class ---

class DocumentChunker:
    """
    Core class implementing document chunking functionality.
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize the chunker.
        
        Args:
            config: Chunking configuration parameters
        """
        self.config = config or ChunkingConfig()
        self.tokenizer = self._get_tokenizer()

    def _get_tokenizer(self):
        """Get appropriate tokenizer based on availability."""
        if TIKTOKEN_AVAILABLE:
            try:
                return tiktoken.get_encoding(self.config.ENCODING_NAME)
            except Exception as e:
                logger.warning(f"Failed to load tiktoken encoding {self.config.ENCODING_NAME}: {e}")
                return None
        else:
            logger.warning("tiktoken not available. Using approximate token counting.")
            return None

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken or approximation.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Approximation: 1 token ≈ 4 characters for English
            return len(text) // 4

    def chunk(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """
        Split document into chunks with configurable overlap.
        
        Args:
            doc_id: Document identifier
            text: Document text to chunk
            metadata: Additional metadata to attach to chunks
            
        Returns:
            List of DocumentChunk objects
        """
        metadata = metadata or {}
        
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Check if document is short enough to be a single chunk
        total_tokens = self.count_tokens(cleaned_text)
        if total_tokens <= self.config.MAX_TOKENS_PER_CHUNK:
            chunk_id = f"{doc_id}_chunk_0"
            return [DocumentChunk(
                content=cleaned_text,
                chunk_id=chunk_id,
                source_doc_id=doc_id,
                chunk_index=0,
                token_count=total_tokens,
                metadata={**metadata, "chunk_id": chunk_id}
            )]
        
        # Split into chunks with overlap
        chunks = []
        if self.config.SENTENCE_BOUNDARY:
            chunks = self._chunk_by_sentences(doc_id, cleaned_text, metadata)
        else:
            chunks = self._chunk_by_tokens(doc_id, cleaned_text, metadata)
        
        # Filter chunks that are too small
        filtered_chunks = [
            chunk for chunk in chunks 
            if chunk.token_count >= self.config.MIN_CHUNK_TOKENS
        ]
        
        logger.info(f"Document {doc_id} split into {len(filtered_chunks)} chunks")
        return filtered_chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove empty lines
        text = re.sub(r'\n\s*\n', '\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text

    def _chunk_by_sentences(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Chunk text by sentence boundaries for better context preservation."""
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence would exceed the limit, finalize current chunk
            if current_tokens + sentence_tokens > self.config.MAX_TOKENS_PER_CHUNK and current_chunk:
                chunk_content = " ".join(current_chunk)
                chunk_id = f"{doc_id}_chunk_{chunk_index}"
                
                chunks.append(DocumentChunk(
                    content=chunk_content,
                    chunk_id=chunk_id,
                    source_doc_id=doc_id,
                    chunk_index=chunk_index,
                    token_count=current_tokens,
                    metadata={**metadata, "chunk_id": chunk_id}
                ))
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_tokens = self.count_tokens(" ".join(current_chunk))
                chunk_index += 1
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk if there's remaining content
        if current_chunk:
            chunk_content = " ".join(current_chunk)
            chunk_id = f"{doc_id}_chunk_{chunk_index}"
            
            chunks.append(DocumentChunk(
                content=chunk_content,
                chunk_id=chunk_id,
                source_doc_id=doc_id,
                chunk_index=chunk_index,
                token_count=self.count_tokens(chunk_content),
                metadata={**metadata, "chunk_id": chunk_id}
            ))
        
        return chunks

    def _chunk_by_tokens(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Simple token-based chunking as fallback."""
        if not self.tokenizer:
            # Character-based approximation
            return self._chunk_by_characters(doc_id, text, metadata)
        
        tokens = self.tokenizer.encode(text)
        chunks = []
        chunk_index = 0
        
        start_idx = 0
        while start_idx < len(tokens):
            # Calculate end index
            end_idx = min(start_idx + self.config.MAX_TOKENS_PER_CHUNK, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode back to text
            chunk_content = self.tokenizer.decode(chunk_tokens)
            chunk_id = f"{doc_id}_chunk_{chunk_index}"
            
            chunks.append(DocumentChunk(
                content=chunk_content,
                chunk_id=chunk_id,
                source_doc_id=doc_id,
                chunk_index=chunk_index,
                token_count=len(chunk_tokens),
                metadata={**metadata, "chunk_id": chunk_id}
            ))
            
            # Move to next chunk with overlap
            start_idx = end_idx - self.config.OVERLAP_TOKENS
            chunk_index += 1
        
        return chunks

    def _chunk_by_characters(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Character-based chunking as ultimate fallback."""
        # Approximate token-to-character ratio
        max_chars = self.config.MAX_TOKENS_PER_CHUNK * 4
        overlap_chars = self.config.OVERLAP_TOKENS * 4
        
        chunks = []
        chunk_index = 0
        
        start_idx = 0
        while start_idx < len(text):
            end_idx = min(start_idx + max_chars, len(text))
            chunk_content = text[start_idx:end_idx]
            chunk_id = f"{doc_id}_chunk_{chunk_index}"
            
            chunks.append(DocumentChunk(
                content=chunk_content,
                chunk_id=chunk_id,
                source_doc_id=doc_id,
                chunk_index=chunk_index,
                token_count=self.count_tokens(chunk_content),
                metadata={**metadata, "chunk_id": chunk_id}
            ))
            
            start_idx = end_idx - overlap_chars
            chunk_index += 1
        
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns."""
        # Improved sentence splitting pattern
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # Clean and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences

    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get overlap sentences based on token count."""
        if not sentences:
            return []
        
        overlap_sentences = []
        overlap_tokens = 0
        
        # Take sentences from the end until we reach overlap limit
        for sentence in reversed(sentences):
            sentence_tokens = self.count_tokens(sentence)
            if overlap_tokens + sentence_tokens <= self.config.OVERLAP_TOKENS:
                overlap_sentences.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break
        
        return overlap_sentences

    def get_chunk_statistics(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Get statistics about the chunking results."""
        if not chunks:
            return {
                "total_chunks": 0,
                "total_tokens": 0,
                "avg_tokens_per_chunk": 0,
                "min_tokens": 0,
                "max_tokens": 0
            }
        
        token_counts = [chunk.token_count for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_tokens": sum(token_counts),
            "avg_tokens_per_chunk": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "config_max_tokens": self.config.MAX_TOKENS_PER_CHUNK,
            "config_overlap_tokens": self.config.OVERLAP_TOKENS
        }
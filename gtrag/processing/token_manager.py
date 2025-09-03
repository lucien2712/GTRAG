"""
gtrag Token management and truncation module.

This module contains the core `TokenManager` class, whose main responsibility
is to ensure that the final context assembled for the large language model (LLM)
does not exceed its token length limits.

It intelligently truncates entities, relationships, and source texts according
to configured token limits, and combines them in a reasonable order to preserve
the most important information within limited space.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional

# Try to import tiktoken, use fallback if failed
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None
    TIKTOKEN_AVAILABLE = False

from ..config.settings import QueryParams

logger = logging.getLogger(__name__)


class TokenManager:
    """Manage token counting and content truncation to meet LLM input limits."""
    
    def __init__(self, query_params: Optional[QueryParams] = None, encoding_name: str = "o200k_base"):
        """
        Initialize TokenManager.
        
        Args:
            query_params: Query parameters containing token limits
            encoding_name: Tiktoken encoding name to use
        """
        self.limits = query_params or QueryParams()
        
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoding = tiktoken.get_encoding(encoding_name)
                self._has_tokenizer = True
            except Exception as e:
                logger.warning(f"Failed to load tiktoken encoding '{encoding_name}': {e}. Using fallback.")
                try:
                    self.encoding = tiktoken.get_encoding("cl100k_base")
                    self._has_tokenizer = True
                except Exception:
                    logger.warning("Failed to load any tiktoken encoding. Using approximation.")
                    self.encoding = None
                    self._has_tokenizer = False
        else:
            logger.warning("tiktoken not available. Using token approximation.")
            self.encoding = None
            self._has_tokenizer = False
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in given text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        if self._has_tokenizer and self.encoding:
            return len(self.encoding.encode(str(text)))
        else:
            # Approximation: 1 token ≈ 4 characters for English text
            return len(str(text)) // 4
    
    def prepare_final_context(
        self, 
        retrieved_entities: List[Dict[str, Any]], 
        retrieved_relations: List[Dict[str, Any]], 
        source_chunks: List[str],
        query: str,
        prompt_template: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Prepare final context for LLM generation with intelligent truncation.
        
        Args:
            retrieved_entities: List of retrieved entities
            retrieved_relations: List of retrieved relations
            source_chunks: List of source text chunks
            query: Original user query
            prompt_template: Prompt template to use
            
        Returns:
            Tuple of (final_prompt, token_statistics)
        """
        # Calculate available token budget
        base_prompt_tokens = self.count_tokens(prompt_template.format(
            context_data="", user_query=query
        ))
        
        available_tokens = self.limits.final_max_tokens - base_prompt_tokens
        if available_tokens <= 0:
            logger.warning("No tokens available for content after base prompt")
            return prompt_template.format(context_data="No content available", user_query=query), {
                "total_tokens": base_prompt_tokens,
                "available_tokens": 0,
                "entities_tokens": 0,
                "relations_tokens": 0,
                "chunks_tokens": 0
            }
        
        # Allocate tokens proportionally
        entity_budget = min(self.limits.entity_max_tokens, int(available_tokens * 0.3))
        relation_budget = min(self.limits.relation_max_tokens, int(available_tokens * 0.3))
        chunk_budget = available_tokens - entity_budget - relation_budget
        
        # Truncate each section
        truncated_entities = self._truncate_entities(retrieved_entities, entity_budget)
        truncated_relations = self._truncate_relations(retrieved_relations, relation_budget)
        truncated_chunks = self._truncate_chunks(source_chunks, chunk_budget)
        
        # Format context data
        context_data = self._format_context_data(
            truncated_entities, truncated_relations, truncated_chunks
        )
        
        # Create final prompt
        final_prompt = prompt_template.format(
            context_data=context_data,
            user_query=query
        )
        
        # Calculate final statistics
        final_tokens = self.count_tokens(final_prompt)
        token_stats = {
            "total_tokens": final_tokens,
            "base_prompt_tokens": base_prompt_tokens,
            "available_tokens": available_tokens,
            "entities_tokens": self.count_tokens(truncated_entities),
            "relations_tokens": self.count_tokens(truncated_relations),
            "chunks_tokens": self.count_tokens(truncated_chunks),
            "entity_count": len(retrieved_entities),
            "relation_count": len(retrieved_relations),
            "chunk_count": len(source_chunks)
        }
        
        return final_prompt, token_stats
    
    def _truncate_entities(self, entities: List[Dict[str, Any]], token_budget: int) -> str:
        """Truncate entities section to fit within token budget."""
        if not entities:
            return ""
        
        entity_lines = []
        current_tokens = 0
        
        # Sort entities by relevance score if available
        sorted_entities = sorted(entities, key=lambda x: x.get('score', 0), reverse=True)
        
        for entity in sorted_entities:
            name = entity.get('name', 'Unknown')
            entity_type = entity.get('type', 'Unknown')
            description = entity.get('description', 'No description')
            date = entity.get('date')
            
            # Truncate description if too long
            if self.count_tokens(description) > 200:
                description = self._truncate_text(description, 200)
            
            if date:
                entity_line = f"- **{name}** ({entity_type}) [{date}]: {description}"
            else:
                entity_line = f"- **{name}** ({entity_type}): {description}"
            line_tokens = self.count_tokens(entity_line)
            
            if current_tokens + line_tokens > token_budget:
                break
            
            entity_lines.append(entity_line)
            current_tokens += line_tokens
        
        return "\n".join(entity_lines) if entity_lines else "No entities available"
    
    def _truncate_relations(self, relations: List[Dict[str, Any]], token_budget: int) -> str:
        """Truncate relations section to fit within token budget."""
        if not relations:
            return ""
        
        relation_lines = []
        current_tokens = 0
        
        # Sort relations by relevance score if available
        sorted_relations = sorted(relations, key=lambda x: x.get('score', 0), reverse=True)
        
        for relation in sorted_relations:
            source = relation.get('source', 'Unknown')
            target = relation.get('target', 'Unknown')
            relation_keywords = relation.get('type', 'related to')
            description = relation.get('description', 'No description')
            date = relation.get('date')
            
            # Truncate description if too long
            if self.count_tokens(description) > 150:
                description = self._truncate_text(description, 150)
            
            if date:
                relation_line = f"- **{source}** {relation_keywords} **{target}** [{date}]: {description}"
            else:
                relation_line = f"- **{source}** {relation_keywords} **{target}**: {description}"
            line_tokens = self.count_tokens(relation_line)
            
            if current_tokens + line_tokens > token_budget:
                break
            
            relation_lines.append(relation_line)
            current_tokens += line_tokens
        
        return "\n".join(relation_lines) if relation_lines else "No relations available"
    
    def _truncate_chunks(self, chunks: List[str], token_budget: int) -> str:
        """Truncate chunks section to fit within token budget."""
        if not chunks:
            return ""
        
        chunk_lines = []
        current_tokens = 0
        
        for i, chunk in enumerate(chunks, 1):
            # Truncate individual chunks if necessary
            max_chunk_tokens = min(800, token_budget // max(len(chunks), 1))
            truncated_chunk = self._truncate_text(chunk, max_chunk_tokens)
            
            chunk_line = f"**Chunk {i}:** {truncated_chunk}"
            line_tokens = self.count_tokens(chunk_line)
            
            if current_tokens + line_tokens > token_budget:
                break
            
            chunk_lines.append(chunk_line)
            current_tokens += line_tokens
        
        return "\n\n".join(chunk_lines) if chunk_lines else "No chunks available"
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to specified token limit."""
        if not text:
            return ""
        
        current_tokens = self.count_tokens(text)
        if current_tokens <= max_tokens:
            return text
        
        if self._has_tokenizer and self.encoding:
            # Use precise token-based truncation
            tokens = self.encoding.encode(text)
            if len(tokens) > max_tokens:
                truncated_tokens = tokens[:max_tokens]
                return self.encoding.decode(truncated_tokens) + "..."
        else:
            # Character-based approximation
            estimated_chars = max_tokens * 4
            if len(text) > estimated_chars:
                return text[:estimated_chars] + "..."
        
        return text
    
    def _format_context_data(self, entities: str, relations: str, chunks: str) -> str:
        """Format context data into structured sections."""
        sections = []
        
        if entities.strip():
            sections.append(f"### Entities\n{entities}")
        
        if relations.strip():
            sections.append(f"### Relationships\n{relations}")
        
        if chunks.strip():
            sections.append(f"### Document Chunks\n{chunks}")
        
        return "\n\n".join(sections) if sections else "No context data available"
    
    def estimate_tokens_for_content(self, entities: List[Dict], relations: List[Dict], chunks: List[str]) -> Dict[str, int]:
        """Estimate token requirements for given content without truncation."""
        entity_lines = []
        for e in entities:
            name = e.get('name', 'Unknown')
            entity_type = e.get('type', 'Unknown')
            description = e.get('description', 'No description')
            date = e.get('date')
            if date:
                entity_lines.append(f"- **{name}** ({entity_type}) [{date}]: {description}")
            else:
                entity_lines.append(f"- **{name}** ({entity_type}): {description}")
        entity_text = "\n".join(entity_lines)
        
        relation_lines = []
        for r in relations:
            source = r.get('source', 'Unknown')
            target = r.get('target', 'Unknown')
            relation_keywords = r.get('type', 'related to')
            description = r.get('description', 'No description')
            date = r.get('date')
            if date:
                relation_lines.append(f"- **{source}** {relation_keywords} **{target}** [{date}]: {description}")
            else:
                relation_lines.append(f"- **{source}** {relation_keywords} **{target}**: {description}")
        relation_text = "\n".join(relation_lines)
        
        chunk_text = "\n\n".join([f"**Chunk {i+1}:** {chunk}" for i, chunk in enumerate(chunks)])
        
        return {
            "entities": self.count_tokens(entity_text),
            "relations": self.count_tokens(relation_text),
            "chunks": self.count_tokens(chunk_text),
            "total": self.count_tokens(f"{entity_text}\n\n{relation_text}\n\n{chunk_text}")
        }
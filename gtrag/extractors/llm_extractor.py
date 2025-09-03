"""
gtrag information extraction module.

This module contains the core `LLMExtractor` class, which is responsible for calling
large language models (LLM) to extract structured information from text,
including entities and relationships.

It uses `PromptTemplates` to generate appropriate prompts and supports both
standard OpenAI API and custom LLM functions for extensibility.
"""

import json
import logging
from typing import List, Dict, Any, Callable, Tuple, Optional
from dataclasses import dataclass, field

# Try to import openai, set to None if failed
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False

from ..config.settings import ModelConfig
from ..config.prompts import PromptConfig
from ..config.entity_types import EntityTypes

# Import prompt constants for fallback
try:
    import sys
    from pathlib import Path
    config_path = Path(__file__).parent.parent.parent / "configs" / "prompts.py"
    if config_path.exists():
        sys.path.insert(0, str(config_path.parent))
        from prompts import KEYWORDS_EXTRACTION_PROMPT
    else:
        KEYWORDS_EXTRACTION_PROMPT = "Extract keywords from: {query}"
except ImportError:
    KEYWORDS_EXTRACTION_PROMPT = "Extract keywords from: {query}"

logger = logging.getLogger(__name__)

# --- Data Structure Definitions ---

@dataclass
class Entity:
    """Define data structure for an extracted entity."""
    name: str
    type: str
    description: str
    source_doc_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash((self.name, self.type))

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary format."""
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "source_doc_id": self.source_doc_id,
            "metadata": self.metadata
        }

@dataclass
class Relation:
    """Define data structure for an extracted relationship."""
    source: str
    target: str
    keywords: str  # relationship_keywords from LLM extraction
    description: str
    evidence: str  # Direct evidence from original text
    source_doc_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert relation to dictionary format."""
        return {
            "source": self.source,
            "target": self.target,
            "keywords": self.keywords,  # relationship_keywords
            "description": self.description,
            "evidence": self.evidence,
            "source_doc_id": self.source_doc_id,
            "metadata": self.metadata
        }


# --- Prompt Template Generator ---

class PromptTemplates:
    """
    Dynamically generate extraction prompts based on configuration files.
    """
    def __init__(self):
        # Get prompt configuration from singleton
        self.prompt_config = PromptConfig.get_instance()
        self.entity_types = EntityTypes.get_instance()

    def get_extraction_prompt(self, text: str) -> Dict[str, str]:
        """Get entity and relation extraction system and user prompts."""
        config = self.prompt_config.get_extraction_prompt()
        system_prompt = config.get("system_prompt", "")
        
        # Get entity types for prompt
        entity_type_names = self.entity_types.get_all_types()
        
        # Format the extraction prompt with actual values
        user_prompt = self.prompt_config.format_extraction_prompt(
            text, entity_type_names, language="English"
        )
        
        return {"system": system_prompt, "user": user_prompt}

    def get_query_understanding_prompt(self, query: str) -> Dict[str, str]:
        """Get query understanding system and user prompts."""
        config = self.prompt_config.get_query_understanding_prompt()
        system_prompt = config.get("system_prompt", "")
        template = config.get("template", KEYWORDS_EXTRACTION_PROMPT)
        user_prompt = template.format(query=query)
        return {"system": system_prompt, "user": user_prompt}


# --- Core Extractor ---

class LLMExtractor:
    """
    Use large language models (LLM) to extract entities and relationships from text.
    
    This class encapsulates all logic for interacting with LLMs, including:
    1. Initialize client (OpenAI or custom function)
    2. Use PromptTemplates to generate prompts
    3. Send requests and parse returned structured data
    4. Error handling
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: Optional[str] = None, 
                 llm_func: Optional[Callable[[str, str], str]] = None):
        """
        Initialize LLMExtractor.

        Supports two modes:
        1. `api_key` mode: Use standard OpenAI API
        2. `llm_func` mode: Use a user-defined function to call any LLM

        Args:
            api_key: OpenAI API key (optional)
            model: Model name to use, defaults from ModelConfig
            llm_func: Custom LLM calling function with signature func(system_prompt: str, user_prompt: str) -> str
        """
        if llm_func:
            self.llm_call = self._custom_llm_call
            self.llm_func = llm_func
        elif api_key and OPENAI_AVAILABLE:
            self.client = openai.OpenAI(api_key=api_key)
            self.llm_call = self._openai_call
        else:
            raise ValueError(
                "Must provide either OpenAI `api_key` or a custom `llm_func`. "
                "If using api_key, ensure `openai` package is installed."
            )
            
        self.model = model or ModelConfig.DEFAULT_MODEL
        self.temperature = ModelConfig.TEMPERATURE
        self.prompt_templates = PromptTemplates()

    def _custom_llm_call(self, system_prompt: str, user_prompt: str) -> str:
        """Use custom LLM function for calls."""
        return self.llm_func(system_prompt, user_prompt)

    def _openai_call(self, system_prompt: str, user_prompt: str) -> str:
        """Use OpenAI API for calls."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"}  # Request JSON output
        )
        return response.choices[0].message.content

    def extract(self, text: str, doc_id: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[List[Entity], List[Relation]]:
        """
        Extract entities and relationships from a single text segment.

        Args:
            text: Text segment to extract information from
            doc_id: Source document ID
            metadata: Metadata to attach to entities and relationships

        Returns:
            Tuple of (entities list, relations list)
        """
        prompts = self.prompt_templates.get_extraction_prompt(text)
        
        try:
            response_text = self.llm_call(prompts["system"], prompts["user"])
            
            # Debug logging to see what Gemini actually returns
            logger.info(f"LLM response for {doc_id} (first 500 chars): {response_text[:500]}")
            
            # Parse LLM response in the format "('entity'<|>...)" 
            entities, relations = self._parse_extraction_output(response_text, text, doc_id, metadata or {})
            
            if not entities and not relations:
                logger.warning(f"No entities/relations extracted from {doc_id}. Response format may be incorrect.")
            
            return entities, relations
        except Exception as e:
            logger.error(f"Error extracting information from document ID '{doc_id}': {e}")
            return [], []

    def extract_keywords(self, query: str) -> Dict[str, Any]:
        """
        Extract keywords from user query for retrieval.
        
        Args:
            query: User's question
            
        Returns:
            Dictionary with extracted keywords
        """
        prompts = self.prompt_templates.get_query_understanding_prompt(query)
        
        try:
            response_text = self.llm_call(prompts["system"], prompts["user"])
            # Parse JSON response
            query_analysis = json.loads(response_text)
            return {
                "original_query": query,
                "high_level_keywords": query_analysis.get("high_level_keywords", []),
                "low_level_keywords": query_analysis.get("low_level_keywords", [])
            }
        except Exception as e:
            logger.error(f"Error extracting keywords from query '{query}': {e}")
            return {
                "original_query": query,
                "high_level_keywords": [],
                "low_level_keywords": []
            }

    def _parse_extraction_output(self, response_text: str, text: str, doc_id: str, metadata: Dict[str, Any]) -> Tuple[List[Entity], List[Relation]]:
        """Parse LLM JSON output for entities and relationships."""
        entities = []
        relations = []
        
        try:
            import json
            json_response = json.loads(response_text.strip())
            logger.info(f"Received JSON response from LLM with {len(json_response.get('entities', []))} entities, {len(json_response.get('relationships', []))} relations")
            
            # Parse entities
            json_entities = json_response.get('entities', [])
            for entity_data in json_entities:
                entity = Entity(
                    name=entity_data.get('entity_name', ''),
                    type=entity_data.get('entity_type', 'Other'),
                    description=entity_data.get('entity_description', ''),
                    source_doc_id=doc_id,
                    metadata={**metadata, 'chunk_id': metadata.get('chunk_id')}
                )
                entities.append(entity)
            
            # Parse relationships
            json_relations = json_response.get('relationships', [])
            for relation_data in json_relations:
                relation = Relation(
                    source=relation_data.get('source_entity', ''),
                    target=relation_data.get('target_entity', ''),
                    keywords=relation_data.get('relationship_keywords', 'related'),
                    description=relation_data.get('relationship_description', ''),
                    evidence=text[:200] + "..." if len(text) > 200 else text,
                    source_doc_id=doc_id,
                    metadata={**metadata, 'chunk_id': metadata.get('chunk_id')}
                )
                relations.append(relation)
            
            logger.info(f"Successfully parsed JSON format: {len(entities)} entities, {len(relations)} relations")
            return entities, relations
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from LLM: {e}")
            logger.error(f"Response text: {response_text[:500]}...")
            return [], []

    def get_stats(self) -> Dict[str, Any]:
        """Get extractor statistics."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "has_custom_llm": hasattr(self, 'llm_func'),
            "openai_available": OPENAI_AVAILABLE
        }
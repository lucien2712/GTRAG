"""
Prompt configuration management for gtrag system.

This class dynamically loads prompt templates from the prompts module,
allowing users to centrally manage all LLM interaction prompts.
Uses singleton pattern for consistency.
"""
import sys
import importlib.util
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class PromptConfig:
    """
    Load and manage prompt templates from Python module.
    
    This class dynamically loads the 'configs/prompts.py' file,
    allowing users to centrally manage all prompts for LLM interactions.
    Uses singleton pattern.
    """
    _instance = None
    _prompt_config: Dict[str, Any] = field(default_factory=dict, init=False)
    _prompts_module: Any = field(default=None, init=False)

    def __post_init__(self):
        self._load_prompts()

    def _load_prompts(self):
        """Dynamically load prompt templates from Python module."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "prompts.py"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Prompts configuration file 'prompts.py' not found in 'configs' folder: {config_path}"
            )

        spec = importlib.util.spec_from_file_location("prompts_config", config_path)
        prompts_module = importlib.util.module_from_spec(spec)
        if spec.loader:
            spec.loader.exec_module(prompts_module)
        
        self._prompts_module = prompts_module
        if hasattr(prompts_module, 'get_all_configs'):
            self._prompt_config = prompts_module.get_all_configs()

    @classmethod
    def get_instance(cls):
        """Get singleton instance of PromptConfig."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_entity_extraction_prompt(self) -> Dict[str, Any]:
        """Get entity extraction prompt template."""
        return self._prompt_config.get("entity_extraction", {})

    def get_relation_extraction_prompt(self) -> Dict[str, Any]:
        """Get relation extraction prompt template."""
        return self._prompt_config.get("relation_extraction", {})

    def get_query_understanding_prompt(self) -> Dict[str, Any]:
        """Get query understanding prompt template."""
        return self._prompt_config.get("query_understanding", {})

    def get_result_summarization_prompt(self) -> Dict[str, Any]:
        """Get result summarization prompt template."""
        return self._prompt_config.get("rag_response", {})
    
    def reload_config(self):
        """Reload prompt configuration (useful for development)."""
        if self._prompts_module:
            importlib.reload(self._prompts_module)
            if hasattr(self._prompts_module, 'get_all_configs'):
                self._prompt_config = self._prompts_module.get_all_configs()
                
    def format_extraction_prompt(self, text: str, entity_types: list, language: str = "English") -> str:
        """Format entity extraction prompt with actual values."""
        if hasattr(self._prompts_module, 'format_extraction_prompt'):
            return self._prompts_module.format_extraction_prompt(text, entity_types, language)
        return text
        
    def format_rag_prompt(self, context_data: str, user_query: str, response_type: str = "comprehensive answer") -> str:
        """Format RAG response prompt with actual values.""" 
        if hasattr(self._prompts_module, 'format_rag_response_prompt'):
            return self._prompts_module.format_rag_response_prompt(context_data, user_query, response_type)
        return user_query
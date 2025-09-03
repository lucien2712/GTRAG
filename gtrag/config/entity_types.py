"""
Entity types configuration management.

Loads and manages entity type definitions from JSON configuration files.
Uses singleton pattern to ensure global consistency.
"""
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path


@dataclass
class EntityTypes:
    """
    Load and manage entity type definitions from JSON configuration.
    
    This class reads the 'configs/entity_types.json' file and dynamically
    sets entity types as class attributes for easy access in code.
    Uses singleton pattern to ensure global consistency.
    """
    _instance = None
    _entity_config: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self._load_entity_types()

    def _load_entity_types(self):
        """Load entity types from JSON configuration file."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "entity_types.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Entity types configuration file 'entity_types.json' not found: {config_path}"
            )

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self._entity_config = config
        
        # Dynamically set entity types as class attributes, e.g., self.COMPANY = "COMPANY"
        if "entity_types" in config:
            for entity_type in config["entity_types"]:
                setattr(self, entity_type, entity_type)

    @classmethod
    def get_instance(cls):
        """Get singleton instance of EntityTypes."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_all_types(self) -> List[str]:
        """Get list of all entity type names."""
        return list(self._entity_config.get("entity_types", {}).keys())

    def get_descriptions(self) -> Dict[str, str]:
        """Get all entity type Chinese descriptions."""
        return {
            entity_type: details.get("description_zh", "")
            for entity_type, details in self._entity_config.get("entity_types", {}).items()
        }

    def get_descriptions_en(self) -> Dict[str, str]:
        """Get all entity type English descriptions.""" 
        return {
            entity_type: details.get("description_en", "")
            for entity_type, details in self._entity_config.get("entity_types", {}).items()
        }
        
    def get_examples(self, entity_type: str) -> List[str]:
        """Get examples for a specific entity type."""
        entity_data = self._entity_config.get("entity_types", {}).get(entity_type, {})
        return entity_data.get("examples", [])
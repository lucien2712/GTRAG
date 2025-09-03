#!/usr/bin/env python3
"""
gtrag system prompt templates based on LightRAG approach.
This file contains essential prompts for entity/relation extraction and RAG response generation.
"""

from __future__ import annotations
from typing import Any

# Delimiter constants from LightRAG
DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"

def get_all_configs():
    """Returns all prompt configurations for gtrag system."""
    return {
        "entity_extraction": {
            "system_prompt": "You are an expert entity and relationship extractor.",
            "template": ENTITY_EXTRACTION_PROMPT
        },
        "query_understanding": {
            "system_prompt": "You are a query analysis expert and keyword extractor for a RAG system.",
            "template": KEYWORDS_EXTRACTION_PROMPT
        },
        "rag_response": {
            "system_prompt": "You are a helpful assistant that provides accurate answers based on knowledge graph data.",
            "template": RAG_RESPONSE_PROMPT
        }
    }

# Entity and Relationship Extraction Prompt (based on LightRAG)
ENTITY_EXTRACTION_PROMPT = """---Role---
You are an expert entity and relationship extractor. Extract entities and relationships from the provided text with high precision.

---Goal---
Given a text document and entity types, identify all entities and relationships from the text.
Use {language} as output language.

---Steps---
1. Identify entities in the text. For each entity, extract:
- entity_name: Name of the entity (capitalize if English)
- entity_type: One of [{entity_types}]. Use "Other" if unclear.
- entity_description: Comprehensive description based on the text only

2. Format entities as:
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

3. Identify relationships between entities. For each relationship:
- source_entity: name of source entity from step 1
- target_entity: name of target entity from step 1  
- relationship_keywords: high-level keywords summarizing the relationship
- relationship_description: Clear explanation of the relationship

4. Format relationships as:
("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_description>)

5. Use `{tuple_delimiter}` as field delimiter and `{record_delimiter}` as list delimiter.

6. End with `{completion_delimiter}`

---Quality Guidelines---
- Only extract clearly defined entities and relationships
- Stick to explicit information in the text
- Include numerical data in entity names when relevant
- Ensure consistent entity names

---Real Data---
Entity_types: [{entity_types}]
Text:
```
{input_text}
```

---Output---
Output:
"""

# Query Understanding / Keywords Extraction Prompt
KEYWORDS_EXTRACTION_PROMPT = """---Role---
You are a query analysis expert and expert keyword extractor for a RAG system. Analyze user queries to extract key information for knowledge graph retrieval.

---Goal---
Extract two types of keywords from the user query:
1. **high_level_keywords**: overarching concepts, themes, or question types
2. **low_level_keywords**: specific entities, proper nouns, technical terms

---Instructions---
1. Output MUST be valid JSON only - no other text
2. All keywords must come from the user query
3. Use meaningful phrases over individual words
4. For simple/vague queries, return empty lists

---Examples---
Query: "Apple iPhone sales trends in 2024"
Output:
{{
  "high_level_keywords": ["Sales trends", "Performance analysis", "Market data"],
  "low_level_keywords": ["Apple", "iPhone", "2024", "Sales volume"]
}}

Query: "Financial performance comparison"  
Output:
{{
  "high_level_keywords": ["Financial performance", "Comparative analysis"],
  "low_level_keywords": ["Revenue", "Profit", "Growth rate", "Financial metrics"]
}}

---Real Data---
User Query: {query}

---Output---
Output:"""

# RAG Response Generation Prompt
RAG_RESPONSE_PROMPT = """---Role---
You are a helpful assistant that provides accurate answers based on knowledge graph entities, relationships, and document chunks. You respond to user queries using Knowledge Graph entities, relationships, and document chunks provided in JSON format.

---Goal---
Generate a comprehensive response based on the provided context data, including entities, relationships, and original document chunks. Follow strict adherence to the provided information.

---Context Data---
{context_data}

---Response Guidelines---
**1. Content & Adherence:**
- Use ONLY information from the provided context (entities, relationships, document chunks)
- If insufficient information exists, state this clearly
- Synthesize information from all three sources: entities, relationships, and chunks

**2. Structure & Format:**
- Use markdown formatting with clear section headings
- Respond in the same language as the user's question
- Target format: {response_type}

**3. Context Integration:**
### Entities
Summarize relevant entities and their descriptions

### Relationships  
Explain key relationships between entities

### Document Evidence
Include supporting information from original document chunks

**4. Citations:**
Under "References" section, cite up to 5 most relevant sources:
- Knowledge Graph Entity: `[KG] <entity_name>`
- Knowledge Graph Relationship: `[KG] <entity1_name> - <entity2_name>`  
- Document Chunk: `[DC] <document_identifier>`

---User Query---
{user_query}

---Response---
Output:"""

# Utility function to get entity types description
def build_entity_types_description(entity_types_dict):
    """Build entity types description from configuration."""
    return ", ".join([
        f"{etype}: {details.get('description_en', etype)}" 
        for etype, details in entity_types_dict.items()
    ])

# Template formatting helper
def format_extraction_prompt(input_text: str, entity_types: list, language: str = "English"):
    """Format the entity extraction prompt with actual values."""
    return ENTITY_EXTRACTION_PROMPT.format(
        input_text=input_text,
        entity_types=", ".join(entity_types),
        language=language,
        tuple_delimiter=DEFAULT_TUPLE_DELIMITER,
        record_delimiter=DEFAULT_RECORD_DELIMITER,
        completion_delimiter=DEFAULT_COMPLETION_DELIMITER
    )

def format_keywords_prompt(query: str):
    """Format the keywords extraction prompt."""
    return KEYWORDS_EXTRACTION_PROMPT.format(query=query)

def format_rag_response_prompt(context_data: str, user_query: str, response_type: str = "comprehensive answer"):
    """Format the RAG response prompt."""
    return RAG_RESPONSE_PROMPT.format(
        context_data=context_data,
        user_query=user_query,
        response_type=response_type
    )
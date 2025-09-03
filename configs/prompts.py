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

# Entity and Relationship Extraction Prompt (JSON Format)
ENTITY_EXTRACTION_PROMPT = """---Role---
You are an expert entity and relationship extractor. Extract entities and relationships from the provided text with high precision.

---Goal---
Extract all relevant entities and relationships from the text and return them in JSON format.
Use {language} as output language.

---Instructions---
1. Identify entities in the text. For each entity, extract:
   - entity_name: Name of the entity (capitalize if English)
   - entity_type: One of [{entity_types}]. Use "Other" if unclear.
   - entity_description: Comprehensive description based on the text only

2. Identify relationships between entities. For each relationship:
   - source_entity: name of source entity from step 1
   - target_entity: name of target entity from step 1  
   - relationship_keywords: high-level keywords summarizing the relationship
   - relationship_description: Clear explanation of the relationship

3. Return ONLY valid JSON in this exact format:
{{
  "entities": [
    {{
      "entity_name": "Entity Name",
      "entity_type": "ENTITY_TYPE",
      "entity_description": "Description of the entity"
    }}
  ],
  "relationships": [
    {{
      "source_entity": "Source Entity Name",
      "target_entity": "Target Entity Name", 
      "relationship_keywords": "High-level keywords",
      "relationship_description": "Description of relationship"
    }}
  ]
}}

---Quality Guidelines---
- Only extract clearly defined entities and relationships
- Stick to explicit information in the text
- Include numerical data in entity names when relevant
- Ensure consistent entity names
- Return valid JSON only, no additional text

---Real Data---
Entity_types: [{entity_types}]
Text:
```
{input_text}
```

---JSON Output---"""

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
You are a helpful assistant that provides comprehensive, well-structured answers based on knowledge graph entities, relationships, and document chunks.

---Goal---
Generate a natural, coherent response that answers the user's question using the provided context data. Write in complete paragraphs with clear, flowing narrative.

---Context Data---
{context_data}

---Response Guidelines---
**1. Content & Adherence:**
- Use ONLY information from the provided context (entities, relationships, document chunks)
- If insufficient information exists, state this clearly
- Synthesize information from all three sources naturally

**2. Structure & Format:**
- Write in natural, flowing paragraphs
- Start with a direct answer to the user's question
- Provide supporting details and evidence
- Use clear, professional language
- Respond in the same language as the user's question

**3. Writing Style:**
- Begin with key findings that directly address the query
- Follow with supporting evidence and details
- Integrate entity information, relationships, and document evidence naturally
- End with any relevant conclusions or insights

**4. Citations:**
At the end, include a "References:" section listing source documents:
- Format: "References: doc_id_1, doc_id_2, doc_id_3"
- Use actual document IDs from the provided chunks

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
# TimeRAG

A temporal-aware Retrieval-Augmented Generation (RAG) framework designed for analyzing and understanding cross-temporal information, particularly suited for financial report analysis, market research, and other time-series data analysis scenarios.

---

## Core Features

- **Temporal-Aware Knowledge Graph**: Automatically builds entity states at different time points and connects them into evolution paths.
- **Knowledge Graph-Driven**: Converts unstructured text into structured knowledge graphs for precise information retrieval.  
- **Centralized Model Configuration**: All LLM and embedding models configured once during system initialization, simplifying subsequent calls.
- **Highly Customizable**: All system components (entity types, prompts, model parameters) are easily configurable.
- **Custom Model Support**: Easy integration of any custom LLM and embedding model functions.

## System Architecture

TimeRAG operates through the following workflow:

```
Documents -> [1. Chunking] -> [2. LLM Extraction] -> [3. Graph Building] -> [4. Temporal Linking]

User Query -> [5. Query Understanding] -> [6. Graph Retrieval] -> [7. Context Assembly] -> [8. LLM Answer Generation]
```

### Core Components

1. **Document Chunking**: Split long documents into manageable segments
2. **Information Extraction**: Use LLM to extract entities and relationships from each segment  
3. **Knowledge Graph Construction**: Store extracted information in `networkx` graph structure
4. **Temporal Linking**: Build "evolution" relationship edges between same entities across different time periods
5. **Query Understanding**: Use LLM to analyze user questions and extract keywords and intent
6. **Graph Retrieval**: Search knowledge graph for nodes and paths most relevant to the question
7. **Context Assembly**: Combine retrieved information and truncate based on token limits
8. **Answer Generation**: Pass assembled context and original question to LLM for final answer

## Installation

```bash
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file with your API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Quick Start

The following example shows how to initialize the system, index documents, and make queries. You can run `examples/demo.py` directly to see it in action.

```python
# Adapted from examples/demo.py

import os
from dotenv import load_dotenv
from timerag import TimeRAGSystem, QueryParams

# Load environment variables from .env file
load_dotenv()

# --- Custom Model Functions (Optional) ---
def gpt_4o_mini_llm(system_prompt: str, user_prompt: str) -> str:
    # ... Implementation details for calling OpenAI API omitted ...
    pass

def openai_embedding_func(text: str) -> list:
    # ... Implementation details for calling OpenAI Embedding API omitted ...
    pass

# 1. Initialize System
# All models and API keys are configured here.
# If llm_func or embedding_func are not provided, system uses default OpenAI models.
rag = TimeRAGSystem(
    llm_func=gpt_4o_mini_llm,
    embedding_func=openai_embedding_func
)

# 2. Insert Documents
# Quarter information passed through metadata is key for temporal awareness
documents = [
    {
        "text": "Apple Inc. iPhone sales reached 80 million units in Q4 2023.", 
        "doc_id": "apple_q4_2023", 
        "metadata": {"quarter": "2023Q4"}
    },
    {
        "text": "By Q1 2024, Apple's iPhone sales grew to 90 million units due to new model releases.", 
        "doc_id": "apple_q1_2024", 
        "metadata": {"quarter": "2024Q1"}
    },
]

for doc in documents:
    rag.insert(doc["text"], doc["doc_id"], doc["metadata"])

# 3. Build Temporal Links
# After indexing all documents, execute this step to create cross-temporal associations
rag.build_temporal_links()

# 4. Query the System
question = "What are the trends in Apple iPhone sales?"
result = rag.query(question)

# 5. View Results
print("Answer:", result.get("answer"))
print("Token Usage:", result.get("token_stats"))
```

## Complete Workflow Example

### Step 1: System Initialization
```python
from timerag import TimeRAGSystem, QueryParams, ChunkingConfig

# Basic initialization with default models
rag = TimeRAGSystem()

# Or with custom configuration
chunk_config = ChunkingConfig(
    MAX_TOKENS_PER_CHUNK=2000,
    OVERLAP_TOKENS=300
)

query_params = QueryParams(
    top_k=15,
    similarity_threshold=0.3,
    max_hops=4
)

rag = TimeRAGSystem(
    chunking_config=chunk_config,
    query_params=query_params
)
```

### Step 2: Document Processing
```python
# Financial report example
documents = [
    {
        "text": "Microsoft reported cloud revenue growth of 30% in Q1 2024, driven by Azure services expansion.",
        "doc_id": "msft_q1_2024_earnings",
        "metadata": {"quarter": "2024Q1"}
    },
    {
        "text": "In Q2 2024, Microsoft's cloud business continued strong performance with 35% growth year-over-year.",
        "doc_id": "msft_q2_2024_earnings", 
        "metadata": {"quarter": "2024Q2"}
    }
]

# Insert documents
for doc in documents:
    rag.insert(doc["text"], doc["doc_id"], doc["metadata"])
    
# Build temporal connections (essential step)
rag.build_temporal_links()
```

### Step 3: Querying
```python
# Query with default parameters
result = rag.query("How is Microsoft's cloud business performing over time?")

# Query with custom parameters
custom_params = QueryParams(
    top_k=20,
    similarity_threshold=0.2,
    max_hops=3
)

result = rag.query(
    "Compare Microsoft's cloud growth trends across quarters",
    query_params=custom_params
)

# Access comprehensive results
answer = result["answer"]
entities = result["retrieved_entities"] 
relations = result["retrieved_relations"]
source_chunks = result["retrieved_source_chunks"]
token_stats = result["token_stats"]
```

### Step 4: Advanced Usage
```python
# Save/load knowledge graph
rag.save_graph("my_knowledge_graph.pkl")

# Later session
new_rag = TimeRAGSystem()
new_rag.load_graph("my_knowledge_graph.pkl")

# Get system statistics
stats = rag.get_stats()
print(f"Indexed {stats['indexed_documents']} documents")
print(f"Graph has {stats['num_nodes']} nodes, {stats['num_edges']} edges")
```

## Configuration

You can easily customize the system by modifying configuration files in the `timerag/config/` directory:

- **Entity Types**: Modify `configs/entity_types.json` to add, remove, or modify entity types and their descriptions
- **Prompts**: Modify `configs/prompts.py` to change how the system interacts with LLMs
- **Parameters**: Use `QueryParams` and `ChunkingConfig` classes for runtime parameter adjustment

### Entity Types Configuration
```json
{
  "entity_types": {
    "COMPANY": {
      "description_en": "Companies, enterprises, organizations",
      "examples": ["Apple", "Microsoft", "Google"],
      "extraction_hints": ["Company name", "Corporation", "Enterprise"]
    }
  }
}
```

### Custom Prompt Templates
```python
# In configs/prompts.py
RAG_RESPONSE_PROMPT = """
Generate a comprehensive response based on:

### Entities
{entities}

### Relationships
{relationships}  

### Document Evidence
{chunks}

Answer the user's question: {user_query}
"""
```

## Key Implementation Notes

1. **Temporal Metadata**: Always include `quarter` information in document metadata for time-aware functionality
2. **Two-Step Process**: Document insertion must be followed by `build_temporal_links()` to complete graph construction
3. **Custom Models**: The system supports custom LLM and embedding functions passed during initialization
4. **Token Management**: Automatic token limit management prevents exceeding model context limits
5. **Graph Storage**: Uses NetworkX for graph operations - no external graph database required

## Package Structure

```
timerag/
├── core/              # Core system orchestrator
├── config/            # Configuration management
├── extractors/        # LLM-based information extraction
├── graph/            # Knowledge graph construction and retrieval
├── processing/       # Document chunking and token management
└── storage/          # Vector database integration
```

## Development

### Running Tests
```bash
# No test suite currently configured
```

### Running Examples
```bash
python examples/demo.py
```

### Custom Development
```python
import timerag

# Access all components
system = timerag.TimeRAGSystem()
extractor = timerag.LLMExtractor()
builder = timerag.GraphBuilder()
retriever = timerag.GraphRetriever(builder, extractor)
```

## Requirements

- Python 3.8+
- OpenAI API key
- Dependencies: `openai`, `networkx`, `sentence-transformers`, `numpy`, `tiktoken`, `pandas`, `textract`, `python-docx`
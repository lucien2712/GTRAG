# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TimeRAG is a graph-based Retrieval-Augmented Generation (RAG) system for temporal data analysis. It specializes in cross-document queries and time-series analysis, particularly for financial report analysis across quarterly earnings.

## Development Commands

### Setup
```bash
pip install -r requirements.txt
```

### Running the System
```bash
# Run the main demo
python examples/demo.py

# Required: Create .env file with OPENAI_API_KEY
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Batch Processing
```bash
# For multi-file processing (via batch_processor.py)
python -c "from batch_processor import BatchProcessor; # custom batch processing"
```

**Note**: This codebase has no test suite or linting configuration currently set up.

## Core Architecture

### Component Hierarchy (in dependency order)
1. **config.py** - Centralized configuration using dataclasses
   - `QueryParams`, `ChunkingConfig`, `EntityTypes`, `PromptConfig`
   - All system parameters controlled here

2. **document_chunker.py** - Text segmentation with intelligent tokenization
   - Handles document splitting with configurable overlap
   - Uses tiktoken for accurate token counting

3. **llm_extractor.py** - LLM-based information extraction  
   - Extracts entities and relationships using GPT-4
   - Supports custom LLM functions via constructor
   - Returns structured JSON

4. **graph_builder.py** - NetworkX-based knowledge graph construction
   - Creates multi-layered temporal graph structure
   - Manages quarterly subgraphs and cross-time connections
   - Supports custom embedding functions

5. **graph_retriever.py** - Intelligent multi-hop retrieval
   - Time-aware semantic search
   - Importance-scored neighbor expansion
   - Configurable retrieval strategies

6. **token_manager.py** - Context length management
   - Prevents exceeding LLM token limits
   - Smart truncation and prioritization

7. **timerag_system.py** - Main orchestrator (`GraphRAGSystem`)
   - Entry point for all operations
   - Integrates all components
   - Provides simple API: insert → build_temporal_links → query

8. **batch_processor.py** - Multi-document processing
   - Handles various file formats (PDF, DOCX, TXT)
   - Concurrent processing capabilities
   - Automatic metadata extraction

## Configuration System

### Key Configuration Files
- **configs/entity_types.json**: Defines extractable entity types (COMPANY, PERSON, PRODUCT, FINANCIAL_METRIC, etc.)
- **configs/prompts.py**: LLM prompt templates for different operations
- **config.py**: System-wide configuration classes with type safety

### Entity Types Available
The system extracts 8 main entity types: COMPANY, PERSON, PRODUCT, FINANCIAL_METRIC, BUSINESS_CONCEPT, MARKET, TECHNOLOGY, GEOGRAPHIC

## Critical Usage Pattern

```python
from timerag_system import GraphRAGSystem
from config import QueryParams

# Initialize (with optional custom functions)
rag = GraphRAGSystem(
    llm_func=custom_llm_function,  # optional
    embedding_func=custom_embedding_function  # optional
)

# Process documents (temporal metadata is crucial)
rag.insert(text, doc_id, metadata={"quarter": "2024Q1"})

# ESSENTIAL: Build temporal connections after all insertions
rag.build_temporal_links()

# Query the system
result = rag.query("Your question", query_params=QueryParams(...))
```

## Key Implementation Details

1. **Two-Phase Process**: Document insertion must be followed by `build_temporal_links()` to complete graph construction
2. **Temporal Metadata**: The `quarter` field in metadata enables time-aware functionality
3. **NetworkX Storage**: Uses NetworkX for graph operations - no external graph database required
4. **Custom Model Support**: Accepts custom LLM and embedding functions during initialization
5. **Token Management**: Automatic context length management prevents model overflow
6. **Multi-Document Processing**: batch_processor.py handles concurrent processing of multiple file formats

## Graph Structure

The system creates a three-layer temporal graph:
- **Time Layer**: Sequential quarters (TIME_2024Q1 → TIME_2024Q2)  
- **Entity Layer**: Within-quarter entity relationships
- **Cross-Time Layer**: temporal_evolution edges connecting same entities across time

## Environment Requirements

- Python 3.8+
- OpenAI API key in .env file
- Dependencies in requirements.txt: openai, networkx, sentence-transformers, numpy, tiktoken, pandas, textract, python-docx
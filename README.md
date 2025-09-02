# TimeRAG

A **temporal-aware Retrieval-Augmented Generation (RAG)** framework that specializes in analyzing time-series data across multiple documents. Perfect for financial reports, market research, and any scenario requiring cross-temporal analysis.

## 🚀 Quick Start

```python
from timerag import TimeRAGSystem

# 1. Initialize system
rag = TimeRAGSystem()

# 2. Index documents with temporal metadata
rag.insert(
    text="Apple reported iPhone sales of 80M units in Q4 2023.", 
    doc_id="apple_q4_2023", 
    metadata={"quarter": "2023Q4"}
)
rag.insert(
    text="Apple's iPhone sales grew to 90M units in Q1 2024.", 
    doc_id="apple_q1_2024", 
    metadata={"quarter": "2024Q1"}
)

# 3. Build temporal connections
rag.build_temporal_links()

# 4. Query the system
result = rag.query("What are the trends in Apple iPhone sales over time?")
print(result["answer"])
```

## ✨ Core Features

- **🕒 Temporal-Aware**: Automatically tracks entity evolution across time periods
- **🧠 Knowledge Graph**: Converts text into structured, queryable knowledge graphs  
- **🎯 Smart Retrieval**: Multi-hop graph traversal with semantic similarity
- **⚙️ Highly Customizable**: Configurable entity types, prompts, and models
- **🔌 Custom Model Support**: Easy integration with any LLM or embedding model

## System Architecture

TimeRAG operates through the following workflow:

```
Documents -> [1. Chunking] -> [2. LLM Extraction] -> [3. Graph Building] -> [4. Temporal Linking]

User Query -> [5. Keywords Extraction] -> [6. Graph Retrieval] -> [7. Context Assembly] -> [8. LLM Answer Generation]
```

### Detailed Workflow

#### 📝 Document Insertion (`rag.insert()`)

1. **Document Chunking** (`processing/chunker.py`)
   - Split document into manageable segments using intelligent tokenization
   - Apply configurable token limits with overlap between chunks
   - Preserve temporal metadata (quarter information) in each chunk

2. **LLM Information Extraction** (`extractors/llm_extractor.py`)
   - Process each chunk through LLM using specialized prompts
   - Extract structured entities (name, type, description) and relationships (source, target, keywords, description)
   - Support 8 entity types: COMPANY, PERSON, PRODUCT, FINANCIAL_METRIC, BUSINESS_CONCEPT, MARKET, TECHNOLOGY, GEOGRAPHIC

3. **Knowledge Graph Building** (`graph/builder.py`)
   - Add extracted entities and relationships to NetworkX graph structure
   - Generate embeddings for semantic similarity (using SentenceTransformer or custom function)
   - Store original chunk content for later context retrieval

4. **Temporal Linking** (`rag.build_temporal_links()`)
   - **Required step**: Must be called after all document insertions
   - Build "temporal_evolution" edges between same entities across different time periods
   - Create three-layer graph structure: Time Layer (quarters) + Entity Layer (within-quarter) + Cross-Time Layer (evolution connections)

#### 🔍 Query Processing (`rag.query()`)

5. **Keywords Extraction** (`extractors/llm_extractor.py`)
   - Use LLM to extract high-level keywords (concepts, themes) and low-level keywords (specific entities, terms) from user question
   - Return structured keyword dictionary for graph retrieval

6. **Multi-Stage Graph Retrieval** (`graph/retriever.py`)
   - **Stage 1**: Semantic search using extracted keywords against entity/relation embeddings
   - **Stage 2**: Multi-hop expansion with configurable depth (max_hops parameter)
   - **Stage 3**: Time-aware filtering (if time_range specified) with temporal relevance scoring
   - **Stage 4**: Relation-entity expansion (automatically include connected nodes from retrieved relationships)
   - Support multiple temporal expansion modes: strict, with_temporal, expanded

7. **Context Assembly** (`processing/token_manager.py`)
   - **Primary Chunks**: Retrieve original text chunks from entities/relationships found in graph
   - **Supplementary Chunks**: Add time-relevant chunks (if time filtering enabled) to provide broader context
   - Apply deduplication to avoid sending same chunk content multiple times to LLM
   - Smart truncation based on token limits while preserving most relevant information

8. **Answer Generation**
   - Format retrieved context (entities + relationships + chunks) into structured prompt
   - Add time range context if filtering was applied
   - Generate comprehensive answer using LLM with proper citations and evidence

### Core Components

1. **Document Chunking**: Split long documents into manageable segments with intelligent tokenization
2. **Information Extraction**: Use LLM to extract entities and relationships from each segment using configurable prompts  
3. **Knowledge Graph Construction**: Store extracted information in `networkx` graph structure with embeddings
4. **Temporal Linking**: Build "temporal_evolution" relationship edges between same entities across different time periods
5. **Keywords Extraction**: Use LLM to extract high-level and low-level keywords from user questions
6. **Graph Retrieval**: Multi-stage retrieval including semantic search, multi-hop expansion, time filtering, and relation-entity expansion
7. **Context Assembly**: Two-stage chunk retrieval with deduplication and smart truncation based on token limits
8. **Answer Generation**: Pass assembled context and original question to LLM for final answer with citations

## Installation

```bash
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file with your API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## 📖 How to Use: Index and Query

### 🔍 **Method 1: Basic Usage**

```python
from timerag import TimeRAGSystem

# Initialize system (uses OpenAI by default)
rag = TimeRAGSystem()

# Index documents - quarter metadata is essential for temporal analysis
rag.insert(
    text="Microsoft cloud revenue grew 30% in Q1 2024, driven by Azure expansion.",
    doc_id="msft_q1_2024",
    metadata={"quarter": "2024Q1"}  # 👈 This is the key for temporal awareness
)

rag.insert(
    text="Microsoft cloud continued strong growth at 35% in Q2 2024.",
    doc_id="msft_q2_2024", 
    metadata={"quarter": "2024Q2"}
)

# Build temporal connections (required after indexing)
rag.build_temporal_links()

# Query with natural language
result = rag.query("How is Microsoft's cloud business performing over time?")
print(result["answer"])
```

### 🎛️ **Method 2: Custom Configuration**

```python
from timerag import TimeRAGSystem, QueryParams

# Custom LLM function (optional)
def my_llm_function(system_prompt: str, user_prompt: str) -> str:
    # Your custom LLM implementation
    # Return JSON string for extraction, plain text for final answers
    pass

def my_embedding_function(text: str) -> list:
    # Your custom embedding implementation
    # Return list of floats
    pass

# Initialize with custom models
rag = TimeRAGSystem(
    llm_func=my_llm_function,
    embedding_func=my_embedding_function
)

# Index multiple documents efficiently
documents = [
    {"text": "...", "doc_id": "doc1", "metadata": {"quarter": "2023Q4"}},
    {"text": "...", "doc_id": "doc2", "metadata": {"quarter": "2024Q1"}},
    {"text": "...", "doc_id": "doc3", "metadata": {"quarter": "2024Q2"}},
]

for doc in documents:
    rag.insert(doc["text"], doc["doc_id"], doc["metadata"])

rag.build_temporal_links()

# Query with custom parameters
custom_params = QueryParams(
    top_k=15,                    # Retrieve top 15 results
    similarity_threshold=0.2,    # Lower threshold = more results
    max_hops=3,                  # Maximum graph traversal hops
    final_max_tokens=8000        # Context size limit
)

result = rag.query(
    "Compare performance trends across all quarters",
    query_params=custom_params
)
```

### 📊 **Method 3: Complete Result Analysis**

```python
# Query returns comprehensive information
result = rag.query("What are the key business trends?")

# Main answer
answer = result["answer"]

# Retrieved context components  
entities = result["retrieved_entities"]       # Extracted entities
relations = result["retrieved_relations"]     # Entity relationships  
chunks = result["retrieved_source_chunks"]    # Original text segments

# Keywords extraction results
keywords = result["query_keywords"]
print(f"High-level keywords: {keywords['high_level_keywords']}")
print(f"Low-level keywords: {keywords['low_level_keywords']}")

# System metrics
token_stats = result["token_stats"]
print(f"Used {token_stats['total_tokens']} tokens")
print(f"Retrieved {len(entities)} entities, {len(relations)} relations")

# Detailed entity information
for entity in entities[:3]:  # Show top 3
    print(f"Entity: {entity['name']} ({entity['type']})")
    print(f"Score: {entity['score']:.3f}")
    print(f"Description: {entity['description']}")

# Relationship information
for relation in relations[:3]:  # Show top 3
    print(f"Relation: {relation['source']} → {relation['target']}")
    print(f"Type: {relation['type']}")  # relationship_keywords
    print(f"Description: {relation['description']}")
```

## 🔧 Advanced Features

### 💾 **Persistence & Reloading**

```python
# Save complete TimeRAG system to working directory
rag.save_graph("./my_timerag_project/")

# Load in a new session
new_rag = TimeRAGSystem()
new_rag.load_graph("./my_timerag_project/")

# Continue querying with loaded data
result = new_rag.query("Previous analysis question")

# Directory structure created:
# my_timerag_project/
# ├── graph.json              # NetworkX knowledge graph
# ├── chunks.json             # Original text chunks  
# ├── vectors.faiss           # Vector index (if enabled)
# └── vectors.metadata.npy    # Vector metadata (if enabled)
```

### 📈 **System Statistics**

```python
# Get detailed statistics
stats = rag.get_stats()
print(f"📁 Indexed documents: {stats['indexed_documents']}")
print(f"🔗 Graph nodes: {stats['num_nodes']}")
print(f"➡️ Graph edges: {stats['num_edges']}")
print(f"📝 Stored chunks: {stats['stored_chunks']}")
```

### ⚙️ **Configuration Options**

```python
from timerag import TimeRAGSystem, QueryParams, ChunkingConfig

# Custom chunking configuration
chunk_config = Config(
    MAX_TOKENS_PER_CHUNK=2000,    # Chunk size
    OVERLAP_TOKENS=300            # Overlap between chunks
)

# Custom query parameters
query_params = QueryParams(
    top_k=15,                     # Number of results to retrieve
    similarity_threshold=0.3,     # Minimum similarity score  
    max_hops=3,                   # Graph traversal depth
    final_max_tokens=8000         # Maximum context tokens
)

# Initialize with custom configurations
rag = TimeRAGSystem(
    chunking_config=chunk_config,
    query_params=query_params
)
```

## ⚠️ **Important Usage Notes**

### 1. **Temporal Metadata is Essential**
```python
# ✅ Correct: Include quarter information
rag.insert(text, doc_id, metadata={"quarter": "2024Q1"})

# ❌ Incorrect: Missing temporal metadata
rag.insert(text, doc_id)  # Won't work properly for temporal analysis
```

### 2. **Two-Step Indexing Process**
```python
# Step 1: Index all documents
for doc in documents:
    rag.insert(doc["text"], doc["doc_id"], doc["metadata"])

# Step 2: Build temporal connections (REQUIRED)
rag.build_temporal_links()  # Don't forget this!
```

### 3. **Quarter Format Standards**
```python
# ✅ Recommended formats
metadata = {"quarter": "2024Q1"}    # Standard format
metadata = {"quarter": "2023Q4"}    # Works for any year
metadata = {"quarter": "2024Q2"}    # Supports all quarters

# ⚠️ Other formats may work but are not guaranteed
metadata = {"quarter": "Q1 2024"}   # May work but not recommended
```

## 🎬 Run the Demo

```bash
cd examples
python demo.py
```

The demo will show you:
- System initialization
- Document indexing with temporal metadata
- Temporal connection building  
- Multiple query examples
- Detailed result analysis

## 🛠️ Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# If you get import errors, try:
pip install --upgrade urllib3 transformers sentence-transformers
```

**2. Missing API Key**
```python
# Set your OpenAI API key
import os
os.environ["OPENAI_API_KEY"] = "your-key-here"

# Or create .env file:
echo "OPENAI_API_KEY=your-key-here" > .env
```

**3. Empty Results**
```python
# Make sure you build temporal links after indexing
rag.build_temporal_links()  # This is required!

# Check your quarter format
metadata = {"quarter": "2024Q1"}  # Use this format
```

**4. Performance Issues**
```python
# Reduce parameters for better performance
query_params = QueryParams(
    top_k=5,                    # Fewer results
    max_hops=2,                 # Shorter graph traversal
    similarity_threshold=0.5    # Higher threshold = fewer results
)
```

## 🏗️ System Architecture

TimeRAG uses a **three-layer temporal graph** structure:

```
Time Layer:    2023Q4 ──→ 2024Q1 ──→ 2024Q2 ──→ 2024Q3

Entity Layer:  Apple──produces──→iPhone    Microsoft──develops──→Azure
               │                   │        │                      │
               │                   │        │                      │
Temporal:      Apple_2024Q1──evolution──→Apple_2024Q2──evolution──→...
```

### Core Components

1. **📝 Document Chunking**: Intelligently splits long documents
2. **🧠 LLM Extraction**: Extracts entities and relationships using customizable prompts
3. **🕸️ Knowledge Graph**: Creates temporal-aware graph with NetworkX
4. **🔤 Keywords Extraction**: Extracts high-level and low-level keywords from queries
5. **🔍 Smart Retrieval**: Multi-hop graph traversal with semantic similarity
6. **🎯 Answer Generation**: Synthesizes retrieved information into coherent answers

## 📁 Project Structure

```
TimeRAG/
├── timerag/                    # Main package
│   ├── core/                  # System orchestrator  
│   ├── config/               # Configuration management
│   ├── extractors/           # LLM-based extraction
│   ├── graph/               # Graph construction & retrieval
│   ├── processing/          # Document processing
│   └── storage/             # Vector database integration
├── configs/                  # Configuration files
│   ├── entity_types.json   # Entity type definitions
│   └── prompts.py          # LLM prompt templates
└── examples/               # Usage examples
    └── demo.py            # Complete demonstration
```

## 🤝 Contributing

This is a research project focused on temporal-aware RAG systems. The core functionality is complete and ready for research and prototyping use cases.

## 📄 License

This project is available for research and educational purposes.
# GTRAG: Graph-based Temporal-aware RAG

A **GTRAG** framework that specializes in analyzing time-series data across multiple documents. Perfect for financial reports, market research, and any scenario requiring cross-temporal analysis.

## 🚀 Quick Start

```python
from gtrag import gtragSystem, QueryParams

# 1. Initialize system
rag = gtragSystem()

# 2. Index documents with temporal metadata (YYYY-MM format)
rag.insert(
    text="Apple reported iPhone sales of 80M units in Q4 2023.", 
    doc_id="apple_q4_2023", 
    metadata={"date": "2023-12"}  # December 2023 (Q4)
)
rag.insert(
    text="Apple launched iPhone 15 on September 15, 2023.", 
    doc_id="apple_launch_2023", 
    metadata={"date": "2023-09"}  # September 2023
)
rag.insert(
    text="Microsoft Azure grew 28% in March 2024.", 
    doc_id="msft_march_2024", 
    metadata={"date": "2024-03"}  # March 2024
)

# 3. Build temporal connections
rag.build_temporal_links()

# 4. Query with flexible time ranges
result = rag.query("What happened in technology in 2023?")
print(result["answer"])

# 5. Query with specific time range filtering

params = QueryParams(
    time_range=["2023-12", "2024-03"],  # Focus on specific periods
    enable_time_filtering=True
)
result = rag.query("Compare Apple and Microsoft performance", query_params=params)
print(result["answer"])
```

## ✨ Core Features

- **🕒 Standardized Time Format**: Single `date` field uses YYYY-MM format (e.g., 2024-01, 2024-12)
- **🧠 Knowledge Graph**: Converts text into structured, queryable knowledge graphs  
- **🎯 Smart Retrieval**: Multi-hop graph traversal with semantic and temporal similarity
- **⚙️ Highly Customizable**: Configurable entity types, prompts, and models
- **🔌 Custom Model Support**: Easy integration with any LLM or embedding model
- **🎯 Simplified API**: Single `date` field with consistent YYYY-MM format

## System Architecture

gtrag operates through the following workflow:

```
Documents -> [1. Chunking] -> [2. LLM Extraction] -> [3. Temporal Graph Building] -> [4. Temporal Evolution Links]

User Query -> [5. Keywords Extraction] -> [6. Time-Aware Retrieval] -> [7. Context Assembly] -> [8. LLM Answer Generation]
```

### Detailed Workflow

#### 📝 Document Insertion

1. **Document Chunking**
   - Split document into manageable segments using intelligent tokenization
   - Apply configurable token limits with overlap between chunks
   - Preserve temporal metadata (`date` field) in each chunk

2. **LLM Information Extraction**
   - Process each chunk through LLM using specialized prompts
   - Extract structured entities (name, type, description) and relationships (source, target, keywords, description)
   - Support 8 entity types: COMPANY, PERSON, PRODUCT, FINANCIAL_METRIC, BUSINESS_CONCEPT, MARKET, TECHNOLOGY, GEOGRAPHIC

3. **Temporal Graph Building**
   - Add extracted entities and relationships to unified NetworkX graph structure
   - Every entity and relationship node includes temporal metadata from source document
   - Generate embeddings for semantic similarity (using SentenceTransformer or custom function)
   - Store original chunk content for later context retrieval

4. **Temporal Evolution Links**
   - **Required step**: Must be called after all document insertions
   - Build "temporal_evolution" edges between same entities across different time periods
   - Links enable tracking entity evolution over time within single graph structure

#### 🔍 Query Processing

5. **Keywords Extraction**
   - Use LLM to extract high-level keywords (concepts, themes) and low-level keywords (specific entities, terms) from user question
   - Return structured keyword dictionary for graph retrieval

6. **Time-Aware Graph Retrieval**
   - **Stage 1**: Semantic search using extracted keywords against entity/relation embeddings
   - **Stage 2**: Multi-hop expansion with configurable depth (max_hops parameter)  
   - **Stage 3**: Time-aware filtering (if time_range specified) with temporal relevance scoring
   - **Stage 4**: Temporal evolution expansion (automatically include temporal_evolution connections)
   - **Stage 5**: Relation-entity expansion (automatically include connected nodes from retrieved relationships)
   - Support multiple temporal expansion modes: strict, with_temporal, expanded

7. **Context Assembly** 
   - **Primary Chunks**: Retrieve original text chunks from entities/relationships found in graph
   - **Supplementary Chunks**: Add time-relevant chunks (if time filtering enabled) to provide broader context
   - Apply deduplication to avoid sending same chunk content multiple times to LLM
   - Smart truncation based on token limits while preserving most relevant information

8. **Answer Generation**
   - Format retrieved context (entities + relationships + chunks) into structured prompt
   - Add time range context if filtering was applied
   - Generate comprehensive answer using LLM with proper citations and evidence

### Unified Temporal-Aware Graph Structure

gtrag creates a single knowledge graph where every entity and relationship includes temporal information:

```
┌─────────────────── UNIFIED KNOWLEDGE GRAPH ──────────────────────┐
│                                                                   │
│  Apple[2023-12] ◄──produces──► iPhone[2023-12]                   │
│        │                          │                              │
│        │ temporal_evolution        │ temporal_evolution           │
│        ▼                          ▼                              │
│  Apple[2024-03] ◄──produces──► iPhone[2024-03]                   │
│        │                          │                              │
│        │ temporal_evolution        │ temporal_evolution           │
│        ▼                          ▼                              │
│  Apple[2024-06] ◄──produces──► iPhone[2024-06]                   │
│                                                                   │
│                                                                   │
│  Microsoft[2023-12] ◄──develops──► Azure[2023-12]                │
│           │                           │                          │
│           │ temporal_evolution         │ temporal_evolution        │
│           ▼                           ▼                          │
│  Microsoft[2024-03] ◄──develops──► Azure[2024-03]                │
└───────────────────────────────────────────────────────────────────┘

Key Features:
• Every entity node has temporal metadata: Entity[time_period]
• Every relationship edge has temporal metadata  
• temporal_evolution edges connect same entities across time periods
• Semantic relationships (produces, develops) exist within and across time periods
• Single graph structure with time-aware nodes and edges
```

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

For basic usage, see the [Quick Start](#-quick-start) section above.

### 🎛️ **Method 2: Custom Configuration**

```python
from gtrag import gtragSystem, QueryParams

# Custom LLM function (optional)
def my_llm_function(system_prompt: str, user_prompt: str):
    # Your custom LLM implementation
    # Return JSON string for extraction, plain text for final answers
    pass

def my_embedding_function(text: str):
    # Your custom embedding implementation
    # Return list of floats
    pass

# Initialize with custom models
rag = gtragSystem(
    llm_func=my_llm_function,
    embedding_func=my_embedding_function
)

# Index multiple documents efficiently
documents = [
    {"text": "...", "doc_id": "doc1", "metadata": {"date": "2023-12"}},
    {"text": "...", "doc_id": "doc2", "metadata": {"date": "2024-03"}},
    {"text": "...", "doc_id": "doc3", "metadata": {"date": "2024-06"}},
]

for doc in documents:
    rag.insert(doc["text"], doc["doc_id"], doc["metadata"])

rag.build_temporal_links()

# Query with custom parameters and time filtering
custom_params = QueryParams(
    top_k=15,                    # Retrieve top 15 results
    similarity_threshold=0.2,    # Lower threshold = more results
    max_hops=3,                  # Maximum graph traversal hops
    final_max_tokens=8000,       # Context size limit
    time_range=["2024-03", "2024-06"],  # Specific time period
    enable_time_filtering=True   # Enable time-aware filtering
)

result = rag.query(
    "Compare performance trends in early 2024",
    query_params=custom_params
)
```

### ⏰ **Time Range Filtering Examples**

```python
from gtrag.config.settings import QueryParams

# Example 1: Single month filtering
params = QueryParams(time_range=["2023-12"], enable_time_filtering=True)
result = rag.query("What happened in December 2023?", query_params=params)

# Example 2: Multiple months
params = QueryParams(
    time_range=["2023-12", "2024-01", "2024-03"], 
    enable_time_filtering=True
)
result = rag.query("Show trends across different periods", query_params=params)

# Example 3: Temporal expansion modes
params = QueryParams(
    time_range=["2024-03"],
    enable_time_filtering=True,
    temporal_expansion_mode="expanded"  # Include adjacent periods
)
result = rag.query("March 2024 performance with context", query_params=params)
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
# Save complete gtrag system to working directory
rag.save_graph("./my_gtrag_project/")

# Load in a new session
new_rag = gtragSystem()
new_rag.load_graph("./my_gtrag_project/")

# Continue querying with loaded data
result = new_rag.query("Previous analysis question")

# Directory structure created:
# my_gtrag_project/
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
from gtrag import gtragSystem, QueryParams, ChunkingConfig

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
rag = gtragSystem(
    chunking_config=chunk_config,
    query_params=query_params
)
```

## ⚠️ **Important Usage Notes**

### 1. **Temporal Metadata is Essential**
```python
# ✅ Correct: Include time information in 'date' field (YYYY-MM format)
rag.insert(text, doc_id, metadata={"date": "2024-03"})        # March 2024
rag.insert(text, doc_id, metadata={"date": "2023-12"})        # December 2023
rag.insert(text, doc_id, metadata={"date": "2024-06"})        # June 2024

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

### 3. **Standardized Time Format**
```python
# ✅ Required YYYY-MM format (unified 'date' field)
metadata = {"date": "2024-03"}       # March 2024
metadata = {"date": "2023-12"}       # December 2023 
metadata = {"date": "2024-01"}       # January 2024
metadata = {"date": "2024-06"}       # June 2024

# 📝 Standard format: YYYY-MM
# Consistent time representation eliminates conversion complexity!
```

## 🎬 Run the Demos

```bash
cd examples

# Basic functionality demo
python demo_openai,py
```

The demos will show you:
- **demo_openai,py**: Basic functionality with standardized YYYY-MM `date` field
- System initialization and document indexing
- Temporal connection building with standardized YYYY-MM format
- Multiple query examples with time_range filtering
- Detailed result analysis

## 📁 Project Structure

```
gtrag/
├── gtrag/                    # Main package
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
    ├── demo_openai,py            # Complete demonstration
    └── demo_gemini,py 
```

## 🤝 Contributing

This is a research project focused on temporal-aware RAG systems. The core functionality is complete and ready for research and prototyping use cases.

## 📄 License

This project is available for research and educational purposes.
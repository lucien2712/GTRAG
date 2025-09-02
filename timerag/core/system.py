"""
TimeRAG Core System

Main system orchestrator that provides a unified API for document indexing and querying.
Integrates all TimeRAG components into a cohesive system.
"""

import logging
import os
import json
from typing import List, Dict, Any, Callable, Optional
from pathlib import Path

# Optional import for dotenv
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    def load_dotenv():
        pass

from ..config.settings import QueryParams, ChunkingConfig
from ..extractors.llm_extractor import LLMExtractor
from ..graph.builder import GraphBuilder
from ..graph.retriever import GraphRetriever
from ..processing.chunker import DocumentChunker
from ..processing.token_manager import TokenManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeRAGSystem:
    """
    Main TimeRAG system that integrates all components for document indexing and querying.
    
    This class provides a simple API for:
    1. Document insertion with temporal metadata
    2. Building temporal connections between entities
    3. Querying the knowledge graph with intelligent retrieval
    """
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 llm_func: Optional[Callable[[str, str], str]] = None,
                 embedding_func: Optional[Callable[[str], List[float]]] = None,
                 query_params: Optional[QueryParams] = None,
                 chunking_config: Optional[ChunkingConfig] = None):
        """
        Initialize the TimeRAG system with all core components.
        
        Args:
            openai_api_key: OpenAI API key (optional, can use env variable)
            llm_func: Custom LLM function (optional, uses OpenAI if not provided)
            embedding_func: Custom embedding function (optional, uses SentenceTransformer if not provided)
            query_params: Query parameters for retrieval
            chunking_config: Configuration for document chunking
        """
        load_dotenv()
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        # Initialize core components
        self.extractor = LLMExtractor(api_key=api_key, llm_func=llm_func)
        self.graph_builder = GraphBuilder(embedding_func=embedding_func)
        self.retriever = GraphRetriever(self.graph_builder, self.extractor)
        self.chunker = DocumentChunker(config=chunking_config or ChunkingConfig())
        self.token_manager = TokenManager(query_params=query_params or QueryParams())
        
        # Store original chunk content for retrieval context
        self.chunk_store: Dict[str, str] = {}
        
        self.processing_stats = {"indexed_documents": 0, "indexed_chunks": 0}
        logger.info("TimeRAG system initialized successfully.")

    def insert(self, text: str, doc_id: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Index a document by chunking, extracting entities/relations, and building the graph.
        
        Args:
            text: Document content to index
            doc_id: Unique document identifier
            metadata: Document metadata (include 'timestamp', 'quarter', 'date', or 'time' for temporal functionality)
        """
        logger.info(f"Starting document insertion: {doc_id}")
        metadata = metadata or {}
        
        # Standardize time information using TimeHandler
        from ..utils.time_handler import TimeHandler
        time_point = TimeHandler.extract_time_from_metadata(metadata)
        if time_point:
            # Add standardized timestamp while keeping original fields for backward compatibility
            metadata['_standardized_time'] = time_point.value
            metadata['_time_granularity'] = time_point.granularity.value
            logger.info(f"Standardized time for {doc_id}: {time_point.value} ({time_point.granularity.value})")

        # 1. Document chunking
        chunks = self.chunker.chunk(doc_id, text, metadata)
        logger.info(f"Document {doc_id} split into {len(chunks)} chunks.")
        self.processing_stats["indexed_chunks"] += len(chunks)

        # 2. Store original chunk content and extract information
        all_entities = []
        all_relations = []
        for chunk in chunks:
            # Store chunk content for later retrieval context
            self.chunk_store[chunk.chunk_id] = chunk.content
            
            entities, relations = self.extractor.extract(chunk.content, doc_id, chunk.metadata)
            all_entities.extend(entities)
            all_relations.extend(relations)
            logger.info(f"Chunk {chunk.chunk_id} extracted {len(entities)} entities, {len(relations)} relations.")

        # 3. Add extracted entities and relations to the graph
        self.graph_builder.add_entities(all_entities)
        self.graph_builder.add_relations(all_relations)
        self.processing_stats["indexed_documents"] += 1

    def build_temporal_links(self):
        """
        Build temporal connections between entities across different time periods.
        This must be called after all documents are inserted.
        """
        logger.info("Building temporal connections...")
        self.graph_builder.build_temporal_connections()
        logger.info("Temporal connections built successfully.")

    def query(self, question: str, query_params: Optional[QueryParams] = None) -> Dict[str, Any]:
        """
        Query the knowledge graph and generate an answer using retrieved context.
        
        Args:
            question: User's question
            query_params: Optional query parameters to override defaults
            
        Returns:
            Dictionary containing answer, retrieved context, and metadata
        """
        logger.info(f"Processing query: {question}")
        params = query_params or self.token_manager.limits

        # Validate time range if provided
        if params.time_range:
            from ..utils.time_range import TimeRangeParser
            is_valid, error_msg = TimeRangeParser.validate_time_range(params.time_range)
            if not is_valid:
                logger.error(f"Invalid time range: {error_msg}")
                return {
                    "answer": f"Error: {error_msg}",
                    "retrieved_entities": [],
                    "retrieved_relations": [],
                    "retrieved_source_chunks": [],
                    "token_stats": {"total_tokens": 0},
                    "query_keywords": {}
                }
            # Auto-enable time filtering if time_range is provided
            if params.time_range and not params.enable_time_filtering:
                params.enable_time_filtering = True
                logger.info("Auto-enabled time filtering due to time_range specification")

        # 1. Keywords extraction  
        query_keywords = self.retriever.extract_keywords(question)
        logger.info(f"Keywords extracted: high_level={len(query_keywords.get('high_level_keywords', []))}, low_level={len(query_keywords.get('low_level_keywords', []))}")

        # 2. Graph retrieval with enhanced time filtering support
        retrieved_entities, retrieved_relations = self.retriever.search(
            keywords=query_keywords,
            top_k=params.top_k,
            similarity_threshold=params.similarity_threshold,
            time_range=params.time_range,
            enable_time_filtering=params.enable_time_filtering,
            temporal_expansion_mode=params.temporal_expansion_mode,
            temporal_evolution_scope=params.temporal_evolution_scope,
            semantic_weight=params.semantic_weight,
            temporal_weight=params.temporal_weight
        )
        logger.info(f"Retrieved {len(retrieved_entities)} entities, {len(retrieved_relations)} relations.")

        # 3. Two-stage chunk retrieval strategy
        # Stage 1: Get chunks from retrieved entities/relations
        primary_chunk_ids = set()
        for item in retrieved_entities + retrieved_relations:
            if 'chunk_id' in item.get('metadata', {}):
                primary_chunk_ids.add(item['metadata']['chunk_id'])
        
        primary_chunks = [self.chunk_store[cid] for cid in primary_chunk_ids if cid in self.chunk_store]
        
        # Stage 2: Get additional chunks from time range (excluding already retrieved ones)
        supplementary_chunks = []
        if params.enable_time_filtering and params.time_range:
            from ..utils.time_range import TimeRangeParser
            valid_quarters = TimeRangeParser.parse_time_range(params.time_range)
            if valid_quarters:
                # Collect time-relevant chunk IDs that are NOT already in primary chunks
                time_relevant_chunk_ids = set()
                for chunk_id in self.chunk_store.keys():
                    if chunk_id not in primary_chunk_ids:  # Avoid duplicates
                        chunk_time = self._extract_time_from_chunk_id(chunk_id)
                        if chunk_time in valid_quarters:
                            time_relevant_chunk_ids.add(chunk_id)
                
                # Add supplementary chunks (already deduplicated)
                supplementary_chunks = [self.chunk_store[cid] for cid in time_relevant_chunk_ids]
                # Limit to avoid overwhelming context
                supplementary_chunks = supplementary_chunks[:min(3, len(primary_chunks))]
        
        # Combine chunks (no duplicates since we used sets for IDs)
        source_chunks = primary_chunks + supplementary_chunks
        logger.info(f"Found {len(primary_chunks)} primary + {len(supplementary_chunks)} supplementary = {len(source_chunks)} total unique chunks.")

        # 4. Prepare context data (entities + relations + chunks)
        context_data = self._format_context_data(retrieved_entities, retrieved_relations, source_chunks)
        
        # 5. Generate answer using RAG response prompt
        system_prompt = "You are a helpful assistant that provides accurate answers based on knowledge graph entities, relationships, and document chunks."
        
        # Add time range context if filtering was applied
        time_context = ""
        if params.enable_time_filtering and params.time_range:
            from ..utils.time_range import TimeRangeParser
            expanded_quarters = TimeRangeParser.expand_time_range(params.time_range)
            time_context = f"\n---Time Range---\nAnalysis limited to time period: {', '.join(expanded_quarters)}\n"
        
        rag_prompt = f"""---Context Data---
{context_data}{time_context}

---User Query---
{question}

Generate a comprehensive response based on the provided entities, relationships, and document chunks. Structure your response with:

### Key Findings
Summarize the main information from entities and relationships

### Supporting Evidence  
Include relevant details from the document chunks

### References
List the most relevant sources used
"""
        
        final_prompt, token_stats = self.token_manager.prepare_final_context(
            retrieved_entities=retrieved_entities,
            retrieved_relations=retrieved_relations,
            source_chunks=source_chunks,
            query=question,
            prompt_template=rag_prompt
        )
        logger.info(f"Final context prepared, total tokens: {token_stats['total_tokens']}")

        # Generate answer
        answer = self.extractor.llm_call(system_prompt, final_prompt)
        logger.info("Answer generation completed.")

        return {
            "answer": answer,
            "retrieved_entities": retrieved_entities,
            "retrieved_relations": retrieved_relations,
            "retrieved_source_chunks": source_chunks,
            "token_stats": token_stats,
            "query_keywords": query_keywords
        }

    def _format_context_data(self, entities: List[Dict], relations: List[Dict], chunks: List[str]) -> str:
        """Format context data for the RAG response prompt."""
        context_parts = []
        
        if entities:
            context_parts.append("### Entities")
            for entity in entities:
                name = entity.get('name', 'Unknown')
                entity_type = entity.get('type', 'Unknown') 
                description = entity.get('description', 'No description')
                context_parts.append(f"- **{name}** ({entity_type}): {description}")
        
        if relations:
            context_parts.append("\n### Relations")
            for relation in relations:
                source = relation.get('source', 'Unknown')
                target = relation.get('target', 'Unknown')
                relation_type = relation.get('type', 'related to')
                description = relation.get('description', 'No description')
                context_parts.append(f"- **{source}** {relation_type} **{target}**: {description}")
        
        if chunks:
            context_parts.append("\n### Document Chunks")
            for i, chunk in enumerate(chunks, 1):
                context_parts.append(f"**Chunk {i}:** {chunk[:500]}..." if len(chunk) > 500 else f"**Chunk {i}:** {chunk}")
        
        return "\n".join(context_parts)

    def _extract_time_from_chunk_id(self, chunk_id: str) -> Optional[str]:
        """
        Extract time information from chunk ID using flexible TimeHandler.
        Supports various time formats in chunk IDs.
        """
        from ..utils.time_handler import TimeHandler
        import re
        
        # Try to find any recognizable time pattern in chunk ID
        # Common patterns: doc_2024Q1_chunk1, report_2024-03_part1, etc.
        time_patterns = [
            r'(\d{4}Q[1-4])',      # Quarter format
            r'(\d{4}-\d{2}-\d{2})', # ISO date
            r'(\d{4}-\d{2})',      # Year-month
            r'(\d{4})',            # Year only
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, chunk_id)
            if match:
                time_candidate = match.group(1)
                parsed_time = TimeHandler.parse_time(time_candidate)
                if parsed_time:
                    return parsed_time.value
        
        return None

    def save_graph(self, working_dir: str):
        """
        Save the complete TimeRAG system to a working directory.
        
        Args:
            working_dir: Directory path where all TimeRAG files will be saved
            
        Files created:
            - graph.json: NetworkX knowledge graph with entities/relations
            - chunks.json: Original text chunks for context retrieval
            - vectors.faiss: FAISS vector index (if vector store enabled)
            - vectors.metadata.npy: Vector metadata (if vector store enabled)
        """
        # Create working directory if it doesn't exist
        work_path = Path(working_dir)
        work_path.mkdir(parents=True, exist_ok=True)
        
        # Define file paths
        graph_path = work_path / "graph.json"
        chunks_path = work_path / "chunks.json"
        vectors_path = work_path / "vectors"
        
        # Save graph
        self.graph_builder.save(str(graph_path))
        logger.info(f"Knowledge graph saved to: {graph_path}")

        # Save chunk store
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunk_store, f, ensure_ascii=False, indent=2)
        logger.info(f"Chunk store saved to: {chunks_path}")
        
        # Save vector store if it exists
        if hasattr(self.graph_builder, 'vector_store') and self.graph_builder.vector_store:
            try:
                self.graph_builder.vector_store.save(str(vectors_path))
                logger.info(f"Vector store saved to: {vectors_path}.*")
            except Exception as e:
                logger.warning(f"Failed to save vector store: {e}")
        
        logger.info(f"Complete TimeRAG system saved to working directory: {working_dir}")
        return {
            "working_dir": str(work_path),
            "graph_file": str(graph_path),
            "chunks_file": str(chunks_path),
            "vectors_file": str(vectors_path) + ".*" if hasattr(self.graph_builder, 'vector_store') and self.graph_builder.vector_store else None
        }

    def load_graph(self, working_dir: str):
        """
        Load complete TimeRAG system from a working directory.
        
        Args:
            working_dir: Directory path containing TimeRAG files
            
        Expected files:
            - graph.json: Knowledge graph (required)
            - chunks.json: Text chunks (optional, but recommended)
            - vectors.faiss + vectors.metadata.npy: Vector store (optional)
        """
        work_path = Path(working_dir)
        if not work_path.exists():
            raise FileNotFoundError(f"Working directory not found: {working_dir}")
        
        # Define file paths
        graph_path = work_path / "graph.json"
        chunks_path = work_path / "chunks.json"
        vectors_path = work_path / "vectors"
        
        # Load graph (required)
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_path}")
        
        self.graph_builder.load(str(graph_path))
        self.retriever = GraphRetriever(self.graph_builder, self.extractor)
        logger.info(f"Knowledge graph loaded from: {graph_path}")

        # Load chunk store (optional)
        if chunks_path.exists():
            with open(chunks_path, 'r', encoding='utf-8') as f:
                self.chunk_store = json.load(f)
            logger.info(f"Chunk store loaded from: {chunks_path}")
        else:
            logger.warning(f"Chunk store file not found: {chunks_path}. Some functionality may be limited.")
            self.chunk_store = {}
        
        # Load vector store (optional)
        if hasattr(self.graph_builder, 'vector_store') and self.graph_builder.vector_store:
            try:
                # Check if vector files exist
                faiss_file = vectors_path.with_suffix('.faiss')
                metadata_file = vectors_path.with_suffix('.metadata.npy')
                
                if faiss_file.exists() and metadata_file.exists():
                    self.graph_builder.vector_store.load(str(vectors_path))
                    logger.info(f"Vector store loaded from: {vectors_path}.*")
                else:
                    logger.info("Vector store files not found, will rebuild if needed")
            except Exception as e:
                logger.warning(f"Failed to load vector store: {e}")
        
        logger.info(f"Complete TimeRAG system loaded from working directory: {working_dir}")
        return {
            "working_dir": str(work_path),
            "loaded_graph": graph_path.exists(),
            "loaded_chunks": chunks_path.exists(), 
            "loaded_vectors": (vectors_path.with_suffix('.faiss').exists() and 
                             vectors_path.with_suffix('.metadata.npy').exists())
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get system processing statistics."""
        graph_stats = self.graph_builder.get_graph_stats()
        return {
            **self.processing_stats,
            **graph_stats,
            "stored_chunks": len(self.chunk_store)
        }
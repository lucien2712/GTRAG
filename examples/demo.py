#!/usr/bin/env python3
"""
TimeRAG API Usage Demo

This example demonstrates how to initialize the system, index documents, and make queries
using the new refactored timerag package structure.

This demo shows the complete workflow:
1. System initialization with custom LLM/embedding functions
2. Document insertion with temporal metadata
3. Building temporal connections
4. Querying with entity/relation/chunk integration
"""
import os
import sys
from pathlib import Path

# Add project root to Python path to find timerag module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from timerag import TimeRAGSystem, QueryParams

# Optional dotenv loading
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not available. Please set environment variables manually.")

# --- Custom Model Functions ---
# In real applications, you can put these functions in separate files

def gpt_4o_mini_llm(system_prompt: str, user_prompt: str) -> str:
    """Custom LLM function using OpenAI GPT-4o-mini model."""
    try:
        import openai
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except ImportError:
        print("OpenAI library not available. Please install: pip install openai")
        return "{}"
    except Exception as e:
        print(f"Error calling GPT-4o-mini: {e}")
        return "{}"

def openai_embedding_func(text: str) -> list:
    """Custom embedding function using OpenAI text-embedding-3-small model."""
    try:
        import openai
        client = openai.OpenAI()
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except ImportError:
        print("OpenAI library not available. Please install: pip install openai")
        return []
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []


def main():
    """Main execution function"""
    print("=" * 60)
    print("TimeRAG API Usage Demo")
    print("=" * 60)
    
    # Check if API key exists
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
        print("Please set your API key in .env file or environment variable")
        print("Demo will continue but may fail without valid API key")
        print()
    
    # 1. Initialize System
    print("1. 🚀 Initializing TimeRAG System...")
    try:
        rag = TimeRAGSystem(
            llm_func=gpt_4o_mini_llm,
            embedding_func=openai_embedding_func
        )
        print("   ✅ System initialization completed successfully")
    except Exception as e:
        print(f"   ❌ System initialization failed: {e}")
        return
    
    # 2. Index Documents with Temporal Metadata
    print("\n2. 📄 Indexing documents with temporal metadata...")
    documents = [
        {
            "text": "Apple Inc. reported iPhone sales of 80 million units in Q4 2023, showing strong demand for the latest models.", 
            "doc_id": "apple_q4_2023", 
            "metadata": {"quarter": "2023Q4"}
        },
        {
            "text": "By Q1 2024, Apple's iPhone sales increased to 90 million units due to new model releases and improved supply chain.", 
            "doc_id": "apple_q1_2024", 
            "metadata": {"quarter": "2024Q1"}
        },
        {
            "text": "Microsoft's cloud business revenue grew significantly by 30% in Q1 2024, driven by Azure services and enterprise adoption.", 
            "doc_id": "ms_q1_2024", 
            "metadata": {"quarter": "2024Q1"}
        },
        {
            "text": "In Q2 2024, Microsoft continued strong cloud performance with 35% year-over-year growth, expanding into new markets.",
            "doc_id": "ms_q2_2024",
            "metadata": {"quarter": "2024Q2"}
        },
        {
            "text": "Apple's services revenue reached $22.3 billion in Q1 2024, representing 16% growth from the previous year.",
            "doc_id": "apple_services_q1_2024",
            "metadata": {"quarter": "2024Q1"}
        }
    ]
    
    for i, doc in enumerate(documents, 1):
        try:
            rag.insert(doc["text"], doc["doc_id"], doc["metadata"])
            print(f"   ✅ Indexed [{i}/{len(documents)}]: {doc['doc_id']}")
        except Exception as e:
            print(f"   ❌ Failed to index {doc['doc_id']}: {e}")
    
    # 3. Build Temporal Links
    print("\n3. 🔗 Building temporal connections...")
    try:
        rag.build_temporal_links()
        print("   ✅ Temporal connections built successfully")
    except Exception as e:
        print(f"   ❌ Failed to build temporal connections: {e}")
        return
    
    # Get system statistics
    try:
        stats = rag.get_stats()
        print(f"\n📊 System Statistics:")
        print(f"   - Documents indexed: {stats.get('indexed_documents', 0)}")
        print(f"   - Graph nodes: {stats.get('num_nodes', 0)}")
        print(f"   - Graph edges: {stats.get('num_edges', 0)}")
        print(f"   - Stored chunks: {stats.get('stored_chunks', 0)}")
    except Exception as e:
        print(f"   ❌ Failed to get stats: {e}")
    
    # 4. Query Examples
    queries = [
        "What are the trends in Apple iPhone sales over time?",
        "How is Microsoft's cloud business performing across quarters?",
        "Compare Apple and Microsoft's performance in 2024 Q1.",
        "What is Apple's services revenue trend?"
    ]
    
    for i, question in enumerate(queries, 1):
        print(f"\n{3+i}. 🔍 Query {i}: {question}")
        print("-" * 80)
        
        # Define query parameters
        custom_query_params = QueryParams(
            top_k=8,
            similarity_threshold=0.2,
            max_hops=2,
            final_max_tokens=8000
        )
        
        try:
            result = rag.query(question, query_params=custom_query_params)
            
            # Display answer
            answer = result.get('answer', 'No answer generated')
            print(f"📝 Answer:\n{answer}")
            
            # Show retrieved context information
            entities = result.get('retrieved_entities', [])
            relations = result.get('retrieved_relations', [])
            chunks = result.get('retrieved_source_chunks', [])
            
            print(f"\n📈 Retrieved Context:")
            print(f"   - Entities: {len(entities)}")
            if entities:
                print("     Top entities:")
                for entity in entities[:3]:
                    name = entity.get('name', 'Unknown')
                    entity_type = entity.get('type', 'Unknown')
                    score = entity.get('score', 0)
                    print(f"       • {name} ({entity_type}) - Score: {score:.3f}")
            
            print(f"   - Relations: {len(relations)}")
            if relations:
                print("     Top relations:")
                for relation in relations[:3]:
                    source = relation.get('source', 'Unknown')
                    target = relation.get('target', 'Unknown')
                    rel_type = relation.get('type', 'Unknown')
                    score = relation.get('score', 0)
                    print(f"       • {source} → {target} ({rel_type}) - Score: {score:.3f}")
            
            print(f"   - Source chunks: {len(chunks)}")
            
            # Token usage statistics
            token_stats = result.get('token_stats', {})
            print(f"   - Total tokens: {token_stats.get('total_tokens', 0)}")
            print(f"   - Entity tokens: {token_stats.get('entities_tokens', 0)}")
            print(f"   - Relation tokens: {token_stats.get('relations_tokens', 0)}")
            print(f"   - Chunk tokens: {token_stats.get('chunks_tokens', 0)}")
            
        except Exception as e:
            print(f"   ❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎉 Demo Completed Successfully!")
    print("=" * 60)
    
    # Show final system statistics
    try:
        final_stats = rag.get_stats()
        print(f"\n📊 Final System State:")
        for key, value in final_stats.items():
            print(f"   - {key}: {value}")
    except Exception as e:
        print(f"Failed to get final stats: {e}")


def demo_batch_processing():
    """Demonstrate batch processing capabilities"""
    print("\n" + "=" * 60)
    print("Batch Processing Demo")
    print("=" * 60)
    
    print("📁 Batch processing allows you to process multiple documents at once.")
    print("   Example usage:")
    print("""
    from timerag.processing import BatchProcessor, BatchProcessingConfig
    
    # Configure batch processing
    config = BatchProcessingConfig(
        max_workers=4,
        batch_size=10,
        supported_formats=['.pdf', '.docx', '.txt', '.json']
    )
    
    # Initialize batch processor
    batch_processor = BatchProcessor(rag_system, config)
    
    # Process all documents in a directory
    results = batch_processor.process_directory("./documents/")
    
    # Process specific files
    file_list = ["doc1.pdf", "doc2.docx", "doc3.txt"]
    results = batch_processor.process_files(file_list)
    """)
    print("\n   📝 This would process multiple files concurrently with:")
    print("      • Automatic file format detection")
    print("      • Quarter extraction from filenames/content")
    print("      • Progress tracking and error handling")
    print("      • Parallel processing for better performance")


if __name__ == "__main__":
    try:
        main()
        demo_batch_processing()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
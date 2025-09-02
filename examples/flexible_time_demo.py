#!/usr/bin/env python3
"""
gtrag Flexible Time Format Demonstration

This demo showcases gtrag's new flexible time format support,
demonstrating various time formats beyond just quarters.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to Python path so we can import gtrag
sys.path.insert(0, str(Path(__file__).parent.parent))

from gtrag.core.system import gtragSystem
from gtrag.config.settings import QueryParams

def main():
    print("=== gtrag Flexible Time Format Demo ===\n")
    
    # Initialize gtrag system
    print("1. Initializing gtrag system...")
    rag = gtragSystem()
    
    print("2. Indexing documents with various time formats...\n")
    
    # === Various Time Format Examples ===
    
    # Quarter format (unified 'date' field)
    rag.insert(
        text="Apple reported record iPhone 15 sales of 85M units in Q4 2023, driving strong holiday revenues.",
        doc_id="apple_q4_2023",
        metadata={"date": "2023Q4", "company": "Apple"}
    )
    
    # ISO Date format
    rag.insert(
        text="Apple launched iPhone 15 Pro with titanium design on 2023-09-15, marking a significant upgrade.",
        doc_id="apple_launch_2023",
        metadata={"date": "2023-09-15", "company": "Apple", "event": "product_launch"}
    )
    
    # Year-Month format
    rag.insert(
        text="Microsoft Azure cloud revenue grew 28% year-over-year in March 2024, exceeding analyst expectations.",
        doc_id="msft_march_2024",
        metadata={"date": "2024-03", "company": "Microsoft"}
    )
    
    # Month name format
    rag.insert(
        text="Google's Pixel 8 series gained 12% market share in November 2023 following the launch event.",
        doc_id="google_nov_2023",
        metadata={"date": "November 2023", "company": "Google"}
    )
    
    # Year only format
    rag.insert(
        text="Tesla delivered over 1.8 million vehicles globally in 2023, setting a new annual record.",
        doc_id="tesla_2023_annual",
        metadata={"date": "2023", "company": "Tesla"}
    )
    
    # Custom label format
    rag.insert(
        text="During Phase-Alpha of the Mars mission, SpaceX successfully tested the new Raptor engine design.",
        doc_id="spacex_phase_alpha",
        metadata={"date": "Phase-Alpha", "company": "SpaceX", "project": "Mars Mission"}
    )
    
    # Another custom format
    rag.insert(
        text="In Sprint-3 of the AI development cycle, OpenAI achieved breakthrough performance on reasoning tasks.",
        doc_id="openai_sprint3",
        metadata={"date": "Sprint-3", "company": "OpenAI", "project": "AI Development"}
    )
    
    print("3. Building temporal connections...")
    rag.build_temporal_links()
    
    print("4. Querying with different time ranges...\n")
    
    # === Query Examples with Different Time Formats ===
    
    queries = [
        {
            "question": "What are the key technology developments in 2023?",
            "time_range": ["2023"],
            "description": "Year-based time filtering"
        },
        {
            "question": "How did tech companies perform in Q4 2023?",
            "time_range": ["2023Q4"],
            "description": "Quarter-based time filtering (backward compatible)"
        },
        {
            "question": "What product launches happened in September 2023?",
            "time_range": ["2023-09"],
            "description": "Month-based time filtering"
        },
        {
            "question": "What developments occurred in March 2024?",
            "time_range": ["March 2024"],
            "description": "Month name format filtering"
        },
        {
            "question": "What happened during custom project phases?",
            "time_range": ["Phase-Alpha", "Sprint-3"],
            "description": "Custom label time filtering"
        },
        {
            "question": "Show me all Apple-related activities across different time periods",
            "time_range": ["2023Q4", "2023-09-15"],
            "description": "Mixed time format filtering"
        }
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"=== Query {i}: {query['description']} ===")
        print(f"Question: {query['question']}")
        print(f"Time Range: {query['time_range']}")
        
        # Query with time filtering
        params = QueryParams(
            top_k=10,
            similarity_threshold=0.2,
            time_range=query['time_range'],
            enable_time_filtering=True,
            temporal_expansion_mode="with_temporal"
        )
        
        result = rag.query(query['question'], query_params=params)
        
        print(f"\n📋 Answer:")
        print(result['answer'])
        
        print(f"\n📊 Retrieved Context:")
        print(f"- Entities: {len(result['retrieved_entities'])}")
        print(f"- Relations: {len(result['retrieved_relations'])}")
        print(f"- Source Chunks: {len(result['retrieved_source_chunks'])}")
        
        if result['retrieved_entities']:
            print(f"\n🔍 Top Entities:")
            for entity in result['retrieved_entities'][:3]:
                entity_time = entity.get('metadata', {}).get('_standardized_time', 'N/A')
                print(f"  - {entity['name']} ({entity['type']}) [Time: {entity_time}]")
        
        print(f"\n⏰ Time Range Used: {query['time_range']}")
        print("-" * 80)
    
    print("\n5. Advanced Time Range Demonstrations...")
    
    # Advanced example: Compare different temporal expansion modes
    print("\n=== Comparing Temporal Expansion Modes ===")
    base_question = "What technologies emerged in 2023?"
    
    for mode in ["strict", "with_temporal", "expanded"]:
        print(f"\n--- Mode: {mode} ---")
        params = QueryParams(
            time_range=["2023Q4"],
            enable_time_filtering=True,
            temporal_expansion_mode=mode,
            top_k=5
        )
        
        result = rag.query(base_question, query_params=params)
        print(f"Retrieved {len(result['retrieved_entities'])} entities")
        print(f"Mode '{mode}' summary: {result['answer'][:200]}...")
    
    print("\n6. System Statistics...")
    
    # Show system stats
    stats = rag.get_stats()
    print(f"\n📈 System Statistics:")
    print(f"- Indexed documents: {stats['indexed_documents']}")
    print(f"- Graph nodes: {stats['num_nodes']}")
    print(f"- Graph edges: {stats['num_edges']}")
    print(f"- Stored chunks: {stats['stored_chunks']}")
    
    print("\n7. Saving system with flexible time data...")
    rag.save_graph("./flexible_time_demo/")
    print("System saved to: ./flexible_time_demo/")
    
    print("\n=== Demo Complete ===")
    print("\nKey Features Demonstrated:")
    print("✅ Unified 'date' field supporting all time formats: quarters, ISO dates, months, years, custom labels")
    print("✅ Flexible time_range query filtering with QueryParams")
    print("✅ Mixed time format support in single queries")
    print("✅ Temporal expansion modes: strict, with_temporal, expanded")
    print("✅ Temporal evolution connections across different time formats")
    print("✅ Smart time similarity scoring")
    print("✅ Simplified metadata structure with single 'date' field")


if __name__ == "__main__":
    main()
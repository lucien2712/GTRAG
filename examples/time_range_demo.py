"""
gtrag Time Range Query Demo

This demo shows how to use the time range filtering functionality
to query specific time periods in temporal data.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from gtrag.core.system import gtragSystem
from gtrag.config.settings import QueryParams


def main():
    # Initialize gtrag system
    rag = gtragSystem()
    
    # Sample temporal documents
    sample_documents = [
        {
            "doc_id": "earnings_2024q1",
            "text": """Apple Inc. Q1 2024 Earnings Report
            
            Revenue for Q1 2024 reached $119.6 billion, up 2% year-over-year.
            iPhone revenue was $69.7 billion. Services revenue grew to $23.1 billion.
            CEO Tim Cook expressed optimism about AI integration and new products.
            The company announced a $0.24 dividend per share.
            """,
            "metadata": {"date": "2024Q1", "company": "Apple", "report_type": "earnings"}
        },
        {
            "doc_id": "earnings_2024q2", 
            "text": """Apple Inc. Q2 2024 Earnings Report
            
            Revenue for Q2 2024 was $90.8 billion, down 4% from previous year.
            iPhone sales declined to $51.0 billion due to market saturation.
            Services revenue continued growing to $24.2 billion.
            Tim Cook announced major AI partnerships and Vision Pro expansion.
            Dividend remained at $0.24 per share.
            """,
            "metadata": {"date": "2024Q2", "company": "Apple", "report_type": "earnings"}
        },
        {
            "doc_id": "earnings_2024q3",
            "text": """Apple Inc. Q3 2024 Earnings Report
            
            Q3 2024 revenue reached $94.9 billion, showing 5% growth.
            iPhone revenue rebounded to $39.3 billion with new model launches.
            Services hit record $24.2 billion driven by App Store growth.
            Tim Cook highlighted successful AI feature rollouts and market expansion.
            Special dividend of $0.25 per share announced.
            """,
            "metadata": {"date": "2024Q3", "company": "Apple", "report_type": "earnings"}
        }
    ]
    
    print("=== gtrag Time Range Demo ===\n")
    
    # Insert sample documents
    print("1. Inserting sample documents...")
    for doc in sample_documents:
        rag.insert(doc["text"], doc["doc_id"], doc["metadata"])
        print(f"   - Inserted {doc['doc_id']}")
    
    # Build temporal connections
    print("\n2. Building temporal connections...")
    rag.build_temporal_links()
    print("   - Temporal connections built successfully")
    
    # Test cases for time range queries
    test_cases = [
        {
            "name": "Query All Time Periods (No Filter)",
            "question": "What was Apple's revenue performance?",
            "query_params": QueryParams()
        },
        {
            "name": "Query Q1 2024 Only",
            "question": "What was Apple's revenue performance?",
            "query_params": QueryParams(
                time_range=["2024Q1"],
                enable_time_filtering=True
            )
        },
        {
            "name": "Query Q2-Q3 2024 Range",
            "question": "How did iPhone sales perform?", 
            "query_params": QueryParams(
                time_range=["2024Q2", "2024Q3"],
                enable_time_filtering=True
            )
        },
        {
            "name": "Query with Year Specification",
            "question": "What did Tim Cook say about AI?",
            "query_params": QueryParams(
                time_range=["2024"],
                enable_time_filtering=True,
                include_temporal_evolution=False  # Focus only on 2024 data
            )
        },
        {
            "name": "Query Single Quarter (Q3)",
            "question": "What was the dividend announcement?",
            "query_params": QueryParams(
                time_range=["2024Q3"],
                enable_time_filtering=True
            )
        }
    ]
    
    # Run test cases
    print("\n3. Running time range query tests...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"--- Test {i}: {test_case['name']} ---")
        print(f"Question: {test_case['question']}")
        
        if test_case['query_params'].time_range:
            print(f"Time Range: {test_case['query_params'].time_range}")
            print(f"Time Filtering: {test_case['query_params'].enable_time_filtering}")
            print(f"Include Temporal Evolution: {test_case['query_params'].include_temporal_evolution}")
        else:
            print("Time Range: Not specified (query all periods)")
        
        try:
            result = rag.query(test_case['question'], test_case['query_params'])
            
            print(f"Retrieved Entities: {len(result['retrieved_entities'])}")
            print(f"Retrieved Relations: {len(result['retrieved_relations'])}")
            print(f"Retrieved Chunks: {len(result['retrieved_source_chunks'])}")
            
            # Show relevant quarters found in results
            found_quarters = set()
            for entity in result['retrieved_entities']:
                quarter = entity.get('metadata', {}).get('quarter')
                if quarter:
                    found_quarters.add(quarter)
            
            if found_quarters:
                print(f"Data from quarters: {sorted(found_quarters)}")
            
            print(f"Answer: {result['answer'][:200]}...")
            
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print("\n" + "="*60 + "\n")
    
    # Demonstrate time range validation
    print("4. Testing time range validation...\n")
    
    invalid_cases = [
        ["2024Q5"],  # Invalid quarter
        ["2024Q0"],  # Invalid quarter  
        ["24Q1"],    # Invalid year format
        ["Q1"],      # Missing year
        [2024]       # Wrong type
    ]
    
    for invalid_range in invalid_cases:
        try:
            params = QueryParams(
                time_range=invalid_range,
                enable_time_filtering=True
            )
            result = rag.query("Test query", params)
            if "Error:" in result['answer']:
                print(f"✓ Correctly rejected invalid range: {invalid_range}")
                print(f"  Error: {result['answer']}")
            else:
                print(f"✗ Failed to reject invalid range: {invalid_range}")
        except Exception as e:
            print(f"✓ Correctly rejected invalid range: {invalid_range}")
            print(f"  Exception: {str(e)}")
        print()
    
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()
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
    
    # Sample temporal documents with various time formats
    sample_documents = [
        {
            "doc_id": "earnings_2024q1",
            "text": """Apple Inc. Q1 2024 Earnings Report
            
            Revenue for Q1 2024 reached $119.6 billion, up 2% year-over-year.
            iPhone revenue was $69.7 billion. Services revenue grew to $23.1 billion.
            CEO Tim Cook expressed optimism about AI integration and new products.
            The company announced a $0.24 dividend per share.
            """,
            "metadata": {"date": "2024Q1", "company": "Apple", "report_type": "earnings"}  # Quarter format
        },
        {
            "doc_id": "product_launch_sep_2024", 
            "text": """Apple iPhone 16 Launch Event - September 9, 2024
            
            Apple unveiled the iPhone 16 series with breakthrough AI capabilities.
            New camera control features and enhanced computational photography.
            Pre-orders begin September 13, with availability September 20, 2024.
            Starting price remains competitive at $799 for base model.
            """,
            "metadata": {"date": "2024-09-09", "company": "Apple", "report_type": "launch"}  # ISO date format
        },
        {
            "doc_id": "microsoft_march_2024",
            "text": """Microsoft Cloud Services Update - March 2024
            
            Azure cloud services showed remarkable 35% growth in March 2024.
            New AI-powered developer tools launched for enterprise customers.
            Microsoft Teams integration with Copilot drives productivity gains.
            Strong enterprise adoption across Fortune 500 companies.
            """,
            "metadata": {"date": "2024-03", "company": "Microsoft", "report_type": "update"}  # Year-month format
        },
        {
            "doc_id": "google_annual_2024",
            "text": """Google Annual Innovation Summary - 2024
            
            Google achieved breakthrough advances in AI research throughout 2024.
            Quantum computing milestones reached with new processor designs.
            Search algorithm improvements enhance user experience globally.
            Sustainability initiatives show measurable environmental impact.
            """,
            "metadata": {"date": "2024", "company": "Google", "report_type": "annual"}  # Year only format
        },
        {
            "doc_id": "tesla_phase_beta",
            "text": """Tesla Full Self-Driving Beta Phase Update
            
            During Phase-Beta testing, Tesla's FSD showed significant improvements.
            Neural network updates provide better object recognition capabilities.
            Safety metrics improved by 40% compared to previous phase testing.
            Regulatory approval processes advancing in multiple regions.
            """,
            "metadata": {"date": "Phase-Beta", "company": "Tesla", "report_type": "development"}  # Custom label format
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
    
    # Test cases for time range queries with diverse formats
    test_cases = [
        {
            "name": "Query All Time Periods (No Filter)",
            "question": "What technology developments occurred across companies?",
            "query_params": QueryParams()
        },
        {
            "name": "Query Q1 2024 Only (Quarter Format)",
            "question": "What was Apple's Q1 2024 performance?",
            "query_params": QueryParams(
                time_range=["2024Q1"],
                enable_time_filtering=True
            )
        },
        {
            "name": "Query Specific Date (ISO Format)",
            "question": "What happened during the iPhone 16 launch?", 
            "query_params": QueryParams(
                time_range=["2024-09-09"],
                enable_time_filtering=True
            )
        },
        {
            "name": "Query Month Period (Year-Month Format)",
            "question": "What Microsoft developments occurred in March 2024?",
            "query_params": QueryParams(
                time_range=["2024-03"],
                enable_time_filtering=True
            )
        },
        {
            "name": "Query Year Range (Year Only Format)",
            "question": "What innovations did Google achieve in 2024?",
            "query_params": QueryParams(
                time_range=["2024"],
                enable_time_filtering=True,
                include_temporal_evolution=False  # Focus only on 2024 data
            )
        },
        {
            "name": "Query Custom Phase (Custom Label Format)",
            "question": "What progress was made during Phase-Beta testing?",
            "query_params": QueryParams(
                time_range=["Phase-Beta"],
                enable_time_filtering=True
            )
        },
        {
            "name": "Query Mixed Time Formats",
            "question": "Compare Q1 earnings to September launch and March updates",
            "query_params": QueryParams(
                time_range=["2024Q1", "2024-09-09", "2024-03"],
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
            
            # Show relevant time periods found in results
            found_time_periods = set()
            for entity in result['retrieved_entities']:
                time_period = entity.get('metadata', {}).get('_standardized_time')
                if time_period:
                    found_time_periods.add(time_period)
            
            if found_time_periods:
                print(f"Data from time periods: {sorted(found_time_periods)}")
            
            print(f"Answer: {result['answer'][:200]}...")
            
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print("\n" + "="*60 + "\n")
    
    # Demonstrate time range validation
    print("4. Testing time range validation...\n")
    
    invalid_cases = [
        ["2024Q5"],        # Invalid quarter
        ["2024Q0"],        # Invalid quarter  
        ["24Q1"],          # Invalid year format
        ["Q1"],            # Missing year
        ["2024-13-01"],    # Invalid month in ISO date
        ["2024-02-30"],    # Invalid day in ISO date
        ["13-2024"],       # Invalid month format
        [2024],            # Wrong type (should be string)
        ["invalid-date"]   # Unrecognizable format
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
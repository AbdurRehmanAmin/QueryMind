#!/usr/bin/env python3
"""
Test script for the case bank implementation.
"""

import os
import sys
from pprint import pprint

# Add the development directory to the path
sys.path.append("development")

# Import the CaseBank
from MLAgentBench.case_bank import CaseBank, retrieve_similar_cases, generate_case_prompt

def test_case_bank():
    """Test the case bank by retrieving similar cases for a sample query"""
    # Create a case bank instance pointing to the data directory
    case_bank = CaseBank(os.path.join("development", "data"))
    
    # Print the number of cases loaded
    print(f"Loaded {len(case_bank.cases)} cases")
    
    # Print the first few cases to verify they loaded correctly
    print("\nSample cases:")
    for i, case in enumerate(case_bank.cases[:3]):
        print(f"Case {i+1}: {case.task_name}")
        print(f"  Category: {case.metadata.get('category', 'None')}")
        print(f"  Description: {case.task_description[:100]}...")
        print()
    
    # Test retrieving similar cases using a sample query
    print("\nTesting case retrieval:")
    
    # Test with a tabular query
    tabular_query = "I need to predict house prices based on features like location, size, and number of rooms."
    print(f"Query: {tabular_query}")
    tabular_cases = case_bank.retrieve_similar_cases(tabular_query, top_k=2)
    print(f"Found {len(tabular_cases)} similar tabular cases")
    for i, (case, similarity) in enumerate(tabular_cases):
        print(f"  Match {i+1}: {case.task_name} (similarity: {similarity:.3f})")
        print(f"    Category: {case.metadata.get('category', 'None')}")
        print(f"    Description: {case.task_description[:100]}...")
    
    # Test with an NLP query
    nlp_query = "I need to evaluate student summaries and determine their quality."
    print(f"\nQuery: {nlp_query}")
    nlp_cases = case_bank.retrieve_similar_cases(nlp_query, top_k=2)
    print(f"Found {len(nlp_cases)} similar NLP cases")
    for i, (case, similarity) in enumerate(nlp_cases):
        print(f"  Match {i+1}: {case.task_name} (similarity: {similarity:.3f})")
        print(f"    Category: {case.metadata.get('category', 'None')}")
        print(f"    Description: {case.task_description[:100]}...")
    
    # Test generating a prompt
    print("\nTesting case prompt generation:")
    prompt = case_bank.generate_case_prompt(tabular_query, top_k=2)
    print(f"Generated prompt (first 200 chars): {prompt[:200]}...")
    
    # Test the helper functions
    print("\nTesting helper functions:")
    helper_cases = retrieve_similar_cases(tabular_query, top_k=2)
    print(f"Retrieved {len(helper_cases)} cases using helper function")

if __name__ == "__main__":
    test_case_bank() 
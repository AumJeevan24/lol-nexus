import pytest
import json
import os
import glob
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from rag_core.chain import get_response
from rag_core.test_config import groq_judge

# Metrics Setup
faithfulness = FaithfulnessMetric(
    threshold=0.7, 
    model=groq_judge, 
    include_reason=True
)
relevancy = AnswerRelevancyMetric(
    threshold=0.7, 
    model=groq_judge, 
    include_reason=True
)

def load_latest_synthetic_data():
    """Finds the most recent JSON file in rag_core"""
    list_of_files = glob.glob('rag_core/*.json') 
    
    if not list_of_files:
        print("Warning: No JSON dataset found in rag_core/")
        return []
        
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Loading test data from: {latest_file}")
    
    try:
        with open(latest_file, "r") as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return []

# Load the auto-generated questions dynamically
synthetic_cases = load_latest_synthetic_data()

@pytest.mark.parametrize("case", synthetic_cases)
def test_rag_performance(case):
    # DeepEval JSON structure
    question = case.get("input")
    expected_answer = case.get("expected_output")
    
    # FIX: Handle Context Data Types Correctly
    # The JSON might have "context" as a string OR a list.
    # We must ensure we pass a List[str] to LLMTestCase.
    raw_context = case.get("context", [])
    
    if isinstance(raw_context, str):
        # If it's a string, wrap it in a list
        final_retrieval_context = [raw_context]
    elif isinstance(raw_context, list):
        # If it's already a list, use it directly (don't double-wrap)
        final_retrieval_context = raw_context
    else:
        final_retrieval_context = []

    print(f"\nTesting Synthetic Question: {question}")
    
    # 1. Run RAG
    actual_response = get_response(question)
    answer_text = actual_response["answer"]
    
    # 2. Create Test Case
    test_case = LLMTestCase(
        input=question,
        actual_output=answer_text,
        expected_output=expected_answer, 
        retrieval_context=final_retrieval_context # <--- The fixed variable
    )

    # 3. Assert
    assert_test(test_case, [faithfulness, relevancy])
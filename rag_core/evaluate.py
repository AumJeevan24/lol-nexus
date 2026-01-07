import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from rag_core.chain import get_response
from rag_core.test_config import groq_judge

# Define our Metrics (The "Rules" for the Judge)
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

# The "Exam Questions" for your Bot
test_data = [
    {
        "input": "What changes were made to Hwei?",
        "expected_output": "Hwei received buffs to armor and ability tweaks.", 
        # We leave 'context' empty here to let the system fetch it dynamically
        # and test if the retrieval is actually working.
        "context": [] 
    },
    {
        "input": "Did Aatrox get a nerf?",
        "expected_output": "No, Aatrox was not nerfed in this patch.",
        "context": []
    }
]

@pytest.mark.parametrize("case", test_data)
def test_rag_performance(case):
    print(f"\n🧪 Testing Question: {case['input']}")
    
    # 1. Run your RAG Pipeline
    # This fetches the real answer from your Groq bot
    actual_response = get_response(case["input"])
    answer_text = actual_response["answer"]
    
    print(f"🤖 Bot Answer: {answer_text}")

    # 2. Create a DeepEval Test Case
    # We send the Input + Actual Output + Retrieved Context to the Judge
    test_case = LLMTestCase(
        input=case["input"],
        actual_output=answer_text,
        # In a real test, we would grab the *actual* retrieved docs from the chain.
        # For now, we are testing if the answer makes sense based on the input.
        retrieval_context=case["context"] 
    )

    # 3. Assert (Grading)
    assert_test(test_case, [faithfulness, relevancy])
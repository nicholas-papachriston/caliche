"""
Evaluation Framework Example

Demonstrates how to evaluate agent outputs using different evaluators
and load evaluation datasets.
"""

import asyncio
from caliche import (
    AgentBuilder,
    ExactMatchEvaluator,
    ContainsEvaluator,
    LLMEvaluator,
    EvaluationDataset,
)


async def main():
    print("Evaluation Framework Example\n")

    # Create an agent to evaluate
    agent = await AgentBuilder().set_provider("openai").set_model("gpt-4").build()

    # Test cases
    test_cases = [
        {"input": "What is 2+2?", "expected": "4"},
        {"input": "What is the capital of France?", "expected": "Paris"},
        {"input": "Write a hello world program in Python", "expected": "print"},
    ]

    # 1. Exact Match Evaluator
    print("1. Exact Match Evaluator")
    exact_evaluator = ExactMatchEvaluator(case_sensitive=True)

    for test_case in test_cases[:2]:
        response = await agent.chat(test_case["input"])
        result = exact_evaluator.evaluate(response[0], test_case["expected"])
        print(f"  Input: {test_case['input']}")
        print(f"  Expected: {test_case['expected']}")
        print(f"  Output: {response[0][:50]}...")
        print(f"  Score: {result['score']}, Passed: {result['passed']}\n")

    # 2. Contains Evaluator
    print("2. Contains Evaluator")
    contains_evaluator = ContainsEvaluator(case_sensitive=False)

    coding_response = await agent.chat(test_cases[2]["input"])
    contains_result = contains_evaluator.evaluate(
        coding_response[0], test_cases[2]["expected"]
    )
    print(f"  Input: {test_cases[2]['input']}")
    print(f"  Expected to contain: {test_cases[2]['expected']}")
    print(f"  Score: {contains_result['score']}, Passed: {contains_result['passed']}\n")

    # 3. LLM Evaluator
    print("3. LLM Evaluator")
    llm_evaluator = LLMEvaluator(
        "Evaluate if the response is helpful, accurate, and well-formatted.",
        provider="openai",
        model="gpt-4",
    )

    llm_response = await agent.chat("Explain quantum computing simply")
    llm_result = llm_evaluator.evaluate(
        llm_response[0],
        "A clear, simple explanation of quantum computing",
        context={"context": "Educational content for beginners"},
    )
    print("  Input: Explain quantum computing simply")
    print(f"  Score: {llm_result['score']}, Passed: {llm_result['passed']}")
    if "reasoning" in llm_result.get("details", {}):
        print(f"  Reasoning: {llm_result['details']['reasoning']}\n")

    # 4. Load dataset from file (example)
    print("4. Dataset Loading")
    try:
        # This would load from a JSONL file if it exists
        # dataset = EvaluationDataset.from_jsonl("./test_dataset.jsonl")
        # print(f"  Loaded dataset: {dataset['name']}")
        # print(f"  Test cases: {len(dataset['test_cases'])}")
        print("  (Dataset loading requires a valid JSONL file)")
    except Exception as e:
        print(f"  Dataset file not found (this is expected in this example): {e}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    asyncio.run(main())


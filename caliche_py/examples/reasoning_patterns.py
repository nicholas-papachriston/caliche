"""
Caliche Reasoning Patterns Examples

Demonstrates advanced reasoning patterns including ReAct, Plan-Execute, and Reflexion.

uv run examples/reasoning_example.py
"""

import os
from caliche import AgentBuilder, ProviderType
from caliche.reasoning import ReActPattern, PlanExecutePattern, ReflexionPattern


def example_react():
    """ReAct (Reasoning + Acting) pattern example."""
    print("=== ReAct (Reasoning + Acting) Pattern ===\n")

    agent = AgentBuilder() \
        .provider(ProviderType.OpenAI) \
        .model("gpt-4o-mini") \
        .api_key(os.getenv("OPENAI_API_KEY", "")) \
        .enable_tool("calculator") \
        .build()

    react = ReActPattern(agent, max_iterations=5)  # Max 5 iterations

    result = react.execute("What is 15 * 23? Show your work.")

    print(f"Final Answer: {result['final_answer']}")
    print("\nReasoning Steps:")
    for i, step in enumerate(result['steps']):
        print(f"\nStep {i + 1} [{step['step_type']}]:")
        print(step['content'])
        if step.get('metadata'):
            print(f"Metadata: {step['metadata']}")

    if result.get('token_usage'):
        print(f"\nToken Usage: {result['token_usage']}")
    print()


def example_plan_execute():
    """Plan-Execute pattern example."""
    print("=== Plan-Execute Pattern ===\n")

    agent = AgentBuilder() \
        .provider(ProviderType.OpenAI) \
        .model("gpt-4o-mini") \
        .api_key(os.getenv("OPENAI_API_KEY", "")) \
        .enable_tool("calculator") \
        .build()

    planner = PlanExecutePattern(agent, max_iterations=10)  # Max 10 steps

    result = planner.execute(
        "Create a step-by-step plan for learning Rust programming, then execute the first step."
    )

    print(f"Final Answer: {result['final_answer']}")
    print("\nExecution Steps:")
    for i, step in enumerate(result['steps']):
        print(f"\nStep {i + 1} [{step['step_type']}]:")
        print(step['content'])
    print()


def example_reflexion():
    """Reflexion pattern example."""
    print("=== Reflexion Pattern ===\n")

    agent = AgentBuilder() \
        .provider(ProviderType.OpenAI) \
        .model("gpt-4o-mini") \
        .api_key(os.getenv("OPENAI_API_KEY", "")) \
        .build()

    reflexion = ReflexionPattern(agent, max_iterations=3)  # Max 3 iterations

    result = reflexion.execute(
        "Write a haiku about artificial intelligence, then critique and improve it."
    )

    print(f"Final Answer: {result['final_answer']}")
    print("\nReflection Steps:")
    for i, step in enumerate(result['steps']):
        print(f"\nStep {i + 1} [{step['step_type']}]:")
        print(step['content'])
    print()


def main():
    try:
        example_react()
        example_plan_execute()
        example_reflexion()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


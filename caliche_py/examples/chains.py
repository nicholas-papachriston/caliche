"""
Caliche Chains Examples

Demonstrates how to use LLM chains, sequential chains, and map-reduce chains
for composable LLM workflows.

uv run examples/chains_example.py
"""

import os
from caliche import AgentBuilder, ProviderType
from caliche.chains import LLMChain, SequentialChain, MapReduceChain


def example_llm_chain():
    """Basic LLM chain example."""
    print("=== LLM Chain Example ===\n")

    # Create an agent
    agent = AgentBuilder() \
        .provider(ProviderType.OpenAI) \
        .model("gpt-4o-mini") \
        .api_key(os.getenv("OPENAI_API_KEY", "")) \
        .build()

    # Create a chain with a prompt template
    chain = LLMChain(agent, "Summarize the following text in one sentence: {text}")

    # Run the chain with input variables
    result = chain.run({
        "text": "Artificial intelligence is transforming industries across the globe. "
                "From healthcare to finance, AI technologies are enabling new capabilities "
                "and improving efficiency. Machine learning algorithms can process vast amounts "
                "of data to identify patterns and make predictions that were previously impossible."
    })

    print(f"Input: {result['output']['text']}")
    print(f"Summary: {result['output']['output']}")
    print()


def example_sequential_chain():
    """Sequential chain example."""
    print("=== Sequential Chain Example ===\n")

    agent = AgentBuilder() \
        .provider(ProviderType.OpenAI) \
        .model("gpt-4o-mini") \
        .api_key(os.getenv("OPENAI_API_KEY", "")) \
        .build()

    # Create multiple chains that will execute in sequence
    analyze_chain = LLMChain(agent, "Analyze the following topic and identify key points: {input}")
    summarize_chain = LLMChain(agent, "Summarize the analysis: {output}")
    format_chain = LLMChain(agent, "Format as a bulleted list: {output}")

    # Create a sequential chain
    sequential_chain = SequentialChain() \
        .add_chain("analyze", analyze_chain) \
        .add_chain("summarize", summarize_chain) \
        .add_chain("format", format_chain)

    # Run the sequential chain
    result = sequential_chain.run({
        "input": "The impact of renewable energy on global economies"
    })

    print(f"Final output: {result['output']['output']}")
    print()


def example_map_reduce_chain():
    """Map-reduce chain example."""
    print("=== Map-Reduce Chain Example ===\n")

    # Create agents for map and reduce operations
    map_agent = AgentBuilder() \
        .provider(ProviderType.OpenAI) \
        .model("gpt-4o-mini") \
        .api_key(os.getenv("OPENAI_API_KEY", "")) \
        .build()

    reduce_agent = AgentBuilder() \
        .provider(ProviderType.OpenAI) \
        .model("gpt-4o-mini") \
        .api_key(os.getenv("OPENAI_API_KEY", "")) \
        .build()

    # Create map and reduce chains
    map_chain = LLMChain(map_agent, "Extract the main theme from this article: {text}")
    reduce_chain = LLMChain(reduce_agent, "Combine these themes into a unified summary: {items}")

    # Create map-reduce chain
    map_reduce_chain = MapReduceChain(map_chain, reduce_chain)

    # Process multiple articles in parallel, then combine results
    articles = [
        {"text": "Article 1: Quantum computing promises to revolutionize cryptography and drug discovery..."},
        {"text": "Article 2: Machine learning models are becoming more efficient and accessible..."},
        {"text": "Article 3: Cloud computing infrastructure enables scalable AI deployments..."}
    ]

    result = map_reduce_chain.run(articles)

    print(f"Combined summary: {result['output']['output']}")
    print()


def main():
    try:
        example_llm_chain()
        example_sequential_chain()
        example_map_reduce_chain()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


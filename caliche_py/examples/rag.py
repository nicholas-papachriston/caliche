"""
RAG Service Example

Demonstrates how to use RagService for programmatic document indexing and retrieval.
This example shows:
- Creating a RagService with an agent
- Indexing multiple documents
- Basic querying with similarity threshold
- Displaying results with relevance scores

uv run examples/rag_service_example.py
"""

import os
from caliche import AgentBuilder, ProviderType, RagService


def main():
    # Create an agent with embedding provider
    agent = (
        AgentBuilder()
        .provider(ProviderType.OpenAI)
        .model("gpt-4o")
        .api_key(os.getenv("OPENAI_API_KEY", ""))
        .build()
    )

    # Create RAG service
    print("Creating RAG service...\n")
    rag = RagService(agent)

    # Index multiple documents
    documents = [
        {
            "id": "rust_intro",
            "content": (
                "Rust is a systems programming language focused on memory safety "
                "without garbage collection. It uses ownership and borrowing to prevent "
                "data races and memory errors. Rust's type system ensures memory safety "
                "at compile time."
            ),
        },
        {
            "id": "python_intro",
            "content": (
                "Python is a high-level programming language known for simplicity and "
                "readability. It uses dynamic typing and automatic memory management. "
                "Python's extensive standard library makes it suitable for many applications."
            ),
        },
        {
            "id": "javascript_intro",
            "content": (
                "JavaScript is a dynamic programming language primarily used for web "
                "development. It runs in browsers and on servers via Node.js. JavaScript "
                "supports both object-oriented and functional programming paradigms."
            ),
        },
    ]

    print("Indexing documents...\n")
    for doc in documents:
        result = rag.index_document(doc["content"], doc["id"])
        print(
            f'Indexed "{doc["id"]}": {result.chunk_count} chunks, '
            f"{result.usage.get('embedding_tokens', 0) if result.usage else 0} embedding tokens"
        )

    # Get statistics
    stats = rag.stats()
    print(f"\nStatistics: {stats.embedding_count} embeddings, {stats.text_count} text chunks\n")

    # Query for relevant information
    print("Querying for relevant information...\n")
    query = "What are Rust's memory safety features?"
    results = rag.query(query, 5, 0.3)

    print(f'Query: "{query}"\n')
    print(f"Found {len(results.results)} relevant chunks:")
    top_sim_str = f"{results.top_similarity:.3f}" if results.top_similarity else "N/A"
    print(f"Top similarity: {top_sim_str}\n")

    for i, result in enumerate(results.results, 1):
        # Extract similarity score from result string format: "[Similarity: 0.xxx] text..."
        if "[Similarity:" in result:
            parts = result.split("] ", 1)
            if len(parts) == 2:
                score_part = parts[0].replace("[Similarity: ", "")
                text = parts[1]
                print(f"{i}. [Score: {score_part}] {text[:100]}...")
            else:
                print(f"{i}. {result[:100]}...")
        else:
            print(f"{i}. {result[:100]}...")

    # Enhanced query with hybrid search (if available)
    print("\n\nEnhanced query with hybrid search...\n")
    enhanced_results = rag.enhanced_query(
        query="programming language safety",
        top_k=3,
        min_similarity=0.3,
        search_mode="dense",
        rerank=False,  # Set to True if rag-reranking feature is enabled
    )

    print(f"Found {len(enhanced_results.results)} results:")
    if enhanced_results.warnings:
        print(f"Warnings: {', '.join(enhanced_results.warnings)}")

    for i, result in enumerate(enhanced_results.results, 1):
        print(
            f"\n{i}. [Score: {result['score']:.3f}] "
            f"{result.get('text', result.get('id', ''))[:150]}..."
        )

    print("\nExample completed!")


if __name__ == "__main__":
    main()


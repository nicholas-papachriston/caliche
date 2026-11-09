/**
 * RAG Service Example
 *
 * Demonstrates how to use RagService for programmatic document indexing and retrieval.
 * This example shows:
 * - Creating a RagService with an agent
 * - Indexing multiple documents
 * - Basic querying with similarity threshold
 * - Displaying results with relevance scores
 *
 * bun run examples/rag-service.ts
 */

import { AgentBuilder, RagService } from "../src/index";

async function main() {
  // Create an agent with embedding provider
  const agent = await new AgentBuilder()
    .provider("openai")
    .model("gpt-4o")
    .apiKey(process.env.OPENAI_API_KEY || "")
    .build();

  // Create RAG service
  console.log("Creating RAG service...\n");
  const rag = await RagService.new(agent);

  // Index multiple documents
  const documents = [
    {
      id: "rust_intro",
      content:
        "Rust is a systems programming language focused on memory safety without garbage collection. It uses ownership and borrowing to prevent data races and memory errors. Rust's type system ensures memory safety at compile time.",
    },
    {
      id: "python_intro",
      content:
        "Python is a high-level programming language known for simplicity and readability. It uses dynamic typing and automatic memory management. Python's extensive standard library makes it suitable for many applications.",
    },
    {
      id: "javascript_intro",
      content:
        "JavaScript is a dynamic programming language primarily used for web development. It runs in browsers and on servers via Node.js. JavaScript supports both object-oriented and functional programming paradigms.",
    },
  ];

  console.log("Indexing documents...\n");
  for (const doc of documents) {
    const result = await rag.indexDocument(doc.content, doc.id);
    console.log(
      `Indexed "${doc.id}": ${result.chunk_count} chunks, ${result.usage?.embedding_tokens || 0} embedding tokens`
    );
  }

  // Get statistics
  const stats = await rag.stats();
  console.log(
    `\nStatistics: ${stats.embedding_count} embeddings, ${stats.text_count} text chunks\n`
  );

  // Query for relevant information
  console.log("Querying for relevant information...\n");
  const query = "What are Rust's memory safety features?";
  const results = await rag.query(query, 5, 0.3);

  console.log(`Query: "${query}"\n`);
  console.log(`Found ${results.results.length} relevant chunks:`);
  console.log(`Top similarity: ${results.top_similarity?.toFixed(3) || "N/A"}\n`);

  for (let i = 0; i < results.results.length; i++) {
    const result = results.results[i];
    // Extract similarity score from result string format: "[Similarity: 0.xxx] text..."
    const match = result?.match(/\[Similarity: ([\d.]+)\]\s*(.+)/);
    if (match) {
      const score = parseFloat(match[1] || "0");
      const text = match[2];
      console.log(`${i + 1}. [Score: ${score.toFixed(3)}] ${text?.substring(0, 100) || "..."}...`);
    } else {
      console.log(`${i + 1}. ${result?.substring(0, 100) || "..."}...`);
    }
  }

  // Enhanced query with hybrid search (if available)
  console.log("\n\nEnhanced query with hybrid search...\n");
  const enhancedResults = await rag.enhancedQuery({
    query: "programming language safety",
    top_k: 3,
    min_similarity: 0.3,
    search_mode: "dense",
    rerank: false, // Set to true if rag-reranking feature is enabled
  });

  console.log(`Found ${enhancedResults.results.length} results:`);
  if (enhancedResults.warnings.length > 0) {
    console.log(`Warnings: ${enhancedResults.warnings.join(", ")}`);
  }

  for (let i = 0; i < enhancedResults.results.length; i++) {
    const result = enhancedResults.results[i];
    console.log(
      `\n${i + 1}. [Score: ${result?.score.toFixed(3) || "0"}] ${result?.text?.substring(0, 150) || result?.id || "..."}...`
    );
  }

  console.log("\nExample completed!");
}

main().catch((error) => {
  console.error("Error:", error);
  process.exit(1);
});

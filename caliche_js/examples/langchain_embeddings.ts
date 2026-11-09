/**
 * LangChain Embeddings Integration Example
 *
 * This example demonstrates how to use LangChain embedding providers with Caliche
 * for vector memory and RAG features.
 *
 * Prerequisites:
 * - Install dependencies: bun install
 * - Set OPENAI_API_KEY environment variable
 *
 * Usage:
 *   bun run examples/langchain_embeddings.ts
 */

import { OpenAIEmbeddings } from "@langchain/openai";
import { LangChainEmbeddingProvider } from "../src/adapters/langchain-embeddings";
import { AgentBuilder, type ChatResponse } from "../src/index";

async function exampleVectorMemory(): Promise<void> {
  console.log("=== Example 1: Vector Memory with LangChain Embeddings ===\n");

  if (!process.env.OPENAI_API_KEY) {
    console.log("Skipping: OPENAI_API_KEY not set");
    return;
  }

  // Create LangChain embeddings
  const embeddings = new OpenAIEmbeddings({
    modelName: "text-embedding-3-small",
  });

  // Wrap it with Caliche adapter
  const embeddingProvider = new LangChainEmbeddingProvider(embeddings);

  // Create agent with LangChain embeddings for vector memory
  const agent = await new AgentBuilder()
    .provider("openai") // Use OpenAI for LLM
    .model("gpt-4o-mini")
    .langchainEmbeddings(embeddingProvider)
    .vectorMemory({
      max_tokens: 8192,
      max_memories: 1000,
      top_k: 5,
      similarity_threshold: 0.7,
      buffer_tokens: 4096,
    })
    .build();

  console.log("Agent created with LangChain embeddings for vector memory");
  console.log(`Provider: ${agent.provider}, Model: ${agent.model}\n`);

  // Add some memories
  const response1: ChatResponse = await agent.chat("My favorite programming language is Rust.");
  console.log(`Response 1: ${response1.response.substring(0, 100)}...\n`);

  const response2: ChatResponse = await agent.chat("What is my favorite programming language?");
  console.log(`Response 2: ${response2.response}\n`);

  // The agent should remember from the conversation history
  const response3: ChatResponse = await agent.chat("Why do I like Rust?");
  console.log(`Response 3: ${response3.response.substring(0, 200)}...\n`);
}

async function exampleRagWithEmbeddings(): Promise<void> {
  console.log("=== Example 2: RAG with LangChain Embeddings ===\n");

  if (!process.env.OPENAI_API_KEY) {
    console.log("Skipping: OPENAI_API_KEY not set");
    return;
  }

  // Create LangChain embeddings
  const embeddings = new OpenAIEmbeddings({
    modelName: "text-embedding-3-small",
  });

  // Wrap it with Caliche adapter
  const embeddingProvider = new LangChainEmbeddingProvider(embeddings);

  // Create agent with LangChain embeddings for RAG
  const agent = await new AgentBuilder()
    .provider("openai")
    .model("gpt-4o-mini")
    .langchainEmbeddings(embeddingProvider)
    .enableTool("semantic_search") // Enable RAG tool which uses embeddings
    .build();

  console.log("Agent created with LangChain embeddings for RAG");
  console.log(`Provider: ${agent.provider}, Model: ${agent.model}\n`);

  // Index some documents (this would typically be done via RAG service)
  // For this example, we'll just demonstrate the setup
  console.log("Agent is ready to use RAG with LangChain embeddings");
  console.log("You can now index documents and query them using the RAG tool\n");
}

async function main(): Promise<void> {
  console.log("=".repeat(60));
  console.log("LangChain Embeddings Integration Examples");
  console.log("=".repeat(60));
  console.log();

  if (!process.env.OPENAI_API_KEY) {
    console.error("Error: OPENAI_API_KEY environment variable is not set");
    console.log("\nPlease set your OpenAI API key:");
    console.log("  export OPENAI_API_KEY='sk-...'");
    process.exit(1);
  }

  try {
    await exampleVectorMemory();
    await exampleRagWithEmbeddings();

    console.log("=".repeat(60));
    console.log("All examples completed successfully!");
    console.log("=".repeat(60));
  } catch (error) {
    console.error("\nError running examples:", error);
    process.exit(1);
  }
}

main().catch(console.error);

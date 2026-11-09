/**
 * LangChain Provider Integration Example
 *
 * This example demonstrates how to use LangChain LLM providers with Caliche.
 * This allows you to leverage LangChain's extensive provider ecosystem while
 * benefiting from Caliche's tool execution, memory strategies, and RAG capabilities.
 *
 * Prerequisites:
 * - Install dependencies: bun install
 * - Set OPENAI_API_KEY environment variable (or other provider credentials)
 *
 * Usage:
 *   bun run examples/langchain_provider.ts
 */

import { ChatAnthropic } from "@langchain/anthropic";
import { ChatOpenAI } from "@langchain/openai";
import { LangChainLLMProvider } from "../src/adapters/langchain-llm";
import { AgentBuilder, type ChatResponse } from "../src/index";

async function exampleOpenAIProvider(): Promise<void> {
  console.log("=== Example 1: OpenAI Provider via LangChain ===\n");

  if (!process.env.OPENAI_API_KEY) {
    console.log("Skipping: OPENAI_API_KEY not set");
    return;
  }

  // Create LangChain LLM
  const llm = new ChatOpenAI({
    model: "gpt-4o-mini",
    temperature: 0.7,
  });

  // Wrap it with Caliche adapter
  const provider = new LangChainLLMProvider(llm, true, true);

  // Create agent with LangChain provider
  const agent = await new AgentBuilder()
    .langchainProvider(provider)
    .enableTool("calculator")
    .enableTool("fs")
    .build();

  // Use the agent
  console.log("Agent created with LangChain OpenAI provider");
  console.log(`Provider: ${agent.provider}, Model: ${agent.model}\n`);

  const response: ChatResponse = await agent.chat(
    "What is 15 * 23? Use the calculator tool if needed."
  );
  console.log(`Response: ${response.response}`);
  console.log(`Token usage: ${response.usage ? JSON.stringify(response.usage, null, 2) : "N/A"}\n`);
}

async function exampleAnthropicProvider(): Promise<void> {
  console.log("=== Example 2: Anthropic Provider via LangChain ===\n");

  if (!process.env.ANTHROPIC_API_KEY) {
    console.log("Skipping: ANTHROPIC_API_KEY not set");
    return;
  }

  // Create LangChain LLM
  const llm = new ChatAnthropic({
    model: "claude-haiku-4-5",
    temperature: 0.7,
  });

  // Wrap it with Caliche adapter
  const provider = new LangChainLLMProvider(llm, true, true);

  // Create agent with LangChain provider
  const agent = await new AgentBuilder()
    .langchainProvider(provider)
    .systemPrompt("You are a helpful assistant that explains concepts clearly.")
    .enableTool("calculator")
    .build();

  console.log("Agent created with LangChain Anthropic provider");
  console.log(`Provider: ${agent.provider}, Model: ${agent.model}\n`);

  const response: ChatResponse = await agent.chat("Explain quantum computing in simple terms.");
  console.log(`Response: ${response.response.substring(0, 200)}...`);
  console.log(`Token usage: ${response.usage ? JSON.stringify(response.usage, null, 2) : "N/A"}\n`);
}

async function exampleStreaming(): Promise<void> {
  console.log("=== Example 3: Streaming with LangChain Provider ===\n");

  if (!process.env.OPENAI_API_KEY) {
    console.log("Skipping: OPENAI_API_KEY not set");
    return;
  }

  // Create LangChain LLM with streaming
  const llm = new ChatOpenAI({
    model: "gpt-4o-mini",
    temperature: 0.7,
    streaming: true,
  });

  // Wrap it with Caliche adapter
  const provider = new LangChainLLMProvider(llm, false, true);

  // Create agent with LangChain provider
  const agent = await new AgentBuilder().langchainProvider(provider).build();

  console.log("Streaming response:");
  console.log("-".repeat(50));

  // Note: Streaming API may vary - check Caliche streaming documentation
  const response: ChatResponse = await agent.chat("Write a short poem about AI.");
  console.log(response.response);
  console.log("-".repeat(50));
  console.log(`Token usage: ${response.usage ? JSON.stringify(response.usage, null, 2) : "N/A"}\n`);
}

async function exampleWithTools(): Promise<void> {
  console.log("=== Example 4: LangChain Provider with Tools ===\n");

  if (!process.env.OPENAI_API_KEY) {
    console.log("Skipping: OPENAI_API_KEY not set");
    return;
  }

  // Create LangChain LLM
  const llm = new ChatOpenAI({
    model: "gpt-4o-mini",
    temperature: 0.7,
  });

  // Wrap it with Caliche adapter
  const provider = new LangChainLLMProvider(llm, true, true);

  // Create agent with LangChain provider and multiple tools
  const agent = await new AgentBuilder()
    .langchainProvider(provider)
    .enableTool("calculator")
    .enableTool("fs")
    .enableTool("fs")
    .build();

  console.log("Agent with LangChain provider and tools");
  console.log(`Provider: ${agent.provider}, Model: ${agent.model}\n`);

  // Multi-turn conversation with tool usage
  const response1: ChatResponse = await agent.chat("Calculate 123 * 456 using the calculator.");
  console.log(`Turn 1: ${response1.response}`);
  console.log(`Usage: ${response1.usage ? JSON.stringify(response1.usage, null, 2) : "N/A"}\n`);

  const response2: ChatResponse = await agent.chat("Now echo back: 'Hello from Caliche!'");
  console.log(`Turn 2: ${response2.response}`);
  console.log(`Usage: ${response2.usage ? JSON.stringify(response2.usage, null, 2) : "N/A"}\n`);
}

async function main(): Promise<void> {
  console.log("=".repeat(60));
  console.log("LangChain Provider Integration Examples");
  console.log("=".repeat(60));
  console.log();

  // Check for at least one API key
  const hasOpenAI = Boolean(process.env.OPENAI_API_KEY);
  const hasAnthropic = Boolean(process.env.ANTHROPIC_API_KEY);

  if (!hasOpenAI && !hasAnthropic) {
    console.error("Error: No API keys found");
    console.log("\nPlease set at least one of:");
    console.log("  export OPENAI_API_KEY='<openai-api-key>'");
    console.log("  export ANTHROPIC_API_KEY='<anthropic-api-key>'");
    process.exit(1);
  }

  try {
    await exampleOpenAIProvider();
    await exampleAnthropicProvider();
    await exampleStreaming();
    await exampleWithTools();

    console.log("=".repeat(60));
    console.log("All examples completed successfully!");
    console.log("=".repeat(60));
  } catch (error) {
    console.error("\nError running examples:", error);
    process.exit(1);
  }
}

main().catch(console.error);

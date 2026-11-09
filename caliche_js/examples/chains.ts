/**
 * Caliche Chains Examples
 *
 * Demonstrates how to use LLM chains, sequential chains, and map-reduce chains
 * for composable LLM workflows.
 *
 * bun run examples/chains.ts
 */

import { AgentBuilder, LLMChain, MapReduceChain, SequentialChain } from "../src/index";

async function exampleLLMChain() {
  console.log("=== LLM Chain Example ===\n");

  // Create an agent
  const agent = await new AgentBuilder()
    .provider("openai")
    .model("gpt-4o-mini")
    .apiKey(process.env.OPENAI_API_KEY || "")
    .build();

  // Create a chain with a prompt template
  const chain = new LLMChain(agent, "Summarize the following text in one sentence: {text}");

  // Run the chain with input variables
  const result = await chain.run({
    text: "Artificial intelligence is transforming industries across the globe. From healthcare to finance, AI technologies are enabling new capabilities and improving efficiency. Machine learning algorithms can process vast amounts of data to identify patterns and make predictions that were previously impossible.",
  });

  console.log("Input:", result.output.text);
  console.log("Summary:", result.output.output || JSON.stringify(result.output));
  console.log();
}

async function exampleSequentialChain() {
  console.log("=== Sequential Chain Example ===\n");

  const agent = await new AgentBuilder()
    .provider("openai")
    .model("gpt-4o-mini")
    .apiKey(process.env.OPENAI_API_KEY || "")
    .build();

  // Create multiple chains that will execute in sequence
  const analyzeChain = new LLMChain(
    agent,
    "Analyze the following topic and identify key points: {input}"
  );
  const summarizeChain = new LLMChain(agent, "Summarize the analysis: {output}");
  const formatChain = new LLMChain(agent, "Format as a bulleted list: {output}");

  // Create a sequential chain
  const sequentialChain = new SequentialChain();
  sequentialChain.addChain("analyze", analyzeChain);
  sequentialChain.addChain("summarize", summarizeChain);
  sequentialChain.addChain("format", formatChain);

  // Run the sequential chain
  const result = await sequentialChain.run({
    input: "The impact of renewable energy on global economies",
  });

  console.log("Final output:", result.output.output || JSON.stringify(result.output));
  console.log();
}

async function exampleMapReduceChain() {
  console.log("=== Map-Reduce Chain Example ===\n");

  // Create agents for map and reduce operations
  const mapAgent = await new AgentBuilder()
    .provider("openai")
    .model("gpt-4o-mini")
    .apiKey(process.env.OPENAI_API_KEY || "")
    .build();

  const reduceAgent = await new AgentBuilder()
    .provider("openai")
    .model("gpt-4o-mini")
    .apiKey(process.env.OPENAI_API_KEY || "")
    .build();

  // Create map-reduce chain (takes agents and prompts directly)
  const mapReduceChain = new MapReduceChain(
    mapAgent,
    "Extract the main theme from this article: {text}",
    reduceAgent,
    "Combine these themes into a unified summary: {items}"
  );

  // Process multiple articles in parallel, then combine results
  const articles = [
    {
      text: "Article 1: Quantum computing promises to revolutionize cryptography and drug discovery...",
    },
    { text: "Article 2: Machine learning models are becoming more efficient and accessible..." },
    { text: "Article 3: Cloud computing infrastructure enables scalable AI deployments..." },
  ];

  const result = await mapReduceChain.run(articles);

  console.log("Combined summary:", result.output.output || JSON.stringify(result.output));
  console.log();
}

async function main() {
  try {
    await exampleLLMChain();
    await exampleSequentialChain();
    await exampleMapReduceChain();
  } catch (error) {
    console.error("Error:", error);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

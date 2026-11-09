/**
 * AI SDK v6 Provider + MCP Interop Example
 *
 * Demonstrates Caliche as an AI SDK LanguageModelV3 provider.
 *
 * Usage:
 *   bun run examples/ai_sdk_provider.ts
 */

import { generateText, streamText, tool } from "ai";
import { z } from "zod";
import { type CalicheMcpServerBinding, caliche, InProcessMcpServerBuilder } from "../src";

async function basicGenerationExample(): Promise<void> {
  console.log("=== Example 1: generateText with caliche(...) ===\n");

  const result = await generateText({
    model: caliche("openai/gpt-4o-mini", {
      apiKey: process.env.OPENAI_API_KEY,
      systemPrompt: "Be concise and factual.",
    }),
    prompt: "Explain what Caliche is in one sentence.",
  });

  console.log(result.text);
  console.log();
}

async function streamingExample(): Promise<void> {
  console.log("=== Example 2: streamText with caliche(...) ===\n");

  const result = streamText({
    model: caliche("openai/gpt-4o-mini", {
      apiKey: process.env.OPENAI_API_KEY,
    }),
    prompt: "Write two short lines about Rust performance.",
  });

  for await (const chunk of result.textStream) {
    process.stdout.write(chunk);
  }
  process.stdout.write("\n\n");
}

async function toolExample(): Promise<void> {
  console.log("=== Example 3: AI SDK tool loop (client-executed) ===\n");

  const result = await generateText({
    model: caliche("openai/gpt-4o-mini", {
      apiKey: process.env.OPENAI_API_KEY,
    }),
    prompt: "Call add_numbers with a=4 and b=9, then answer with only the result.",
    tools: {
      add_numbers: tool({
        description: "Add two numbers.",
        inputSchema: z.object({
          a: z.number(),
          b: z.number(),
        }),
        execute: async ({ a, b }) => `${a + b}`,
      }),
    },
  });

  console.log(result.text);
  console.log();
}

async function mcpInteropExample(): Promise<void> {
  console.log("=== Example 4: MCP discovery + invocation bridge ===\n");

  const server = new InProcessMcpServerBuilder("calc-mcp")
    .version("1.0.0")
    .addTool({
      name: "multiply",
      description: "Multiply two numbers",
      parameters: {
        type: "object",
        properties: {
          x: { type: "number" },
          y: { type: "number" },
        },
        required: ["x", "y"],
      },
      execute: (args) => {
        const x = Number(args.x ?? 0);
        const y = Number(args.y ?? 0);
        return { success: true, content: String(x * y) };
      },
    })
    .build();

  const mcpServers: CalicheMcpServerBinding[] = [
    {
      name: "local",
      server,
      resources: [
        {
          uri: "resource://local/readme",
          name: "Local README",
          description: "Example MCP resource descriptor",
          mimeType: "text/plain",
        },
      ],
      prompts: [{ name: "local_helper", description: "Example MCP prompt descriptor" }],
    },
  ];

  const model = caliche("openai/gpt-4o-mini", {
    apiKey: process.env.OPENAI_API_KEY,
    mcpServers,
  });

  const snapshot = await model.getMcpSnapshot();
  console.log(
    "MCP tools discovered:",
    snapshot.tools.map((item) => item.name)
  );
  console.log(
    "MCP resources discovered:",
    snapshot.resources.map((item) => item.uri)
  );
  console.log(
    "MCP prompts discovered:",
    snapshot.prompts.map((item) => item.name)
  );
  console.log();
}

async function main(): Promise<void> {
  if (!process.env.OPENAI_API_KEY) {
    console.log("Skipping example run: OPENAI_API_KEY is not set.");
    console.log("Set OPENAI_API_KEY to run this integration example.");
    return;
  }

  await basicGenerationExample();
  await streamingExample();
  await toolExample();
  await mcpInteropExample();
}

main()
  .catch((error) => {
    console.error("AI SDK provider example failed:", error);
    process.exit(1);
  })
  .finally(() => {
    process.exit(0);
  });

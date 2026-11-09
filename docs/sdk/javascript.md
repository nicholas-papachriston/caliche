# JavaScript and TypeScript SDK

The JavaScript SDK calls the Rust runtime through native bindings built with NAPI-RS. The examples use Bun.

## Packages

| Package | Contents |
| --- | --- |
| `@caliche/runtime-node` | Loads the native runtime. |
| `@caliche/core` | The agent builder, the agent runtime, and streaming helpers. |
| `@caliche/contracts` | Wire types and parsers, with no native binding. |
| `@caliche/mcp` | MCP server builders and SSE transport helpers. |
| `@caliche/hooks` | Lifecycle hooks and session controls. |
| `@caliche/adapters-ai-sdk` | An adapter for the Vercel AI SDK. |
| `@caliche/adapters-langchain` | An adapter for LangChain. |

## Installation

The packages are marked `private` in their `package.json` files and are not published to npm, so build them from a checkout:

```bash
cd caliche_js
bun install
bun run develop
```

`bun run develop` builds a debug native binding, builds the TypeScript packages, and checks that the packages can be imported.

`@caliche/core` depends on `@caliche/runtime-node`, so you do not need to add the runtime package yourself. Add `@caliche/mcp`, `@caliche/hooks`, or an adapter package only when you use it.

`@caliche/runtime-node` does not contain a native binary. It lists one package per platform, such as `@caliche/runtime-node-darwin-arm64`, in `optionalDependencies`, and the package manager installs only the one that matches the host. The platform packages are in `caliche_js/npm/`, which is a workspace, and every `@caliche/*` dependency uses a `workspace:` specifier, so no `@caliche/*` name is ever fetched from the public registry. `tests/workspace-internal-deps.test.ts` fails if one does not. In a checkout, the loader uses the binding that `bun run develop` or `bun run build:rust` writes next to `index.js`.

`zod` and `zod-to-json-schema` are optional peer dependencies of `@caliche/core`. Install them only when you pass Zod schemas. The LangChain adapter loads `zod-to-json-schema` only when a LangChain tool carries a Zod schema.

`bun run build:rust` builds the release binding with the `release-bindings` Cargo profile. A Rust panic in an async call then rejects the promise, and a panic in a synchronous call throws a JavaScript error. Neither ends the process. The script then loads the built `.node` file, calls `newEntryId()`, and fails if the module does not load.

## Running an agent

```typescript
import { AgentBuilder, isTextChunkEvent } from "@caliche/core";

const agent = await new AgentBuilder()
  .provider("openai")
  .model("gpt-4o")
  .systemPrompt("You are a helpful assistant.")
  .build();

const response = await agent.chat("Hello");
console.log(response.response);

for await (const event of agent.streamChat("Tell me a story")) {
  if (isTextChunkEvent(event)) {
    process.stdout.write(event.content);
  }
}
```

## Builder configuration

```typescript
const agent = await new AgentBuilder()
  .runtimeConfig({
    provider: "openai",
    model: "gpt-4o",
    credentials: {
      defaultApiKey: process.env.OPENAI_API_KEY,
    },
  })
  .temperature(0.7)
  .maxTokens(2048)
  .enableTool("calculator")
  .enableTools(["fs", "http"])
  .jsonMode()
  .build();
```

## Structured output

```typescript
const schema = {
  type: "object",
  properties: {
    name: { type: "string" },
    age: { type: "integer" },
  },
  required: ["name", "age"],
} as const;

const agent = await new AgentBuilder()
  .provider("openai")
  .model("gpt-4o")
  .jsonSchema("Person", schema)
  .build();
```

## Attachments

```typescript
import { readFileSync } from "node:fs";

const attachments = [
  {
    type: "image" as const,
    data: readFileSync("screenshot.png").toString("base64"),
    media_type: "image/png",
    filename: "screenshot.png",
  },
];

const response = await agent.chatWithAttachments(
  "Explain this screenshot.",
  attachments
);
```

## Single completions without an agent

Use `LlmClientBuilder` for one-off completions that need no tools or memory.

```typescript
import { LlmClientBuilder, isTextChunkEvent } from "@caliche/core";

const client = await new LlmClientBuilder()
  .provider("openai")
  .model("gpt-4o")
  .apiKey(process.env.OPENAI_API_KEY ?? "")
  .build();

const response = await client.completeText("What is 2 + 2?");
console.log(response);

for await (const event of client.streamText("Tell me a story")) {
  if (isTextChunkEvent(event)) {
    process.stdout.write(event.content);
  }
}
```

## Memory

```typescript
builder.bufferMemory({
  max_tokens: 4096,
  min_messages: 4,
  preserve_system: true,
  summary_placeholder: true,
});

builder.summaryMemory({
  max_tokens: 8192,
  summarize_threshold_pct: 0.8,
  chunk_size: 12,
  max_summaries: 8,
});

builder.vectorMemory({
  max_tokens: 8192,
  max_memories: 1000,
  top_k: 5,
  similarity_threshold: 0.7,
  buffer_tokens: 4096,
});
```

## Vector backends

```typescript
builder.vectorBackendMemory(86400, 100000);
builder.vectorBackendRedis("redis://localhost:6379", "prefix:");
builder.vectorBackendQdrant("http://localhost:6333", 1536, "kb_", apiKey);
builder.vectorBackendQdrantGrpc("http://localhost:6334", 1536, "kb_", apiKey, 100);
builder.vectorBackendPinecone(indexHost, apiKey, 1536, 30);
builder.vectorBackendChroma("http://localhost:8000", "prefix_", 1536, 30);
builder.vectorBackendWeaviate("http://localhost:8080", 1536, "prefix_", apiKey);
builder.vectorBackendMilvus("http://localhost:19530", 1536, "prefix_");
builder.vectorBackendDynamodb("vectors", "embeddings", "us-east-1", 1536, false);
builder.vectorBackendFromEnv();
```

A tiered backend keeps frequently used vectors in a fast tier and the rest in slower tiers:

```typescript
builder.vectorBackendTiered(
  [
    {
      backend: { type: "memory", ttlSecs: 3600, maxEntries: 10_000 },
      tierConfig: { name: "hot", maxVectors: 10_000, promotionThreshold: 1 },
    },
    {
      backend: { type: "redis", url: "redis://localhost:6379", prefix: "tier:" },
      tierConfig: { name: "warm", maxVectors: 100_000, promotionThreshold: 2 },
    },
  ],
  {
    writeStrategy: "write_to_source_and_hot",
    backgroundSync: true,
    syncIntervalSecs: 60,
  }
);
```

## Custom tools

```typescript
const customTool = {
  name: "echo",
  description: "Echo input",
  parameters: {
    type: "object",
    properties: { input: { type: "string" } },
    required: ["input"],
  },
  execute: async (args: unknown) => ({
    success: true,
    content: JSON.stringify(args),
  }),
};

builder.addTool(customTool);
```

## Permission modes and allowed tools

```typescript
import { PermissionMode } from "@caliche/hooks";

const agent = await new AgentBuilder()
  .provider("openai")
  .model("gpt-4o")
  .permissionMode(PermissionMode.acceptEdits())
  .allowedTools(["calculator", "fs"])
  .build();
```

## Running an MCP server

```typescript
import { McpServerBuilder } from "@caliche/mcp";

const server = await new McpServerBuilder()
  .name("example-mcp-server")
  .version("1.0.0")
  .enableTools(true)
  .build();

await server.runStdio();
```

## Building and testing from source

Run these commands in `caliche_js`:

```bash
bun install
bun run build
bun run typecheck
bun run test:types
bun test
```

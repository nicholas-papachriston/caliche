# Caliche JavaScript packages

This folder contains the JavaScript and TypeScript packages for Caliche. They call the Rust runtime from Node.js or Bun through native bindings built with NAPI-RS.

## Packages

| Package | Contents |
| --- | --- |
| `@caliche/runtime-node` | Loads the native runtime. Each platform's native module is in its own package, `@caliche/runtime-node-<platform>`, under `npm/`. |
| `@caliche/core` | The agent builder, the agent runtime, and streaming helpers. |
| `@caliche/contracts` | Wire types and parsers, with no native binding. |
| `@caliche/mcp` | MCP server builders and SSE transport helpers. |
| `@caliche/hooks` | Lifecycle hooks and session controls. |
| `@caliche/adapters-ai-sdk` | An adapter for the Vercel AI SDK. |
| `@caliche/adapters-langchain` | An adapter for LangChain. |

## Installation

The packages are marked `private` in their `package.json` files and are not published to npm, so build them from this checkout:

```bash
bun install
bun run develop
```

`bun run develop` builds a debug native binding, builds the TypeScript packages, and checks that every public package can be imported.

`@caliche/core` depends on `@caliche/runtime-node`, so you do not need to add the runtime package yourself. Use `@caliche/mcp`, `@caliche/hooks`, or an adapter package only when you need it.

## Requirements

- Node.js 18 or later, or Bun 1.4.2 or later.
- TypeScript 5.0 or later for TypeScript projects.
- A Rust toolchain, to build the native binding.
- Bun as the package manager. `packageManager` in `package.json` pins the version.

## Quick start

```typescript
import { AgentBuilder, BuiltinTool, isTextChunkEvent } from "@caliche/core";

const agent = await new AgentBuilder()
  .provider("openai")
  .model("gpt-4o")
  .systemPrompt("You are a helpful assistant.")
  .enableTool("calculator")
  .build();

const response = await agent.chat("What is 2 + 2?");
console.log(response.response);

// A built-in tool can also be called directly:
const calc = BuiltinTool.calculator();
console.log(calc.execute({ operation: "add", a: 2, b: 2 }));

for await (const event of agent.streamChat("Tell me a story")) {
  if (isTextChunkEvent(event)) {
    process.stdout.write(event.content);
  }
}
```

## Imports

```typescript
import { AgentBuilder, BuiltinTool } from "@caliche/core";
import { McpServerBuilder } from "@caliche/mcp";
import { PermissionMode, MutationStyle } from "@caliche/hooks";
import { caliche } from "@caliche/adapters-ai-sdk";
import { LangChainAdapter } from "@caliche/adapters-langchain";
```

## Streaming events

Stream iterators emit event types in snake case, matching the Rust runtime. Tool calls produce `tool_call_start` and `tool_call_result` events. `@caliche/core` exports type guards for the events, such as `isTextChunkEvent`, `isToolCallStartEvent`, and `isToolCallResultEvent`.

## Lifecycle hooks

Lifecycle hooks from `@caliche/hooks` deny the tool call when they fail. If an `onPreToolUse` hook throws, returns invalid JSON, or returns no `decision` field, the tool does not run.

## Documentation

The documentation is in the `docs/` folder of the repository:

- [JavaScript and TypeScript SDK](../docs/sdk/javascript.md)
- [MCP integration](../docs/integrations/mcp.md)
- [Feature flags](../docs/reference/feature-flags.md)
- [Environment variables](../docs/reference/environment-variables.md)

## Examples

The examples are in `examples/`:

```bash
bun run examples/agent.ts
bun run examples/streaming.ts
bun run examples/mcp_server.ts
```

## Developing from source

```bash
bun install
bun run develop
bun run typecheck
bun run test:types
bun test
```

| Script | What it does |
| --- | --- |
| `bun run develop` | Builds a debug native binding and the TypeScript packages, then checks that the packages can be imported. |
| `bun run build:rust` | Builds the release native binding with the `release-bindings` Cargo profile. |
| `bun run build` | Compiles the TypeScript and prepares the packages. |
| `bun run smoke` | Imports every public package in the workspace. |
| `bun run lint` | Runs `oxfmt` and `oxlint --fix`, then fails if any problems remain. |
| `bun run lint:fix` | Runs `oxfmt` and `oxlint --fix`. |
| `bun run format` | Runs `oxfmt`. |

## License

Licensed under either of [Apache-2.0](../LICENSE-APACHE) or [MIT](../LICENSE-MIT), at your option.

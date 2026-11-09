# MCP

Caliche supports the Model Context Protocol (MCP) in both directions: it can expose tools as an MCP server, and it can use tools from external MCP servers.

## Running an MCP server

```rust
use caliche::mcp::{McpServerBuilder, Tool};
use std::sync::Arc;

let server = McpServerBuilder::new()
    .name("my-mcp-server")
    .version("1.0.0")
    .enable_tools(true)
    .add_tool(Arc::new(MyTool))
    .build()?;

server.run_stdio_transport().await?;
```

The MCP server supports:

- tools
- prompts
- resources
- the stdio transport
- protocol version `2026-07-28`

The server speaks only protocol version `2026-07-28`. A client includes that version in `_meta` on every request and calls `server/discover`. When Caliche acts as a client and the remote server does not support `server/discover`, Caliche falls back to sending `initialize`.

## The `caliche_mcp` crate

The `crates/caliche_mcp` crate contains the protocol types, cursor pagination, protocol version dates, provider traits, and a JSON-RPC router backed by those providers. The crate does not depend on the Caliche engine, so other hosts can reuse the same wire schema and the same list and call patterns.

Caliche keeps thin adapters under `src/mcp/` that expose its tool registries, templates, and RAG collections through those providers and handlers.

## Protocol behavior

- The server advertises only `2026-07-28`, and `server/discover` lists only that version.
- `initialize`, `ping`, and `logging/setLevel` are not server methods.
- Every result includes `resultType`. List and read results also include `ttlMs` and `cacheScope`.
- The list methods (`tools/list`, `prompts/list`, `resources/list`, and `resources/templates/list`) use opaque cursors. An invalid cursor returns the JSON-RPC error `-32602 Invalid params`.
- The optional metadata fields `title`, `icons`, `theme`, `outputSchema`, `structuredContent`, and `_meta` are serialized when they are set.
- A tool result that includes `structuredContent` also includes the same JSON as a text content part, for compatibility with clients that read only text.
- The advertised capabilities are accurate: `listChanged` stays `false` because the server does not send list-changed notifications, and logging is not advertised.

When the client receives a result whose `resultType` is `input_required`, it returns `McpInputRequired`. The client does not complete that extra exchange on its own. An HTTP or HTTPS URL in that payload must pass the MCP egress policy, and a URL that the policy denies is not returned.

`subscriptions/listen` returns a `subscriptionId`. It does not open a long-lived stream, and `listChanged` stays `false`.

If a cancel notification arrives before the request it cancels, that request returns `-32800`, provided both arrive at the same handler. The stdio transport keeps one handler for the whole connection. The HTTP transport builds a new handler for each POST, so a cancellation never affects another client or a later request. A tool that is already running is not interrupted.

When a `HookRegistry` is attached to the server, a pre-tool hook that denies the call blocks it. A hook that asks for approval also blocks the call, because this path has no approval broker. An `Ask` decision from the compiled policy does not block the call.

HTTP requests must send the `MCP-Protocol-Version`, `Mcp-Method`, and `Mcp-Name` headers. A mismatch returns `-32020`. The server does not use `GET /mcp`, `DELETE /mcp`, or `mcp-session-id`.

Connecting to an MCP server and calling its tools both use the compiled policy snapshot. A `Deny` decision blocks the call, and an `Ask` decision does not. HTTP URLs are also checked against the MCP egress class. When `CALICHE_MCP_COMMAND_ALLOWLIST` is set, stdio commands must be on it.

The following parts of the protocol are intentionally not implemented: the Tasks extension, OAuth client registration, MCP Apps, and cancelling a tool that has already started.

## Using external MCP servers

```rust
use caliche::{AgentBuilder, ProviderType};

let agent = AgentBuilder::new()
    .provider(ProviderType::OpenAI)
    .model("gpt-4o-mini".to_string())
    .add_stdio_mcp_server("hn", "uvx", ["mcp-hn"])
    .add_streamable_http_mcp_server("filesystem", "http://127.0.0.1:8000/mcp")
    .build()
    .await?;
```

Tools from a remote server are registered with the server name as a prefix, for example `hn__top_stories`.

## Allowlist for stdio MCP servers

When you register a stdio MCP server through the HTTP API (`/api/mcp/config`), its executable must be listed in `CALICHE_MCP_COMMAND_ALLOWLIST`:

```bash
CALICHE_MCP_COMMAND_ALLOWLIST=npx,uvx
```

Requests for any other command are rejected. A bare name such as `npx` does not allow a path such as `/tmp/npx`; a command given as a path must be listed with exactly that path. `AgentBuilder::add_stdio_mcp_server(...)` applies the same allowlist when the variable is set, and requires the variable when `CALICHE_ENV=production`. `dangerously_add_stdio_mcp_server` skips the check.

## In-process MCP servers

Use an in-process MCP server when its tools should run in the same process as the agent.

```rust
use caliche::mcp::InProcessMcpServer;
use std::sync::Arc;

let server = InProcessMcpServer::new("my-tools")
    .version("1.0.0")
    .add_tool(Arc::new(MyCustomTool))
    .build()?;

let agent = AgentBuilder::new()
    .mcp_server("tools", server)
    .build()
    .await?;
```

## Related examples

- `examples/plugins.rs`
- `examples/agent_controls.rs`
- `caliche_js/examples/mcp_server.ts`
- `caliche_py/examples/mcp_server.py`

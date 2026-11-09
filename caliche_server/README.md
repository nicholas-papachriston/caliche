# caliche_server

`caliche_server` is the HTTP server for Caliche agents. It serves chat, agent sessions, files, gateway webhooks, and the other routes listed below. The CLI runs it with `caliche --server`, the crate also builds a standalone `caliche_server` binary, and you can mount its router in your own program.

Authentication is off by default, the server listens on `127.0.0.1`, and every caller shares one session list. Read [Security](../SECURITY.md) before you expose the server on a network.

## Building

```bash
git clone https://github.com/nicholas-papachriston/caliche.git
cd caliche
cargo build --release -p caliche_server --features app
```

The default features build a lean server. `app` adds AWS Bedrock and Azure OpenAI, every vector store, and the Redis and PostgreSQL session stores. The Docker image uses `app`. A server built without `app` refuses `SESSION_STORE_BACKEND=redis` and `SESSION_STORE_BACKEND=postgres` at startup. The `bedrock`, `redis`, and `postgres` features add one of these at a time.

| Feature | What it adds |
| --- | --- |
| `mcp`, `templates`, `router` | MCP, prompt templates, and provider routing. These are the default features. |
| `app` | The cloud providers, the vector stores, and the Redis and PostgreSQL session stores. |
| `bedrock`, `redis`, `postgres` | AWS Bedrock, the Redis session store, or the PostgreSQL session store, one at a time. |
| `tools-rag`, `document-loaders`, `chains`, `retrieval-chains`, `reasoning-patterns` | The document, chain, and reasoning routes. They are off by default and in the Docker image. |
| `embedded-ui` | The web UI at `/`. See [Feature flags](../docs/reference/feature-flags.md#embedded-web-ui). |
| `lambda`, `stream-lambda` | AWS Lambda entry points. |
| `otlp` | OpenTelemetry export. |
| `full` | `app`, the default features, and `lambda`. |

With `tools-rag` and an embedding provider (`OPENAI_API_KEY`, or `BEDROCK_EMBEDDING_MODEL_ID` with the `bedrock` feature), the agents that the server builds also get the `rag` tool.

## Starting the server

Start it from the CLI, which installs as `caliche`:

```bash
caliche --server
caliche --server --port 8080 --bind 0.0.0.0
```

The first command listens on `127.0.0.1:3000`. The second listens on every interface, logs a warning, and still starts.

The standalone binary takes its settings from flags or the environment variables in [Environment variables](#environment-variables):

```bash
cargo run -p caliche_server -- --port 3000 --provider openai
```

Both choose the provider with the same rule: `--provider`, then the saved settings, then `AGENT_PROVIDER` or `DEFAULT_PROVIDER`, then `openrouter`. The standalone binary does not read the CLI's SQLite settings. See [Which provider a turn uses](../docs/sessions/sessions.md#which-provider-a-turn-uses).

To serve the router from your own program:

```rust
use caliche::ProviderType;
use caliche_server::create_router;
use std::net::SocketAddr;
use tokio::net::TcpListener;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let router = create_router(ProviderType::OpenAI, "gpt-4o".to_string(), None).await?;

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;

    Ok(())
}
```

To run on AWS Lambda, build with `--features lambda` and call `run_lambda_server`. With `stream-lambda`, `run_lambda_streaming_server` streams chat replies through Lambda response streaming (a function URL with `InvokeMode: RESPONSE_STREAM`).

```rust
use caliche_server::lambda::run_lambda_server;

#[tokio::main]
async fn main() -> Result<(), lambda_http::Error> {
    run_lambda_server().await
}
```

## Embedded web UI

With the `embedded-ui` feature, the server serves the web UI at `/` and the API at `/api/*` from the same address. Requests outside the API fall back to the UI. From the repository root, `./caliche rust install` builds the UI and installs the CLI with the feature. For a debug build:

```bash
./caliche web wasm
cargo build -p caliche_cli --features embedded-ui
caliche --server
```

Hashed files under `/assets/` are sent with `Cache-Control: public, max-age=31536000, immutable`, and `index.html` with `no-cache`. A server built without the feature, or built while `caliche_web/dist` was missing, returns `404` for `GET /`. When `GET /health` succeeds, the API is running.

## Storage paths and retention

Unless the environment or the options set them, the server uses these paths:

- The plugin store is `caliche/plugins.db` in the platform data directory. Without one, it is `./caliche_plugins.db`.
- The MCP server list is `caliche/mcp.json` in the platform configuration directory. Without one, it is `./.caliche_mcp.json`.

The fallbacks let the server start in a container that has no platform directories.

Harness sessions are kept in memory for up to one hour without use, up to 1000 sessions, and are also registered with `AgentRuntime`. Eviction after idle time is separate from cancel. `POST /api/agent/sessions/{id}/cancel`, `POST /api/agent/sessions/{id}/stop`, and `DELETE /api/agent/sessions/{id}` end a session, and a later turn on that id returns `404`. See [HTTP sessions](../docs/sessions/sessions.md).

The document collection registry drops an entry that has had no indexing update for seven days. Gateway session bindings are removed when their session is no longer in the session cache.

## Routes

### Chat

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/chat` | Sends a message and returns the reply. |
| `POST` | `/api/chat/stream` | Streams the reply as server-sent events. |
| `POST` | `/api/chat/multipart` | Sends a message with file attachments. |
| `POST` | `/api/chat/stream/multipart` | Streams the reply to a message with file attachments. |
| `POST` | `/api/chat/estimate` | Estimates the cost before you send. |

### Agent

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/agent/execute` | Runs a task and returns the result. |
| `POST` | `/api/agent/stream` | Runs a task and streams its events. |
| `POST` | `/api/agent/chat/stream` | Runs one turn and streams its events. |
| `POST` | `/api/agent/enhanced` | Runs a task with a reasoning pattern. |

### Sessions

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/agent/sessions` | Lists every live session in the process. It returns `[]` when there are none, and it is not filtered by the JWT subject or roles. |
| `POST` | `/api/agent/sessions` | Creates a session. |
| `GET` | `/api/agent/sessions/{id}` | Returns a session. |
| `DELETE` | `/api/agent/sessions/{id}` | Deletes a session. |
| `POST` | `/api/agent/sessions/{id}/cancel` | Cancels a running turn and removes the session. |
| `POST` | `/api/agent/sessions/{id}/stop` | The same as `cancel`. |
| `POST` | `/api/agent/sessions/{id}/turn` | Runs one turn. |
| `POST` | `/api/agent/sessions/{id}/task` | Runs a task. |
| `GET` | `/api/agent/sessions/{id}/events` | Streams the session events. |
| `POST` | `/api/agent/sessions/{id}/checkpoint` | Records a checkpoint. |
| `POST` | `/api/agent/sessions/{id}/rollback` | Rolls back to a checkpoint. |
| `POST` | `/api/agent/sessions/{id}/undo` | Undoes the last file change. |
| `GET` | `/api/agent/sessions/{id}/diff` | Lists the file changes, optionally for one `message_id`. |
| `POST` | `/api/agent/sessions/{id}/revert` | Reverts the file changes of the last message or a given message. |
| `POST` | `/api/agent/sessions/{id}/unrevert` | Restores the last reverted message. |
| `POST` | `/api/agent/sessions/{id}/clear` | Clears the conversation. |
| `GET`, `POST` | `/api/agent/sessions/{id}/mode` | Reads or sets the session mode. |
| `GET`, `DELETE` | `/api/agent/sessions/{id}/plan` | Reads or clears the session plan. |

With authentication off, any caller that can reach the server can cancel any session.

### Shares

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/agent/sessions/{id}/share` | Creates a redacted share and returns `/s/{slug}`. The optional body is `{"visibility": "public"}` or `{"visibility": "authenticated"}`. |
| `DELETE` | `/api/agent/sessions/{id}/share` | Revokes the shares of a session. |
| `GET` | `/api/agent/shares/{slug}` | Returns a share as JSON. |
| `DELETE` | `/api/agent/shares/{slug}` | Revokes one share. |
| `GET` | `/s/{slug}` | Shows a share as HTML, or as JSON with `Accept: application/json`. |

An `authenticated` share requires a valid token to read.

### Context and memory

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/agent/sessions/{id}/context` | Returns the workspace context. |
| `POST` | `/api/agent/sessions/{id}/context/track` | Tracks a file. |
| `POST` | `/api/agent/sessions/{id}/context/refresh-git` | Refreshes the Git context. |
| `POST` | `/api/agent/sessions/{id}/context/detect-project` | Detects the project type. |
| `GET` | `/api/agent/sessions/{id}/context/relevant` | Returns the relevant context. |
| `GET` | `/api/agent/sessions/{id}/memory/stats` | Returns memory statistics. |
| `GET` | `/api/agent/sessions/{id}/memory/inspect` | Returns the memory state. |
| `GET` | `/api/agent/sessions/{id}/memory/search?q=...` | Searches the memory. |
| `POST` | `/api/agent/sessions/{id}/memory/clear` | Clears the memory. |
| `GET` | `/api/agent/sessions/{id}/memory/export` | Exports a memory snapshot. |
| `POST` | `/api/agent/sessions/{id}/memory/import` | Imports a memory snapshot from a JSON body. |

### Files

These routes need `CALICHE_FILE_BROWSING=true`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/files/enabled` | Reports whether file browsing is on. |
| `GET` | `/api/files/list?path=X` | Lists a directory. |
| `GET` | `/api/files/read?path=X` | Reads a file. |

### Gateway

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/gateway/adapters` | Lists the gateway adapters. |
| `GET` | `/api/gateway/delivery/diagnostics` | Returns delivery diagnostics. |
| `POST` | `/api/gateway/ingress/simulate` | Simulates an incoming message with a binding key. |
| `POST` | `/api/gateway/ingress/simulate-envelope` | Simulates an incoming message with a full envelope. |
| `POST` | `/api/gateway/ingress/telegram` | Receives Telegram webhooks. |
| `POST` | `/api/gateway/ingress/discord` | Receives Discord webhooks. |
| `POST` | `/api/gateway/ingress/twilio` | Receives Twilio webhooks. |
| `POST` | `/api/gateway/ingress/slack` | Receives Slack events. |
| `POST` | `/api/gateway/ingress/teams` | Receives Microsoft Teams webhooks. |

### Other routes

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/tools` | Lists the tools. |
| `GET` | `/api/models?provider=X` | Lists the models. |
| `GET` | `/api/storage/options` | Lists the vector backend options. |
| `POST` | `/api/reasoning/react`, `/api/reasoning/plan-execute`, and others | Runs a reasoning pattern. Needs `reasoning-patterns`. |
| `POST` | `/api/chains/llm`, `/api/chains/sequential`, `/api/chains/map-reduce`, and others | Runs a chain. Needs `chains`. |
| `POST` | `/api/documents/index` | Indexes documents. Needs `tools-rag`. |
| `POST` | `/api/documents/search` | Searches indexed documents. Needs `tools-rag`. |
| `POST` | `/api/orchestration/workflow` | Runs a workflow. |
| `GET` | `/api/templates` | Lists the agent templates. |
| `GET` | `/health` | Reports that the server is running. |
| `GET` | `/metrics` | Returns Prometheus metrics. |

## Examples

Send a chat message:

```bash
curl -X POST http://127.0.0.1:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 2+2?", "provider": "openai", "model": "gpt-4o"}'
```

Stream a reply with the same body:

```bash
curl -N -X POST http://127.0.0.1:3000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me a story", "provider": "openai", "model": "gpt-4o"}'
```

Send files with a message. The multipart fields are `message` (required), `provider`, `model`, and one `attachments` or `attachments[]` part for each file. Images, audio, video, PDF, DOCX, XLSX, plain text, and CSV files are accepted.

```bash
curl -X POST http://127.0.0.1:3000/api/chat/multipart \
  -F 'message=Analyze these files together' \
  -F 'provider=openai' \
  -F 'model=gpt-4o' \
  -F 'attachments[]=@./image.png' \
  -F 'attachments[]=@./notes.pdf'
```

`/api/chat/stream/multipart` takes the same fields and streams the reply.

Create a session, run a task in it, and cancel it. The create body does not choose the provider, which the server chose at startup, and unknown fields are ignored.

```bash
curl -X POST http://127.0.0.1:3000/api/agent/sessions \
  -H "Content-Type: application/json" \
  -d '{"max_turns": 1}'

curl -X POST http://127.0.0.1:3000/api/agent/sessions/{session_id}/task \
  -H "Content-Type: application/json" \
  -d '{"task": "Write a hello world program in Rust"}'

curl -X POST http://127.0.0.1:3000/api/agent/sessions/{session_id}/cancel
```

## Environment variables

The server reads the same variables as the rest of Caliche. It reads a `.env` file only when you pass `--env-file <path>`, never one in the working directory, and a variable already set in the shell keeps its value. The file does not supply the flags in the table below: set those in the shell or on the command line. The full list is in [Environment variables](../docs/reference/environment-variables.md) and [.env.example](../.env.example). Before you expose the server, follow the [production authentication checklist](../docs/ops/production-auth-checklist.md).

### Standalone binary

| Variable | Flag | Default | Purpose |
| --- | --- | --- | --- |
| `PORT` | `--port` | `3000` | The listen port. |
| `BIND_ADDRESS` | `--bind` | `127.0.0.1` | The listen address. |
| `DEFAULT_MODEL` | `--model` | The provider's default | The model. |
| `DEFAULT_CONTEXT_WINDOW` | `--context-window` | The model's window | The context window in tokens. |
| `CALICHE_TOOLS` | `--tools` | The default tools | A comma-separated list of tools for session routes. |
| `CALICHE_MAX_TURNS` | `--max-turns` | `50` | The maximum number of turns for session routes. |
| `CALICHE_WORKSPACE` | `--workspace` | The working directory | The workspace for session routes and the file browser. |
| `CALICHE_SYSTEM_PROMPT` | `--system-prompt` | None | A system prompt that replaces the default. |
| `CALICHE_PERMISSION_MODE` | `--permission-mode` | `default` | The permission mode. |
| `CALICHE_TOOL_EXECUTION_DRIVER` | `--tool-execution-driver` | `auto` | The tool execution driver. |
| `CORS_PERMISSIVE` | `--cors-permissive` | `false` | Allows every CORS origin. Use it only for development. The production profile refuses it unless `CALICHE_ALLOW_INSECURE_PRODUCTION=1`. |

`AGENT_PROVIDER` and `DEFAULT_PROVIDER` take part in the provider rule described in [Starting the server](#starting-the-server). `API_HOST` is not a listen address. The server logs a warning at startup when `API_HOST` or `BIND_ADDRESS` is `0.0.0.0` or `::` and authentication is off.

### Authentication and access

| Variable | Default | Purpose |
| --- | --- | --- |
| `JWT_ENABLED` | `false` | Validates JWTs. Also set `JWT_PUBLIC_KEY`, `JWT_PUBLIC_KEY_PATH`, or `JWT_JWKS_URL`, and `JWT_AUDIENCE` and `JWT_ISSUER`. When the keys cannot be loaded, the server exits. |
| `CALICHE_REQUIRE_AUTH` | `false` | Requires a valid JWT on most `/api/*` routes. It fails at startup without `JWT_ENABLED=true`. |
| `CALICHE_PUBLIC_CHAT` | `false` | Keeps `POST /api/chat/stream` open while `CALICHE_REQUIRE_AUTH` is on. |
| `CALICHE_ENV` | None | `production` refuses to start without JWT and `CALICHE_REQUIRE_AUTH`, unless `CALICHE_ALLOW_INSECURE_PRODUCTION=1`. |
| `CALICHE_ALLOW_UNAUTH_ORCHESTRATION` | `false` | Opens `/api/orchestration/*` and `/api/workflows/*` without a JWT, for local use. The production profile ignores it. |
| `CALICHE_TRUSTED_PROXY`, `CALICHE_TRUSTED_PROXY_CIDRS` | Off | Trusts `X-Forwarded-For` only from a peer in the listed CIDR ranges. |
| `CALICHE_FILE_BROWSING` | `false` | Turns on the file routes, limited to `CALICHE_WORKSPACE`. |
| `CALICHE_JSON_RPC_ONLY` | `false` | Returns `410 Gone` for every `/api/*` route except the chat stream routes. `GET /health`, `GET /metrics`, `POST /rpc`, and `GET /ws` keep working. |
| `CALICHE_MCP_COMMAND_ALLOWLIST` | None | The executables allowed to start a stdio MCP server, such as `npx,uvx`. Stdio MCP servers need it. |

### Routing and observability

| Variable | Default | Purpose |
| --- | --- | --- |
| `CALICHE_ROUTER_STRATEGY` | `static` | The provider routing strategy: `static`, `latency`, `weighted`, or `cost`. |
| `CALICHE_ROUTER_LATENCY_CANDIDATES` | None | The `provider:model` list for the `latency` strategy, such as `openai:gpt-4o-mini,anthropic:claude-sonnet-4-5`. Without it, the server warns and uses `static`. |
| `CALICHE_ROUTER_LATENCY_ALPHA` | `0.2` | The smoothing factor of the moving average of latencies. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | None | The OpenTelemetry endpoint, such as `http://localhost:4318`. It needs the `otlp` feature. |

### Gateway

| Variable | Default | Purpose |
| --- | --- | --- |
| `CALICHE_WEBHOOK_REQUIRE_VERIFICATION` | `false` | Rejects unsigned webhooks and requires each enabled adapter's secret. The production profile turns it on unless `CALICHE_ALLOW_INSECURE_PRODUCTION=1`. |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_WEBHOOK_SECRET_TOKEN` | None | The Telegram token and the optional webhook secret. |
| `DISCORD_BOT_TOKEN`, `DISCORD_PUBLIC_KEY` | None | The Discord token and the optional public key for signatures. |
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM`, `TWILIO_WEBHOOK_VALIDATE_SIGNATURE` | None | The Twilio account, sender, and optional signature check. |
| `SLACK_BOT_TOKEN`, `SLACK_SIGNING_SECRET` | None | The Slack token and the optional signing secret. |
| `TEAMS_WEBHOOK_URL` | None | The outgoing Microsoft Teams webhook. |
| `CALICHE_CLUSTER_REGISTER_TOKEN` | None | The secret for `POST /api/gateway/cluster/register`, sent in the `X-Cluster-Register-Token` header. Without it, only a caller with the `cluster-admin` role can register. |
| `CALICHE_CLUSTER_HOST_ALLOWLIST` | None | The host names allowed in a cluster node's `base_url`. Without it, registration is refused. |
| `CALICHE_CLUSTER_ALLOW_HTTP` | `false` | Allows `http://` node addresses. |
| `CALICHE_CLUSTER_ALLOW_PRIVATE` | `false` | Allows loopback, private, link-local, and unique local addresses for nodes. Leave it off in production. |

## License

Licensed under either of [Apache-2.0](../LICENSE-APACHE) or [MIT](../LICENSE-MIT), at your option.

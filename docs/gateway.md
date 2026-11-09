# Gateway runtime

The gateway runtime receives messages from webhooks and chat channels, binds each incoming message to a session, and can run in the background. This page also covers running the HTTP server as a user service, managing running servers, webhook verification, routing rules, the outbound network (egress) policy, durable suspend and resume, and workspace warm-start.

## How gateway runtimes relate

- The first gateway runtime you start becomes the `main` gateway.
- A gateway runtime started later from another directory registers with the main gateway as a `worker` gateway.
- The main gateway can forward incoming messages to workers and persists which worker each route is assigned to.
- A gateway runtime resolves its provider and model from the defaults set for its directory, and falls back to environment variables.

## Running the server as a user service

You can install a user-level service unit that keeps the HTTP server running across terminal sessions:

```bash
caliche daemon install --port 3000
caliche daemon status
caliche daemon stop
caliche daemon start
caliche daemon uninstall
```

- On macOS, the command writes `~/Library/LaunchAgents/<label>.plist`. The default label is `com.caliche.server`.
- On Linux, the command writes `~/.config/systemd/user/<label>.service`. The default label is `caliche`.

The unit runs `caliche --server --port <N>` from the working directory chosen at installation (`--cwd` on `install`). It sets `CALICHE_SERVER_ORIGIN=daemon`, so the process records that a service unit started it. A `caliche --server` process started in the foreground records the origin `foreground` in the same way.

## Managing running servers

To list and stop every recorded server, including servers on ports you chose yourself, use the `server` commands:

```bash
caliche server list
caliche server status --port 3000
caliche server stop --port 3000
caliche server stop <id>
```

`server stop` signals a process only when all of the following are true: the process ID in the record still belongs to this `caliche` binary (`caliche.exe` on Windows), the process was started with `--server`, and it is still listening on the recorded port. If the process is no longer running, the record and any matching `server.lock` are removed as stale, and nothing is signaled.

Each record also stores a start token, so that a process ID reused by an unrelated process is not signaled:

- On Linux and Windows, a record whose start token does not match the running process is removed without signaling.
- On macOS, the start token is the start time reported by `ps`, read in the C locale and in UTC. If it does not match, the process is left running and the record is kept, because a record written by an older build may hold the time in another time zone or locale.
- Any other mismatch leaves the process running and keeps the record.

On Windows, both the first stop attempt and the forced attempt use `TerminateProcess`, so the server is not given time to shut down gracefully. Once the process has exited, `server stop` removes its record and the matching `server.lock`.

Set `CALICHE_CONFIG_DIR` to a `caliche` directory when the server records, `server.lock`, and `plugins.db` must not use the default profile location. That variable does not move `gateway-cluster.db` or `~/.caliche/logs/gateway`, which stay under the home directory. The CLI logs and `endpoints.json` stay in the platform data directory (`%APPDATA%\caliche` on Windows), and the CLI session database and the CLI `tasks.db` stay under `~/.caliche`.

`daemon stop` unloads the OS service unit. `server list` and `server stop` clean up both service-managed and foreground `--server` processes.

The cluster and forwarding controls of the gateway are available through the `/gateway` command in the REPL.

## REPL commands

The REPL provides these gateway commands:

```text
/gateway setup [--forward --provider cloudflare]
/gateway start|stop|restart|status|list [--port N] [--dir PATH]
/gateway forward start|stop|restart|status|list
/gateway route list|pin|unpin
/gateway twilio webhook sync|status
/gateway llm status [--dir <path>]
/gateway llm set [--provider <provider>] [--model <model>] [--dir <path>]
/gateway llm clear [--dir <path>]
```

If the cluster route endpoints return `404`, restart the gateway runtime:

```text
/creds gateway restart
```

## Cluster route API

The main gateway serves these routes:

| Method | Path |
| --- | --- |
| `GET` | `/api/gateway/cluster/nodes` |
| `POST` | `/api/gateway/cluster/register` |
| `GET` | `/api/gateway/cluster/routes` |
| `POST` | `/api/gateway/cluster/routes/pin` |
| `POST` | `/api/gateway/cluster/routes/unpin` |

## Webhook ingress API

| Method | Path |
| --- | --- |
| `POST` | `/api/gateway/ingress/telegram` |
| `POST` | `/api/gateway/ingress/discord` |
| `POST` | `/api/gateway/ingress/twilio` |
| `POST` | `/api/gateway/ingress/slack` |
| `POST` | `/api/gateway/ingress/teams` |

Twilio callbacks that do not carry a message are acknowledged with a `2xx` response. Only payloads that carry a message are passed to the agent.

## Webhook signature verification

The verification helpers live in `caliche::gateway::webhook_verify`:

- Slack: `v0` HMAC signatures.
- Twilio: HMAC-SHA1 signatures.
- Discord: Ed25519 signatures over the payload bytes.
- Teams: a bearer token or a Bot Framework JWT.

The API crate re-exports these helpers and applies them in the ingress handlers. Set `CALICHE_WEBHOOK_REQUIRE_VERIFICATION=true` to require the verification secrets of every enabled adapter.

Twilio and Slack signatures cover the callback URL. Set `CALICHE_PUBLIC_BASE_URL` so that the server can reconstruct that URL. `X-Forwarded-*` headers are ignored unless `CALICHE_TRUST_FORWARDED_HEADERS=true`.

## Routing rules

Set `CALICHE_GATEWAY_ROUTING_RULES_PATH` to a JSON file of routing rules. YAML files are also accepted when the build includes the optional `serde_norway` dependency, which the `templates` and `profiles` features enable.

Each rule has two parts:

- `match`: one of `channel_id`, `adapter_id`, `user_id`, `tenant_id`, `channel_kind`, `all_of`, or `any_of`.
- `overrides`: `system_prompt_hint`, `session_id_prefix`, or both.

The first rule that matches is applied. The related Rust types are in `caliche::gateway`.

## Egress allowlist

The egress policy controls outbound HTTP requests in general, not only webhook delivery. Requests from untrusted tools (`http`, `web_fetch`, and `web_search`) and similar requests are always checked for server-side request forgery (SSRF): private, loopback, cloud metadata, and carrier-grade NAT (CGNAT) addresses are blocked. In production, or when the mode is `default_deny`, every host is denied until you list it.

Configure the policy with a TOML file, with environment variables, or with both:

```bash
CALICHE_EGRESS_POLICY_PATH=./gateway-egress.toml
CALICHE_EGRESS_MODE=default_deny
CALICHE_EGRESS_ALLOW=api.openai.com,api.anthropic.com
CALICHE_EGRESS_TOOLS=api.github.com
CALICHE_EGRESS_MCP=localhost,mcp.example.com
CALICHE_EGRESS_PROVIDERS=api.openai.com
CALICHE_EGRESS_INCLUDE_DEFAULT_PROVIDERS=true
```

```toml
# gateway-egress.toml
mode = "default_deny"
include_default_providers = true

[allow]
hosts = ["api.openai.com"]
tools = ["api.github.com"]
mcp = ["localhost", "mcp.example.com"]
providers = ["api.anthropic.com"]
```

- `mode = "allow_public"` is the default for local development. Public hosts are allowed. Tool HTTP requests still block loopback and private addresses. MCP and provider URLs may use loopback and private addresses, so that a local server can run. Cloud metadata and link-local addresses stay blocked.
- `mode = "default_deny"` rejects every host that is not listed, with the error `egress denied for <class> HTTP: host '…' is not on the allowlist`.
- Each request belongs to one class: `tool` (HTTP tools and web fetch), `mcp` (streamable HTTP MCP), `provider` (model base URLs), or `plugin` (plugin steps and webhooks, which also honor `PLUGIN_HTTP_ALLOWLIST`).
- A host list that contains only `*` allows any public host.
- In production, an allowlist is required (`CALICHE_EGRESS_ALLOW`, a policy file, or `PLUGIN_HTTP_ALLOWLIST`) unless `CALICHE_ALLOW_INSECURE_PRODUCTION=1`.

The Rust types are `caliche::gateway::{EgressPolicy, EgressClass, EgressMode}`.

## Durable suspend and resume

`AgentRuntime::suspend` saves the conversation, its phase, and its checkpoint identity, then releases the harness from memory. `resume_with` and `resume_from_service` load that snapshot again, including after a process restart. In the REPL, use `/suspend [id]` and `/resume [id]`. Set `CALICHE_SUSPEND_DIR`, or call `AgentRuntime::with_persist_dir`, to choose the snapshot directory.

## Workspace warm-start

A `.caliche/workspace.toml` file can declare the workspace root, Git repositories to clone or attach, MCP servers, and skills, so that a session starts with them already set up instead of discovering them again. Load the file with `caliche::workspace_declaration::WorkspaceDeclaration`, which is also re-exported from `caliche::workspace` when the `tools-rag` feature is enabled. This does not change how the workspace indexer works.

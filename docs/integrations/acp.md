# ACP

The Agent Client Protocol (ACP) is JSON-RPC over newline-delimited messages, at protocol version 1. Caliche can act as the agent that an ACP client talks to over stdio, and it can act as the client of another ACP agent over stdio.

The wire types come from the `agent-client-protocol-schema` crate, version 1.10.2, with its default features off and its v2 feature off. Caliche does not depend on the `agent-client-protocol` runtime crate.

## Caliche as the agent

Start Caliche as an ACP agent on stdio:

```bash
cargo run -p caliche_cli -- --acp-stdio
```

While `session/prompt` is running, Caliche writes `session/update` notifications as the turn progresses, and it reads `session/cancel` during the turn.

The `initialize` response advertises `loadSession` and `sessionCapabilities.list`. `authMethods` is always empty, so provide a provider key or use a local model.

`session/list` reads the same live session cache as `GET /api/agent/sessions`. `session/load` replays the history of the session as `session/update` notifications before it returns.

When a tool call needs approval, Caliche sends `session/request_permission` with the tool name, the arguments, the session id, and the tool call id, and waits up to 60 seconds for the answer. Only an `allow-once` answer runs the tool. A rejection, a cancellation, a malformed result, a timeout, or a closed client skips the tool. The `Default`, `AcceptAll`, and `DenyAll` permission modes do not send this request.

## Caliche as the client

The `caliche_acp_client` crate starts another ACP agent as a child process, negotiates protocol version 1, and runs `session/new`, `session/prompt`, and `session/cancel`. The child's standard output carries the protocol, so Caliche never writes log output to it.

```rust
use caliche_acp_client::{AcpClient, AcpSpawn};

async fn drive_peer() -> Result<(), caliche_acp_client::AcpClientError> {
    let client = AcpClient::spawn(
        AcpSpawn::new(["opencode", "acp"], "/absolute/workspace")
            .env("OPENAI_API_KEY", "operator-supplied"),
    )?;
    let _init = client.initialize().await?;
    let session = client.new_session("/absolute/workspace").await?;
    let session_id = session.session_id.0.to_string();
    let turn = client
        .prompt_text(&session_id, "Summarize the repo")
        .await?;
    let _ = turn.stop_reason;
    Ok(())
}
```

### Authentication

After `initialize`, the client continues only when the peer's `authMethods` list is empty, or when every method id in it is exactly `opencode-login` or `hermes-setup`. Any other id is refused before `session/new`. A provider id listed next to `hermes-setup` is also refused.

The client never calls `authenticate`. It does not set `auth.terminal` or `_meta["terminal-auth"]`, and it does not run `opencode auth login`, `hermes setup`, or `hermes acp --setup`.

If the peer returns `auth_required`, the client stops. Configure a provider key or a local model for that peer outside of Caliche. OpenCode reads `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `opencode.json`. Hermes is configured with `hermes config set` and `model.provider`.

### Child environment

Pass provider keys to the child with `AcpSpawn::env`. By default, the child receives only these variables from the parent: the search path, the home directory, the user name, the locale, the terminal type, the temporary directories, and the Windows startup variables. Any variable whose name contains `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `PASSWD`, or `CREDENTIAL`, or ends in `_PASS` or `_PAT`, is left out. `inherit_env(true)` gives the child the full parent environment; the equivalent flag of `acp-peer-smoke` is `--inherit-env`.

The same reduced environment applies to every child process that Caliche starts: `bash` commands, the validation hook, verification commands, formatters, stdio MCP servers, language servers, plugin shell steps, external agents, and session start probes. Proxy, certificate, toolchain home, `XDG_*`, `VIRTUAL_ENV`, and `SSH_AUTH_SOCK` variables are removed. A `bash` command can read a value that you export with the `CALICHE_BASH_` prefix. A stdio MCP server or an external agent that needs a key receives it from the `env` in its configuration.

The client does not serve `fs/*` or `terminal/*` requests from the peer, and it answers a peer's `session/request_permission` with `cancelled`.

## Testing against a peer

`acp-peer-smoke` runs one turn against a peer. Run it from the repository root. The summary JSON and the JSONL transcript redact environment values that look like credentials.

The exit codes are:

- `0`: the turn finished.
- `2`: the peer's authentication was refused. The summary reports `auth_rejected` for a method id outside the allowlist above, or `auth_required` when the peer has no provider key configured outside of the ACP handshake.
- `3`: the peer binary was not found.

`--peer auto` tries OpenCode, then Hermes. It does not fall back to the built-in fixture peer.

```bash
cargo run -p caliche_acp_client --bin acp-peer-smoke -- --peer fixture
cargo run -p caliche_acp_client --bin acp-peer-smoke -- --peer opencode --inherit-env --transcript /tmp/acp-opencode.jsonl --summary /tmp/acp-opencode.json
cargo run -p caliche_acp_client --bin acp-peer-smoke -- --peer hermes --inherit-env --transcript /tmp/acp-hermes.jsonl --summary /tmp/acp-hermes.json
cargo run -p caliche_acp_client --bin acp-peer-smoke -- --matrix
```

OpenCode (`opencode acp`) advertises only `opencode-login`, so the turn continues. Hermes (`hermes acp`, `hermes-acp`, or `python -m acp_adapter`) continues only when `hermes-setup` is its only method id.

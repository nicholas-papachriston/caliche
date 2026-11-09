# Caliche CLI

The `caliche` binary runs an [agent](../docs/concepts.md#agent) in a REPL, as one prompt, or over HTTP with `--server`. You choose the [provider](../docs/concepts.md#provider). The default tools are `fs`, `bash`, and `web_fetch`. See [Tools](../docs/tools.md).

- interactive REPL
- single-shot automation (`--message`, `--pipe`, JSON/JSONL/stream output)
- HTTP sessions (`--server`) and an optional embedded UI
- agentic coding (`code` subcommand)
- credential/keychain management (providers, vault scopes, gateway secrets)
- session management, diagnostics, templates, evaluation, shell completions, and man pages

Binary name: `caliche`

## Installation

```bash
# Build from source (workspace root)
cargo build -p caliche_cli --release

# Run directly from workspace
cargo run -p caliche_cli -- --help
```

## Quick Start

```bash
# Start interactive mode (default)
caliche

# One-shot message
caliche --message "Explain this repository layout"

# One-shot message without flags
caliche "Explain this repository layout"

# Prompt alias
caliche --prompt "List likely refactors"

# Pipe stdin into a prompt
cat src/main.rs | caliche --pipe "Review this file"

# JSON output for automation
caliche --message "status" --output-format json
```

### Server mode (API + optional embedded UI)

Start the HTTP API server (default port 3000):

```bash
caliche --server
# Explicit every-interface listen (warns and still starts):
# caliche --server --bind 0.0.0.0 --port 3000
```

`--bind` and `--port` set the listen address. The security defaults of the server are in [HTTP server defaults](../SECURITY.md#http-server-defaults).

`--server` listens without a model API key. A turn fails when the booted provider has no key. The process keeps that provider. An egress-blocked provider URL, including a restored localhost Ollama base URL, does not prevent listen. The turn then fails with that provider's egress error.

`caliche doctor` prints `listen_bind` and `default_provider_source` (`flag`, `last`, `env`, or `default`). `last` is `settings.json` beside `mcp.json`. When that file is absent, SQLite `last_provider` and `last_model` are used, and they win over environment variables. `--provider` keeps a recorded model for the same provider. `--model` and `--temperature` override the file. Plain `--server` does not probe for a local runtime. Local auto-detect runs only when `--provider` is unset and either `--prefer-local` is set or the process is opening the REPL. Doctor does not probe. `build_id=unknown` means git was missing at compile time. It does not match `git rev-parse HEAD`.

Provider boot order, session create, and cancel are in [HTTP sessions](../docs/sessions/sessions.md). Authentication is in [Security](../SECURITY.md).

To serve the Dioxus web UI at `/` on the same origin as `/api/*`, install from the workspace root with `./caliche rust install` (wasm UI, then `cargo install --path caliche_cli --locked --force --features embedded-ui`). A debug build is `./caliche web wasm` then `./caliche rust build-with-ui` (or `cargo build -p caliche_cli --features embedded-ui`). When you run `caliche --server`, open that server URL in a browser. The REPL in another terminal attaches through `server.lock`. Hashed assets use long-lived cache headers; `index.html` is revalidated.

Without `embedded-ui`, `GET /` returns 404. If `GET /health` is OK, the API is up. That 404 means the UI feature or assets are absent.

When `CALICHE_JSON_RPC_ONLY=on`, CLI-hosted server features are available through `POST /rpc`. Use `cli.routes` to inspect the CLI parity method list; Cloudflare quick tunnel controls are exposed as `cli.gateway.forward.*` methods.

Every `--server` process writes a record under `~/.config/caliche/servers/` (pid, port, bind, build id, start time, working directory, and `foreground` or `daemon` origin). The daemon unit sets `CALICHE_SERVER_ORIGIN=daemon`. `server.lock` remains the REPL attach pointer for the most recent server.

```bash
caliche server list
caliche server status --port 3000
caliche server stop --port 3060
caliche server stop <id>
```

`server stop` signals a process only when that pid is still `caliche` and is still listening on the recorded port. A dead pid is removed as a stale record, including a matching `server.lock`, and is not signaled. If the pid or port does not match, the process is left running. This is the supported cleanup path.

**Server security:** Credential and vault HTTP routes (`/api/cli/config/*`, `/api/cli/vault/*`) are rejected for non-loopback clients even when bound to `0.0.0.0`. When `JWT_ENABLED=true`, the server uses the same fail-closed JWT stack as `caliche_server` (`CALICHE_REQUIRE_AUTH` applies to protected `/api/*` routes). Set both before exposing the server beyond local development.

## Credential and Config Setup

Before first use, configure provider credentials.

### Provider setup

```bash
# Interactive setup wizard
caliche config setup

# Set/remove/get API keys
caliche config set openai --key sk-...
caliche config get openai
caliche config remove openai

# Use environment variable as source
caliche config set anthropic --env ANTHROPIC_API_KEY

# Set defaults
caliche config default openai
caliche config model openai gpt-4o

# Configure AWS credentials for Bedrock
caliche config aws --profile default
# or
caliche config aws --access-key ... --secret-key ... --region us-east-1

# Export configured credentials to a 0600 temp file, then source it
caliche config export --shell bash
```

Provider secrets are stored in the **OS keyring** (macOS Keychain, Windows Credential Manager, Linux keyutils) with scope metadata in SQLite (`~/.caliche/caliche.db`). Interactive setup and `config` commands write directly to the keyring; `credentials.toml` is not used.

On macOS, data-protection keychain + Touch ID is enabled by default. Set `CALICHE_KEYCHAIN_USE_TOUCH_ID=false` to use the regular keychain path.

### Vault (directory-scoped credentials)

```bash
# Show credentials available for current directory
caliche config vault status

# List all known scopes
caliche config vault list

# Import credentials from .env files (interactive confirm before writing to keyring)
caliche config vault import --dir .

# Link credentials from another directory
caliche config vault link ~/other-project

# Promote scoped credential to global
caliche config vault promote openai
```

### Gateway credentials

```bash
caliche config gateway list
caliche config gateway set telegram_bot_token --key <token>
caliche config gateway set slack_signing_secret --env SLACK_SIGNING_SECRET
caliche config gateway get twilio_account_sid --unmask --yes-i-understand-this-prints-secrets
caliche config gateway remove teams_webhook_url
```

Supported gateway credential names:

- `telegram_bot_token`
- `discord_bot_token`
- `twilio_account_sid`
- `twilio_auth_token`
- `twilio_from`
- `slack_bot_token`
- `slack_signing_secret`
- `teams_webhook_url`

## Core Usage Patterns

### Interactive REPL

```bash
caliche
```

Useful flags:

- `--provider <name>`
- `--model <id>`
- `--session <id>`
- `--resume-last`
- `--save-session`
- `--no-autosave`
- `--tools calculator,fs,...`
- `--profile lean` selects the Lean profile: the `fs` and `bash` tools, a short prompt, and no MCP or skill discovery. It does not change the default agent. See [Configuration](../docs/configuration.md#lean-profile).
- `--sandbox read-only|workspace-write|network-restricted|danger-full-access`
- `--permission-mode default|accept-edits|accept-all|deny-all|interactive`

### Session file changes (REPL)

When the REPL runs with an agent harness (typical agent mode), file edits from `write_file` / `edit_file` are grouped per user message for undo:

- `/diff` with no path shows an overview: active change sets, redo queue depth, touched paths, and undo/redo readiness (same semantics as the web session inspector).
- `/diff <path>` shows a unified diff for one modified file (session-tracked paths).
- `/undo` reverts the latest change set; `/redo` restores the most recently reverted set.

Without a harness, `/diff` alone lists session-tracked modified files if any; use `/diff <path>` for the diff body.

### Non-interactive output modes

```bash
caliche --message "hello" --output-format text
caliche --message "hello" --output-format json
caliche --message "hello" --output-format jsonl
caliche --message "hello" --output-format stream
```

### Agentic coding subcommand

```bash
caliche code "Add error handling to startup"
caliche code "Refactor auth flow" --cwd ./caliche_cli
caliche code "Fix lints" --accept-edits
caliche code "Show proposed edits only" --dry-run
caliche code "Run with streamed output" --stream
caliche code "Skip the repository overview for this run" --no-repomap
caliche code "Harden command ledger" --verify "cargo test --lib command_ledger"
caliche code "Plan only, skip checks" --dry-run --no-verify
```

`--no-repomap` turns off [RepoMap](../docs/concepts.md#other-names), the repository overview that `caliche code` adds to the context.

### Workflow file execution

```bash
# Execute a workflow file (JSON or YAML)
caliche --workflow ./workflow.json
caliche --workflow ./workflow.yaml
```

Workflow loading now accepts either `BuiltWorkflow` JSON or YAML and validates structure before execution.

## Top-Level Subcommands

```bash
caliche code ...
caliche agent ...
caliche config ...
caliche session ...
caliche doctor
caliche share --session <id> [--visibility public|authenticated]
caliche unshare --session <id>
caliche daemon install|start|stop|status|uninstall [--port 3000]
caliche server list|status|stop [--port N] [<id>]
caliche init ...
caliche lsp ...
caliche template ...
caliche prompt ...        # alias-style prompt/template operations
caliche test ...
caliche evaluate ...
caliche completions ...
caliche man ...
```

### `agent` (profile inspection)

```bash
caliche agent list
caliche agent show reviewer
```

### `session`

`caliche session resume <id>` reloads that session's saved transcript into the agent before the next prompt. When the transcript exceeds the resume budget, older turns are folded into a summary. The latest messages and the last tool actions stay in context. `/resume` in the REPL does the same reload. A thin-client turn keeps that transcript in the remote session, so the next prompt includes earlier turns and the CLI system prompt.

REPL autosave stores `metadata.reasoning_content` even when `--output-verbosity normal` hides thinking. Review it with `session export --include-thinking` or `/export --include-thinking`. Hiding the live stream keeps the trace on disk. Live service memory stays text only, so the next non-stream turn does not resend the trace. Resume and `/load` keep the trace on the saved transcript and strip it from the model replay so it stays inside the resume budget.

```bash
caliche session list --verbose
caliche session resume <id>
caliche session show <id>
caliche session rename <id> "new name"
caliche session export <id> --format markdown --output session.md
caliche session delete <id> --force
caliche session stats --days 7
```

### `lsp`

```bash
caliche lsp status
caliche lsp list
caliche lsp install rust
caliche lsp enable
caliche lsp disable
```

### `template` / `prompt`

```bash
caliche template list
caliche template show rag_agent:1.0.0
caliche template use rag_agent:1.0.0 --name my_agent
caliche template validate ./templates/my_template.yaml
caliche template init --output ./templates/new_template.yaml
caliche template test rag_agent:1.0.0 --test-inputs "hello,world"
caliche template benchmark rag_agent:1.0.0 --iterations 20

caliche prompt list
caliche prompt get rag_agent
caliche prompt test rag_agent --input '{"input":"hello"}'
```

### `test` / `evaluate`

```bash
caliche test run --template rag_agent
caliche test benchmark --template rag_agent --iterations 10
caliche test record --output replay.json
caliche test replay --input replay.json --compare

caliche evaluate --dataset test.jsonl --evaluator exact_match
caliche evaluate --dataset test.jsonl --evaluator regex --pattern "ok|pass"
caliche evaluate --dataset test.jsonl --evaluator llm --criteria "accuracy" --provider ollama
```

### `completions` / `man`

```bash
caliche completions bash > ~/.bash_completion.d/caliche
caliche completions zsh > ~/.zfunc/_caliche
caliche completions fish > ~/.config/fish/completions/caliche.fish
caliche completions powershell --output caliche.ps1

caliche man generate
caliche man generate --all --output ./man
caliche man install
```

## Interactive Slash Commands

Type `/help` in the REPL for quick help (including `/welcome`, `/sessions`, `/resume`, `/save`, `/load`, `/export`, and `/compact`), or `/help all` for the full list.

### Mode and session

- `/welcome`
- `/mode`, `/plan`, `/ask`, `/agent`, `/showplan` (`/show-plan` alias)
- `/undo` (`/revert` alias), `/redo`, `/checkpoint [name]`, `/rollback [id]`
- `/clear`, `/save [file]`, `/load [file]`
- `/sessions [search]`, `/resume [id|last]`, `/status`

### Context and conversation

- `/context`, `/track <file>`, `/add <glob>`, `/recent`, `/related <file>`
- `/changes`, `/diff [file]`, `/export [file]`
- `/compress`, `/autocompact <on|off|threshold>`
- `/memory [strategy]`
- `/remember <text>`, `/recall <query>`, `/knowledge`, `/forget <id>`

### Provider/model/tools

- `/providers`, `/provider [name]`, `/model [name]`, `/models [provider]`
- `/subagent-provider [name|clear]`, `/subagent-model [name|clear]`
- `/embeddings`, `/embedding [provider[/model]]`
- `/tools`

### Cost, utilities, and development

- `/cost`, `/budget`, `/stats`
- `/skill ...`, `/commands`
- `/init`, `/doctor`, `/lsp`
- `/pr`, `/commit`, `/review`
- `/compact`, `/verbose`, `/editor`, `/copy`
- `/exit`, `/quit`

### Credentials and gateway runtime (REPL)

- `/creds`, `/creds setup`, `/creds providers setup`
- `/creds set <provider> <key>`, `/creds import`, `/creds list`, `/creds link <dir>`
- `/gateway setup [--forward --provider cloudflare]`
- `/gateway start|stop|restart|status|list [--port N] [--dir PATH]`
- `/gateway forward start|stop|restart|status|list`
- `/gateway route list|pin|unpin`
- `/gateway twilio webhook sync|status`

## Context References (`@`)

Attach context inline in prompts:

```bash
Explain @src/main.rs
What changed? @diff
Show modified files @status
Summarize @https://example.com/article
Compare @src/lib.rs and @src/main.rs
Review @artifact:latest
Compare @artifact:before with @artifact:after
Use @resource:rag://docs/chunk-id while reviewing @src/lib.rs
```

Context tags can appear anywhere in a prompt, and multiple tags are resolved in
the order they appear. Generated artifacts use the active `/artifact-dir`
setting; MCP resources use `@resource:<uri>` or `@resource:<server>::<uri>` when
the same URI exists on more than one server.

## Configuration Paths

| What | Location | Description |
| ----- | -------- | ----------- |
| Keyring + vault DB | `~/.caliche/caliche.db` | Scoped credential metadata (secrets in OS keyring under `credentials:data`) |
| Agent profiles | `.caliche/agents/` (project) or `~/.config/caliche/agents/` | Markdown profiles with YAML frontmatter |
| Project config | `caliche.toml` or `.caliche/config.toml` | Optional project settings. Untrusted: only model, output and limit keys are read (the list is in `docs/configuration.md`); every other key, including `provider`, is ignored, a model applies only when the user config lists it under `providers.<provider>.models`, and `{env:}`/`{file:}` are not expanded. |

## Environment Variables

The CLI respects the same environment variables as the rest of Caliche:

- provider keys (for example `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`)
- model/provider defaults where applicable
- vector and RAG backend settings
- `RUST_LOG` for logging
- `CALICHE_KEYCHAIN_USE_TOUCH_ID` (macOS keychain behavior; see above)
- `JWT_ENABLED`, `CALICHE_REQUIRE_AUTH` when running `--server` with auth enabled

For the complete list and descriptions, see:

- root `README.md`
- `.env.example`

## License

Licensed under either of [Apache-2.0](../LICENSE-APACHE) or [MIT](../LICENSE-MIT), at your option.

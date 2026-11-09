# Configuration

Caliche reads its configuration from CLI flags, stored CLI settings, environment variables, and configuration files. Which sources apply depends on how you run it. Configure only the providers, backends, and gateway adapters you use.

This page covers credentials, file locations, the user and project configuration files, agent profiles, and the lean profile. Every environment variable is listed in [Environment variables](reference/environment-variables.md).

## Provider credentials

Run the interactive setup, or manage one provider at a time:

```bash
caliche config setup
caliche config set openai --key sk-...
caliche config get openai
caliche config remove openai
caliche config default openai
caliche config model openai gpt-4o
```

To read a credential from an environment variable instead of storing the key, name the variable:

```bash
caliche config set anthropic --env ANTHROPIC_API_KEY
```

To export the configured credentials to a shell file with restricted permissions, run the following command, then `source` the path that it prints:

```bash
caliche config export --shell bash
```

## Directory-scoped credentials (vault)

The `vault` commands manage credentials that apply to one directory:

```bash
caliche config vault status
caliche config vault list
caliche config vault import --dir .
caliche config vault link ~/other-project
caliche config vault promote openai
```

## Gateway credentials

The gateway adapters have their own credential commands:

```bash
caliche config gateway list
caliche config gateway set telegram_bot_token --key <token>
caliche config gateway set slack_signing_secret --env SLACK_SIGNING_SECRET
caliche config gateway get twilio_account_sid --unmask --yes-i-understand-this-prints-secrets
caliche config gateway remove teams_webhook_url
```

The supported credential names are:

- `telegram_bot_token`
- `discord_bot_token`
- `twilio_account_sid`
- `twilio_auth_token`
- `twilio_from`
- `slack_bot_token`
- `slack_signing_secret`
- `teams_webhook_url`

## File locations

`CALICHE_CONFIG_DIR` replaces the `caliche` configuration directory for the items marked below. The items not marked keep their own locations when the variable is set.

| Item | Location | Follows `CALICHE_CONFIG_DIR` |
| --- | --- | --- |
| Credentials | The OS keyring (service `caliche`, entry `credentials:data`), plus `~/.caliche/caliche.db` | No |
| Agent profiles | `.caliche/agents/` or `~/.config/caliche/agents/` | No |
| Project configuration | `caliche.toml` or `.caliche/config.toml`. This file is untrusted; see [Project configuration](#project-configuration). | No |
| Server lock | `~/.config/caliche/server.lock`. Written by `--server`; the REPL uses it to find and attach to a running server. | Yes |
| Server registry | `~/.config/caliche/servers/*.json`, one file per `--server` process, listed by `caliche server list`. | Yes, as `servers/` beside `server.lock` |
| API plugin store | `dirs::data_dir()/caliche/plugins.db`, or `./caliche_plugins.db` as a fallback. | Yes, as `plugins.db` in that directory |
| API MCP configuration | `dirs::config_dir()/caliche/mcp.json`, or `./.caliche_mcp.json` as a fallback. | Yes, as `mcp.json` in that directory |
| Durable settings | `settings.json` in the same directory as `mcp.json`. It stores the provider, model, temperature, and maximum tokens, but not the system prompt. A CLI flag takes precedence over this file, and this file takes precedence over environment variables. | Yes |
| CLI logs | `dirs::data_dir()/caliche/logs` (`%APPDATA%\caliche\logs` on Windows) | No |
| Endpoint store | `dirs::data_dir()/caliche/endpoints.json` | No |
| Gateway cluster database | `~/.caliche/gateway-cluster.db`. Gateway logs are in `~/.caliche/logs/gateway`. | No |
| CLI session database | `~/.caliche/caliche.db`, resolved from `HOME` | No |
| CLI task ledger | `~/.caliche/tasks.db`, beside the session database. `caliche --server` uses this path. | No for `caliche --server`. The standalone API binary stores `tasks.db` in `CALICHE_CONFIG_DIR` when the variable is set. |

## Project configuration

The user configuration file, `~/.config/caliche/config.toml` (or `config.toml` in `CALICHE_CONFIG_DIR` when that variable is set), accepts every key. A project configuration file (`caliche.toml` or `.caliche/config.toml`) comes with the repository you are working in, so Caliche treats it as untrusted, and it treats remote organization configuration the same way:

- Only these keys are read from it: `model`, `small_model`, `temperature`, `max_tokens`, `context_window`, `streaming`, `verbose`, `output_verbosity`, `banner`, `memory_strategy`, `memory_max_tokens`, `tool_output_threshold`, and `tool_execution_driver`. Every other key is ignored with a warning. The ignored keys include `provider`, `[providers.*]`, `[middleware.*]`, `system_prompt`, `tools`, the file creation guardrails (`max_new_files`, `warn_file_types`, and `confirm_doc_creation`), `auto_skill`, and any key this version does not recognize. A key added in a later version is ignored until it is added to this list.
- `model` and `small_model` apply only when the user configuration lists that model under the user's provider, in `providers.<provider>.models`. Otherwise they are ignored with a warning. As a result, a repository cannot choose the provider, and it cannot choose a model for a local runtime to download and run.
- A value of the wrong type is ignored with a warning, and the rest of the file still applies. A file that is not valid TOML is ignored entirely.
- MCP server discovery is separate from this list. It still reads `[mcp.servers]` from a project `caliche.toml`, in the same way that it reads `.mcp.json`. A stdio MCP server declared in the repository must pass `CALICHE_MCP_COMMAND_ALLOWLIST` and cannot set `env`. An HTTP MCP server declared in the repository connects under the egress policy.
- `{env:NAME}` and `{file:path}` references are not expanded in an untrusted file. Put values such as the `api_key` in the example below in the user configuration.
- "Always allow" approvals are saved to the user configuration.

The following example is a user configuration file:

```toml
provider = "openai"
model = "gpt-4o"
small_model = "gpt-4o-mini"   # used for titles and lightweight scout turns

[providers.openai]
models = ["gpt-4o", "gpt-4o-mini"]   # models that a project or remote configuration may choose
auth_profiles = [
  { name = "primary", api_key = "{env:OPENAI_API_KEY}" },
  { name = "secondary", env_var = "OPENAI_API_KEY_2" },
]
rotation = "round_robin"

[middleware.pre_completion_checklist]
# When `enabled` is omitted it defaults to true, and other values use the built-in defaults.
# enabled = false

[middleware.loop_detection]
# enabled = false

[middleware.environment_onboarding]
# enabled = false
```

The harness middleware is on by default in the CLI. To turn off a middleware, set `enabled = false` in its section.

## Local model first token

LM Studio and Ollama may load the model and read the whole prompt before they send the first token. With a large prompt, this can take minutes. For these two providers, Caliche waits up to `local_first_token_timeout_secs` for the first token. The default is 600 seconds, and the value is clamped to 10 through 3600. Set it in the user configuration:

```toml
local_first_token_timeout_secs = 900
```

When the key is not set, `CALICHE_LOCAL_FIRST_TOKEN_SECS` gives the value. While Caliche waits, the REPL shows `Waiting for the first token...`. Ctrl+C cancels the wait.

When LM Studio reports that the selected model is not loaded (with just-in-time loading off, it answers `No models loaded`), Caliche asks LM Studio to load it (`POST /api/v1/models/load`, LM Studio 0.4 and later), shows `Loading <model> in LM Studio...`, and sends the turn once the model is loaded. It sends one load request, and none when a load is already under way. The same limit bounds the load. Ctrl+C cancels it and the turn. A load that fails or does not finish in time fails the turn with one line that names the model and quotes LM Studio's reason. The embedding model is loaded the same way before the workspace is indexed; only when that load fails does workspace search turn off for the session. After the first token, a gap of more than 60 seconds between chunks fails the turn as a stall. Cloud providers use the 60-second limit for the first token too.

## Project context budget

The REPL and `caliche code` put the project file (for example `AGENTS.md`) in the system prompt. When the model's context window is known, the file gets at most `project_context_share` percent of it. The default is 25, and the value is clamped to 1 through 100. Set the share in the user configuration:

```toml
project_context_share = 40
```

The window comes from, in order:

1. an explicit window: `--context-window`, `context_window` in the configuration, `CALICHE_CONTEXT_WINDOW`, or `/context-window`;
2. the window the provider reports for the loaded model (LM Studio and Ollama report it);
3. none: the whole file is used.

A larger file is cut at a line end. A last line names the file, says how many of its lines are shown, and gives its path, so the agent can read the rest with its file tool. A model with a large window gets the whole file when it fits.

A `/model` or `/provider` switch, or `/context-window`, sizes the file again for the next turn: a smaller window cuts it and a larger one restores it. When the new model's window is not known, the file stays as it is. In the REPL, `/context-window` also reaches the embedded server on the next turn, without a model switch.

Compaction and the context meter (`/context`, `/context-window`) use the same window, except in step 3: when no window is explicit or reported, they use the provider's fixed default (32768 tokens for LM Studio and Ollama).

## Workspace search and embeddings

At startup, the REPL sends one embedding request to check the embedding provider before it indexes the workspace. When the provider answers with an error that a retry cannot fix, such as `400` or `404` for an embedding model that is not on disk, the REPL prints one line and goes on without workspace search:

```text
Workspace search is off for this session: <the provider's error>. Load or download the embedding model, or start with --no-workspace-embed.
```

Nothing more is sent to the embedding provider in that session, unless you run `/reindex` or choose a provider with `/embedding`. The same happens when such an error comes later, while the workspace is being indexed. Errors that a retry can fix (timeouts, `408`, `429` and `5xx`) do not turn off workspace search.

## Runtime identity and caching

`AgentClient` reads `server.lock` to find a running local API process.

`ResolvedRuntimeIdentity`, `resolve_runtime_identity`, and `resolve_runtime_identity_with_discovery` in `agent::service::credentials` resolve the provider identity that the API and the CLI start with.

Provider prompt caching uses the Anthropic `cache_control` field and the OpenAI `prompt_cache_key` field, and usage reports `cache_read_tokens` and `cache_creation_tokens`. The local model response cache matches requests exactly. JWKS responses are cached in the process. See [Logging and observability](operations/logging-observability.md).

On macOS, Caliche stores credentials in the data protection keychain with Touch ID by default. Set `CALICHE_KEYCHAIN_USE_TOUCH_ID=false` to use the regular keychain instead.

## Agent profiles

An agent profile is a Markdown file with YAML front matter that defines a reusable agent. Profiles are read from `.caliche/agents/` or `~/.config/caliche/agents/`.

```bash
caliche agent list
caliche agent show reviewer
caliche --agent reviewer "review this change"
```

In Rust, build an agent from a profile with `AgentBuilder::from_profile(&profile)?` when the `profiles` feature is enabled. An unknown `permission.edit` mode is an error. An empty value and `default` both select the default mode.

## Lean profile

`--profile lean` is an opt-in runtime profile for the default agent with a small tool set and a short prompt. It does not change the normal defaults. Select it on the CLI, including with `--server`:

```bash
caliche --profile lean --server --bind 127.0.0.1
caliche --profile lean -p "Reply with the single word pong."
```

The lean profile limits the tools to `fs` and `bash`, unless `--tools` names other tools; the normal coding default also includes `web_fetch`. For the default agent (the REPL, a single prompt, and `--server`), the profile uses a short system prompt and skips MCP auto-discovery, the on-disk skill index, workspace embedding, the research skill, the plugin store, and `AGENTS.md` injection. To keep workspace retrieval, pass `--workspace-embed` together with `--profile lean`. `caliche code` still uses the coding prompt, and `coding_system_prompt` still loads `AGENTS.md`; `--profile lean` does not remove that.

When you pass neither `--tools` nor `--agent`, the REPL tools line shows `from --profile lean`.


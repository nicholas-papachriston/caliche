# Security policy

## Supported versions

Only the latest commit on `main` is supported. Fixes are not backported to older commits.

## Reporting a vulnerability

Report suspected vulnerabilities privately, by email to `nicholas@papachriston.com`. Do not open a public issue, discussion, or pull request about a vulnerability, an exposed credential, or a way to bypass a security control. Private vulnerability reporting is not enabled for this repository, so email is the only reporting channel.

### Report template

Redact any real secrets, credentials, tokens, or personal data before you send a report. Test only against your own installation, and do not run exploits against systems or accounts that you do not own.

```text
Subject: [caliche security] <short title>

Affected version or commit:
  (a `main` commit SHA, or the output of `caliche doctor`, which prints `build_id`)

Component:
  (for example the CLI, the HTTP API, a tool such as `bash` or `fs`, the sandbox, the policy crate, or a submodule)

Operating system:

Impact:
  (what an attacker can do, and what access or configuration they need)

Steps to reproduce:
  1.
  2.
  3.

Proof of concept:
  (minimal commands, input, or code; attach redacted files if needed)

Suggested fix:
  (optional)
```

Reports are acknowledged within 7 days. The goal is to fix the issue and publish a coordinated disclosure within 90 days.

## Risk model

Caliche runs shell commands, file system operations, and network requests on behalf of a language model. Treat every tool call as code that the model chose to run with your privileges.

- **File system:** tools such as `fs`, `write_file`, `edit_file`, `apply_patch`, and `bash` can create, change, or delete files. A tool that does not run inside the sandbox described below runs as the user who started Caliche, and can reach everything that user can reach.
- **Credentials:** hosts often hold `.env` files, API keys, SSH keys, and cloud credentials. A tool that can read them can leak them into conversation logs, the model context, or remote endpoints.
- **Prompt injection:** content from fetched web pages, repository files, MCP servers, or subagents can try to make the model run commands or read files that the operator did not intend.
- **Network requests and server-side request forgery:** plugin HTTP steps, gateway webhooks, scheduled delivery, and the HTTP tool check each destination in [src/security/http_policy.rs](src/security/http_policy.rs), and block loopback, private, link-local, and metadata addresses. These checks have limits:
  - When `PLUGIN_HTTP_ALLOWLIST` is unset (outside the production profile) or is only `*`, any public host is allowed.
  - MCP URLs and configured model provider endpoints may use loopback and private addresses, so that a local MCP server or a local model server such as Ollama or LM Studio can be used ([src/gateway/egress.rs](src/gateway/egress.rs)). Provider plugins with a loopback base URL are still rejected ([src/plugins/validation.rs](src/plugins/validation.rs)).
  - Commands run through `bash` are not checked at all.
- **HTTP server:** the server has the defaults described in [HTTP server defaults](#http-server-defaults).

## HTTP server defaults

Other pages link here instead of repeating these defaults.

- **Authentication:** JWT authentication is off by default. Unless both `JWT_ENABLED` and `CALICHE_REQUIRE_AUTH` are set, the API does not authenticate requests.
- **Listen address:** the server listens on `127.0.0.1:3000` by default. `API_HOST` does not set that address. An explicit open address such as `--bind 0.0.0.0` only logs a warning (`open_bind_warn`), and the server still starts.
- **Session list:** one server process has one list of sessions, which every caller shares. `GET /api/agent/sessions` returns every live session in the process, and the JWT `sub` claim does not filter it. With authentication off, any caller who can reach the server can list, cancel, and delete every session. See [HTTP sessions](docs/sessions/sessions.md).
- **Web UI:** when the UI is built in, the same address also serves it, and a bearer token pasted into the UI is an operator credential.

For a deployment that is reachable from the internet, follow the [production authentication checklist](docs/ops/production-auth-checklist.md): set `CALICHE_ENV=production`, `JWT_ENABLED=true` with a public key or a JWKS URL, `JWT_AUDIENCE`, `JWT_ISSUER`, and `CALICHE_REQUIRE_AUTH=true`; leave `CALICHE_ALLOW_INSECURE_PRODUCTION` unset; and set `PLUGIN_HTTP_ALLOWLIST`.

## Sandbox

The `caliche_sandbox` crate provides the OS sandbox. It applies only when a sandbox handle is set for the session. The execution engine sends only the tools that are marked for sandbox execution, `fs`, `bash`, and `execute_python`, to the sandbox ([src/agent/tools/tool.rs](src/agent/tools/tool.rs), [src/agent/tools/execution/engine.rs](src/agent/tools/execution/engine.rs)), and all three run inside it.

- `execute_python` refuses to run when no sandbox handle is active, or when the backend does not confine files (`noop` and the Windows Job object backend). It never runs without confinement.
- `bash` still runs under those backends.
- `write_file` has a sandboxed implementation, but the engine does not send it to the sandbox.
- `edit_file` and `apply_patch` have no sandboxed implementation.
- Every other tool runs as the user.

When the sandbox applies and the backend can enforce it, the `workspace-write` policy confines writes to the workspace. Windows Job objects and `noop` do not confine the file system. The sandbox does not protect files inside the workspace, which can still be changed or deleted, and it does not detect prompt injection or malicious intent in model output.

[crates/caliche_policy/src/capability.rs](crates/caliche_policy/src/capability.rs) describes what each kind of host can enforce: Docker, macOS Seatbelt, Linux Landlock, Docker or WSL2 on Windows, Windows Job objects only, and `noop`. The policy crate uses this description; it is not a list of sandbox backends. According to it, Windows Job objects and `noop` cannot enforce outbound network rules, and Linux Landlock can enforce them only when the kernel supports network rules.

The `noop` backend provides no isolation. The production profile refuses `noop` and the `danger` preset when a policy is compiled ([posture.rs](crates/caliche_policy/src/posture.rs)); this is not a check that runs on each tool call.

## Policy engine

The [caliche_policy](crates/caliche_policy/README.md) crate defines a typed policy engine. Before any hook runs, each tool call goes through `evaluate_tool_gate` (`src/agent/tools/execution/middleware.rs`), which applies the `trusted` preset together with the session's overlay. A deny from the floor layer always blocks the call. `CALICHE_POLICY_SHADOW=1` also logs a comparison with the `standard` preset, without changing the gate result.

### Layers and effects

A policy is built from five layers, defined by `Layer` in [posture.rs](crates/caliche_policy/src/posture.rs):

- `Floor`: rules compiled into the binary.
- `Deployment`: the operator's policy.
- `Profile`: the agent profile.
- `Project`: the repository's `caliche.toml`.
- `Session`: grants and mode overlays made during the session.

Each decision has one of three effects, `Allow`, `Ask`, or `Deny`, and the strictest effect wins. The profile, project, and session layers can narrow what the deployment allows, but they cannot override a deny from the deployment.

The floor rules deny a call when the policy is missing or invalid or when a checker fails, change `Ask` to `Deny` when no person is available to approve, and deny access to vendor subscription credentials.

### Presets

A preset sets the default effect and the sandbox requirement of the deployment layer. [posture.rs](crates/caliche_policy/src/posture.rs) defines five:

| Preset | Default effect | Sandbox |
| --- | --- | --- |
| `locked` | `Deny` | Strict sandbox required |
| `review` | `Deny` | Strict sandbox required |
| `standard` | `Ask` | `workspace-write` |
| `trusted` | `Allow` | `workspace-write` |
| `danger` | `Allow` | None. The production profile refuses this preset when the policy is compiled. |

A strict sandbox allows writes only inside the workspace, denies outbound network access, and stops child processes when the sandbox is dropped.

The shell command analyzers are off in the `trusted` preset, which the tool gate uses. They run for `standard`, `locked`, and `review`; see [caliche_policy](crates/caliche_policy/README.md). An analyzer is a warning, not a boundary: when it is on, it raises a prompt; accept-all removes that prompt; and a command can be written so that the analyzer does not detect it.

### Accept-all

The `AcceptAll` session overlay changes every `Ask` decision that is not marked as hard to `Allow`. It never changes a `Deny`. The CLI `--yolo` option approves all tool permissions automatically. `yolo` is not accepted as a permission mode string.

Any file system access outside the workspace (a read, list, search, write, delete or patch) is a hard `Ask` ([policy.rs](crates/caliche_policy/src/policy.rs)). Accept-all, accept-edits and a session grant cannot lower it, so only a person can approve it, and when no approver is available the access is denied. Text in the user's message does not grant access to a path. The CLI REPL also registers `PathRestrictionHook` ([src/agent/hooks/path_restriction.rs](src/agent/hooks/path_restriction.rs)), which denies rejected path forms for the file tools: alternate data streams, device paths and 8.3 short names. Other entry points, such as the HTTP API, do not register this hook. Accept-all still approves every other ask, such as shell commands and network requests, so it is not safe on a machine you care about. Use accept-all only in a disposable or otherwise isolated environment.

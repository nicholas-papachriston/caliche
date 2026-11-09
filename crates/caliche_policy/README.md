# caliche_policy

`caliche_policy` is the typed policy engine for Caliche. It decides whether a tool call is allowed, needs a person's approval, or is denied. It does not depend on the root `caliche` crate.

The crate contains the policy types, the rules compiled into the binary (the floor), strict parsing, a shadow comparison, and the live tool gate. Before any hook runs, the engine calls `evaluate_tool_gate` for each tool call. The gate applies the `trusted` preset together with the session overlay, so a deny from the floor always blocks the call. In the interactive mode, the gate asks before any call with a side effect, including an HTTP `GET` and an MCP tool call. Hooks still run after an allow and after an approved ask.

## Effects and layers

Each decision has one of three effects: `Allow`, `Ask`, or `Deny`. When two rules apply, the stricter effect wins.

A policy is built from five layers, defined by `Layer` in [posture.rs](src/posture.rs):

| Layer | Source | What it can do |
| --- | --- | --- |
| `Floor` | Rules compiled into the binary | Deny, or make an ask hard. A plugin cannot replace it. |
| `Deployment` | The operator's `policy.toml` | Allow, ask, or deny, and set the preset. |
| `Profile` | The agent profile | Narrow what the deployment allows. |
| `Project` | The repository's `caliche.toml` | Narrow what the deployment allows, unless the deployment lets it widen. |
| `Session` | Grants and mode overlays made during the session | Narrow, or approve an ask that is not hard. |

Only `CompiledPolicy` can decide. The `Decide` trait is sealed, so no other crate can implement it. A `PolicySource` can add rules at the deployment layer or below. It cannot add floor rules.

## Floor

The floor rules apply to every policy:

- A missing or unknown policy denies.
- A checker error denies, including under accept-all.
- An ask denies when no person is available to answer it.
- Accept-all cannot lift a deny from the deployment.
- Only a person can answer an approval.
- Vendor auth files and vendor subscription credentials are denied. Use your own provider keys.
- Network access to cloud metadata, loopback, link-local, private, and carrier-grade NAT addresses is denied.

## Presets

A preset sets the default effect and the sandbox requirement of the deployment layer.

| Preset | Default effect | Sandbox | Intended use |
| --- | --- | --- | --- |
| `locked` | `Deny` | Required | New deployments |
| `review` | `Deny` | Required | Planning and review |
| `standard` | `Ask` | `workspace-write` | A workstation |
| `trusted` | `Allow` | `workspace-write` | The live tool gate. The shell analyzers are off. |
| `danger` | `Allow` | None | Refused when the production profile is on |

The live tool gate uses `trusted`, so that the default overlay does not ask about every call. The shadow comparison uses `standard`.

## File paths

`resolve_fs_path` resolves absolute paths, relative paths, and `..` against the working directory, and replaces a symlink that exists with its real path. Paths in patch headers are resolved too. A path outside the workspace is a hard ask. Alternate data streams, device paths, and 8.3 short names are denied. Text in the user's message does not grant access to a path.

## Approval broker

`ApprovalBroker` gives each ask an id that starts with `apr_`, and the approval card shows the tool arguments. Only a person can answer, and the first answer wins. An expired ask is removed. `release` checks that the answer still matches the action, and drops the grant when it does not.

The crate does not prompt on a terminal, draw an approval card, or serve an HTTP route. The engine's tool middleware owns the broker.

## Shell analyzers

The POSIX, PowerShell, and cmd analyzers run only when the preset turns on that shell's facet, which `standard`, `locked`, and `review` do. `trusted` leaves the facets off, so `analyze_shell` does not call the analyzers.

- The POSIX analyzer refuses a recursive `rm` and a download piped to a shell.
- The PowerShell analyzer refuses a recursive `Remove-Item`, `iex` of a web request, an encoded command that matches the denylist, and `Start-Process rm`.
- The cmd analyzer refuses `del /s` and `rd /s`, including inside `cmd /c`.

A command that an analyzer cannot read is never allowed.

## Windows

Windows has no kernel sandbox that Caliche can use. An unattended shell on Windows needs Docker or WSL2 (`HostCapability::WindowsDockerOrWsl2`). `WindowsJobOnly` does not confine files or network access and does not support an unattended shell. `policy_show` prints `network not sandboxed` and the shell effect, which is `Ask` under `standard` and `Deny` under `locked`.

When the session names a host that cannot meet the sandbox requirement, `decide` asks before shell and code execution under `trusted` and denies them under `locked`. An unnamed host does not change the effect. `decide` does not start a sandbox process.

## policy.toml

A version 1 file names a preset (`locked`, `review`, `standard`, `trusted`, or `danger`), an optional `default` effect, an optional `[sandbox] require`, and `[[tools]]` rules. A tool rule sets `class` or `name`, and `effect`. An unknown key is a load error. A missing file is read as the `locked` view for a new deployment, and it is not written until you save it.

```toml
version = 1
preset = "standard"

[[tools]]
class = "http"
effect = "deny"
```

`caliche policy show` and the Permissions page of the web UI show this file, together with what the host can confine. On Linux, that panel uses the Landlock network probe from the `landlock-net` feature. The live tool gate stays on `trusted`.

## Shadow mode

Set `CALICHE_POLICY_SHADOW=1` to evaluate this crate beside the hook path and log the calls where the effects differ. `true`, `yes`, and `on` also turn it on. The log does not contain the tool arguments, and the result of the call does not change. The shadow evaluation uses the `standard` preset and assumes that a person is available, so the comparison shows the effect of the floor and the preset.

## Strict parsing

`parse_mode_overlay`, `parse_preset`, and `parse_effect` return an error for an unknown value. They do not fall back to a default. An empty value and `default` both mean the default overlay. The permission modes are `default`, `accept-edits`, `accept-all`, `deny-all`, and `interactive`.

## License

Licensed under either of [Apache-2.0](../../LICENSE-APACHE) or [MIT](../../LICENSE-MIT), at your option.

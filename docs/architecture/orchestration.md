# Orchestration

Caliche provides four ways to make more than one agent work together. Each one serves a
different purpose, and they are not interchangeable. This page explains when to use each one
and which safety controls apply to it.

| Layer | Type | Control flow | Use it when |
|---|---|---|---|
| `AgentCoordinator` | Agent pool | The caller dispatches one task at a time | You need capability-based selection, load balancing, or supervision |
| `AgentFlow` | Handoff chain | Each agent decides the next agent with `transfer_to` | The path depends on what an agent finds |
| `WorkflowEngine` | Graph | The graph decides the path | The path is known: sequence, branch, fan-out, or loop |
| Agent profiles | Definition | None | You declare a reusable agent: model, tools, persona, and task permissions |

Shotput templates are not an orchestration layer. Shotput assembles the prompt text of one
agent from files, rules, and other templates. Use it to build a system prompt; use the layers
above to make agents cooperate.

## Coordinator

`AgentCoordinator` holds a pool of registered agents and dispatches one task to one agent.
Register an agent with `register_agent`, or with `register_agent_with_privileges` when the
agent must have a tool ceiling and a task ACL. `register_discovered_subagent` registers a
profile that discovery found, with the privileges its front matter declares. External
harnesses (Pi, Hermes, or any `HarnessRuntime`) register with `register_harness`,
`register_pi`, or `register_hermes`.

Dispatch entry points:

- `execute_on_agent` names the target and applies no caller policy.
- `execute_on_agent_as` names the caller and the target. The caller's ACL decides whether
  the target may run, requested tools are intersected with both ceilings, and the result is
  quarantined before it returns.
- `delegate_task` and `delegate_task_as` select the target by capability and load, then
  follow the same path.

Every dispatch runs under `orchestration.dispatch_timeout_secs` when it is set, so a hung
provider call releases the pool slot instead of holding it. `with_supervision` adds OTP-style
restart policy, modeled on Erlang/OTP supervisors, for agents that run as long-lived tasks; `supervisor_metrics` reports the
restart counts and the current window.

## Flow

`AgentFlow` runs a chain of handoffs. An agent calls the `transfer_to` tool, and the flow
moves to the named agent. Build one with `FlowBuilder`; `with_orchestration_config` applies
the configured hop limit, ACL, and quarantine settings in one call.

The flow refuses a handoff to an agent that is not part of the flow, applies the source
node's task ACL, intersects the requested tools with the source and target ceilings, and
stops the run at `orchestration.max_flow_hops` hops. A handoff message that a parent agent
will read is quarantined first.

## Workflow

`WorkflowEngine` runs a graph of nodes: agent, condition, transform, parallel, merge, loop,
and human interrupt. A graph with a cycle must run through `execute_state_machine` with an
entry point; an acyclic graph can run through `execute_dag`.

Controls the engine applies:

- Parallel branches run as separate tasks behind a concurrency semaphore
  (`orchestration.pattern_concurrency`). A `FirstComplete` merge takes the first branch that
  reaches the merge node and cancels the rest.
- `NodeErrorPolicy::Retry` retries a failed agent node with exponential backoff, then halts
  the run. `retry_attempts` and `retry_backoff_ms` in node metadata override the defaults.
- An agent node with `idempotent: true` or an explicit `idempotency_key` in its metadata
  records its output, so a repeat of the same node and prompt returns the recorded result
  instead of calling the model again.
- `orchestration.pattern_max_nodes` and `orchestration.pattern_max_tokens` bound one run. The
  engine refuses to start another agent node once a ceiling is reached, and records the
  refusal as a node error. Token accounting depends on the executor reporting `total_tokens`
  in node metadata.

Patterns in `src/orchestration/patterns/` (pipeline, fan-out/collect, debate, hierarchical,
supervisor) build these graphs for you.

## Profiles

An agent profile is a Markdown file with YAML front matter. Discovery reads profiles from
`.caliche/agents/`, `.agents/`, and the equivalent directories under the home directory.
A profile sets the model, temperature, tools, persona body, permissions, and the task
permissions described below.

```yaml
---
description: Reviews code and reports defects
mode: subagent
model: anthropic/claude-sonnet-4-5
tools: [fs, semantic_search]
task_permissions:
  "*": deny
  reviewer: allow
---
You review code. Report defects with file and line references.
```

## Safety controls

Four controls apply to delegation. They are independent, so a gap in one does not open the
others.

**Task invocation ACL.** An agent declares which other agents it may invoke as glob pattern
to `allow` or `deny`. The most specific pattern wins, so `{"*": "deny", "reviewer": "allow"}`
permits one target and refuses the rest. `ask` counts as `deny`, because a dispatch path has
nobody to ask. An agent with no ACL may invoke every target it can reach: existing
deployments keep working until they configure `orchestration.task_acl` or a profile
`task_permissions` map. The ACL is enforced in the coordinator, the flow, the `transfer_to`
tool, and the REPL `@mention` path.

**Privilege intersection.** A handoff may name the tools the target agent should use. The
request is intersected with the caller's ceiling and the target's ceiling, so a delegation can
narrow privileges and never widen them. Dropped tools are logged. Intersection is always on;
there is no configuration to turn it off.

**Registered targets.** `transfer_to` refuses a target that is not registered with the
coordinator or the flow, so a model cannot invent an agent name. Set
`orchestration.require_registered_targets = false` only for a runtime that resolves agents
some other way.

**Output quarantine.** Text that a subagent or a tool produced is untrusted input to the
parent agent. Before the parent reads it, the text is wrapped in a labeled block that names
the source and states that instructions inside it must not be followed. Markers inside the
body are neutralized, and a body that is already quarantined is not wrapped twice.

`SharedState` has its own control: writes attributed to an agent (`set_as`, `merge_as`) go
through a `StateWritePolicy` of path globs per agent, and `set_in_namespace` keeps an agent
inside its own branch. Unattributed writes stay unrestricted so existing callers are
unaffected.

## Complex tasks and subagent models

Long-running work uses `run_complex_task`. The driver records findings each round, refuses
an early stop, and verifies before it reports complete. CLI `--complex` and API
`POST /api/orchestration/coordinator` with `complex: true` turn this on.

The provider and model used for subagents form a separate setting (`SubagentModelSelection`).
You can set it in any of these places:

- The REPL commands `/subagent-provider` and `/subagent-model`, which persist the choice in
  the CLI settings database.
- The flags `--subagent-provider` and `--subagent-model`, which write the same settings.
- The `[subagent]` section of the configuration file, with `provider` and `model` keys.
- The `model: provider/id` field in the front matter of an agent profile.

REPL `@mention` and `sessions_spawn` both apply that selection. `HarnessSessionSpawner` is
the CLI/API default spawner, so parent agents can start child sessions without a custom
host hook.

## External harnesses

The coordinator dispatches through `HarnessRuntime`, not through `Agent` directly.
Caliche's own runtime is one implementation (`LocalAgentRuntime`). Pi, Hermes, and any other
CLI that accepts a prompt as argv (never as `sh -c`) are another.

```rust
use caliche::orchestration::{
    AgentCoordinator, AgentMetadata, AgentPrivileges, ExternalCliRuntime, ExternalCliSpec,
};
use std::sync::Arc;

let coordinator = AgentCoordinator::new();
let pi = Arc::new(ExternalCliRuntime::pi()?);
coordinator.register_harness_with_privileges(
    AgentMetadata::new("pi", "coder", vec![]),
    pi,
    AgentPrivileges::unrestricted(),
)?;
```

`register_pi` and `register_hermes` are presets for `pi -p <prompt>` and
`hermes chat -Q -q <prompt>`. The prompt is a separate argv element. Shells such as `sh`, `bash`,
`cmd`, and `pwsh` are refused as the executable. Production requires
`CALICHE_HARNESS_COMMAND_ALLOWLIST`; a basename of `pi` does not authorize `/tmp/pi`.

ACL, privilege intersection, registered-target checks, output quarantine, and
`dispatch_timeout_secs` apply to external harnesses the same way they apply to local
agents. Tool ceilings for CLI adapters are advisory prompt text unless the adapter maps
them onto that harness's own flags.

Hosts that want Caliche as a pure orchestration layer can depend on
`caliche_orchestration` without the engine crate, implement `HarnessRuntime`, and keep
using the same ACL, privilege, and quarantine primitives.

## Configuration

```toml
[orchestration]
require_registered_targets = true
quarantine_subagent_output = true
max_flow_hops = 50
dispatch_timeout_secs = 300
pattern_concurrency = 8
# pattern_max_nodes = 64
# pattern_max_tokens = 500000
node_retry_attempts = 2
node_retry_backoff_ms = 250

[orchestration.task_acl]
"*" = "deny"
reviewer = "allow"

[orchestration.supervision]
strategy = "one_for_one"
max_restarts = 3
restart_window_secs = 60
backoff = "exponential"
backoff_base_ms = 500
```

The HTTP orchestration routes are default-deny. See
[environment variables](../reference/environment-variables.md) for
`CALICHE_ALLOW_UNAUTH_ORCHESTRATION`.

## Choosing a layer

- One task, one specialist: coordinator.
- The next agent depends on what this agent finds: flow.
- The path is known in advance, or you need parallel work with a merge: workflow.
- You want the same agent definition in the CLI, the API, and a workflow: a profile, used by
  any of the three layers.

# Concepts

This page defines the terms used throughout the Caliche documentation. Each term is defined here once, and other pages link to these definitions.

## Agent

An agent sends a prompt to a model, runs the tools that the model requests, and returns the result. In Rust, you build an agent with `AgentBuilder`, and the resulting value is an `Agent`.

The CLI, the HTTP API, and the embedded web UI all run the same kind of agent. They differ only in how a prompt arrives and how the reply is returned.

## Session

A session is one conversation. It holds the messages, tool results, and token counts that later turns build on. Sessions served over HTTP live in the memory of the server process. See [HTTP sessions](sessions/sessions.md).

## Turn

A turn is one prompt, together with the model calls and tool calls that answer it. `POST /api/agent/sessions/{id}/turn` runs one turn. A task runs on the same session as a turn, but the request carries a task body instead of a turn body.

## Task

A task asks the agent to finish a piece of work, which can take several turns. `POST /api/agent/sessions/{id}/task` runs one task and reports the number of turns, the files changed, and the checkpoint it recorded.

## Tool

A tool is a function that the model can ask the agent to run, such as `fs`, `bash`, or `web_fetch`. The default tool set, and the rule that blocks a repeated identical failure, are described in [Tools](tools.md).

## Workspace

The workspace is the directory that the agent works in, normally the directory where you start Caliche. File tools resolve relative paths against it, the sandbox allows writes inside it, and policy asks a person before any file access outside it. `CALICHE_WORKSPACE` and `--workspace` set it for the standalone server.

## Hook

A hook is code that runs at a fixed point in the agent loop, such as before or after a tool call or at the start of a turn. A hook can record an event or deny a tool call. Hooks run after the policy gate. See [Runtime lifecycle](architecture/runtime-lifecycle.md).

## Skill

A skill is a reusable package of instructions for an agent: text that is added to the system prompt, the tools it needs, hooks, and settings. `caliche skills install` copies a skill from a Git repository or a local directory.

## Provider

A provider is the model API that the agent calls, such as OpenAI or Anthropic. You select a provider with a CLI flag, with `settings.json`, or with an environment variable. The order in which the server chooses a provider at startup is in [HTTP sessions](sessions/sessions.md#which-provider-a-turn-uses). The environment variables for provider keys are in [Environment variables](reference/environment-variables.md).

## Agent service

`AgentService` wraps an `Agent` and adds settings, a tool registry, and session storage. The CLI and the HTTP server both use it.

## Harness

`AgentHarness` wraps `AgentService` for one interactive session. It adds an event stream, checkpoints, and undo for file edits.

## Checkpoint

A checkpoint is a saved state of the files that a harness session has changed. The harness records one for each task by default (`auto_checkpoint`), and you can also record one by hand. Rolling back to a checkpoint restores those files. A suspended session keeps the identity of its last checkpoint.

## Runtime

`AgentRuntime` holds many harness sessions in one process. The HTTP API registers its live sessions with the runtime. When a host uses `AgentRuntime`, session creation, turns, suspend, and resume all go through it.

## Agent profile

An agent profile is a Markdown file with YAML front matter that names an agent and sets its model, tools, prompt, and permissions. `caliche agent` lists the profiles that it finds. See [Profiles](architecture/orchestration.md#profiles).

## Subagent

A subagent is an agent that another agent starts to do part of its work. Subagents can use a different provider and model from the parent agent. See [Complex tasks and subagent models](architecture/orchestration.md#complex-tasks-and-subagent-models).

## Workflow and handoff

A workflow is a graph of agent steps, conditions, and merges that `WorkflowEngine` runs. A handoff passes the conversation from one agent to another: in an `AgentFlow`, the current agent calls the `transfer_to` tool to name the next agent. See [Orchestration](architecture/orchestration.md).

## Gateway

The gateway runtime receives messages from webhooks and chat channels and binds each message to a session. Gateway credentials are the tokens for those channels, set with `caliche config gateway`. See [Gateway runtime](gateway.md).

## Plugin

A plugin is an extension defined by a schema under `src/plugins/`. A plugin can be a tool, a skill, a hook, an embedding provider, an evaluator, a document loader, a stream backend, or an output parser. The `plugins` Cargo feature loads plugins from a plugin store, and agents then receive a `plugin` tool.

## Policy

Policy decides whether a tool call is allowed, needs approval, or is denied. The typed rules live in `crates/caliche_policy`. Before any hooks run, the tool path calls `evaluate_tool_gate`, which applies the `trusted` preset together with the session overlay. A deny from the floor layer always blocks the call. See [caliche_policy](../crates/caliche_policy/README.md) and [Security](../SECURITY.md).

A policy is built from layers. The floor layer holds the rules compiled into the binary, and the deployment layer holds the operator's policy. The profile, project, and session layers can only narrow what the deployment allows. The exceptions are a deployment that lets the project layer widen it, and a session approval of an ask that is not hard. A preset is a named default for the deployment layer, such as `standard` or `trusted`. An overlay is a change that the session layer makes for the rest of a session, such as a permission mode like accept-edits. See [Layers and effects](../SECURITY.md#layers-and-effects).

The word preset also names a group of Cargo features, such as `agent` or `app`. See [Feature flags](reference/feature-flags.md#presets).

## Sandbox

The sandbox confines tools that run shell commands or access files. The default sandbox policy in the CLI is `workspace-write`. The available backends are described in [caliche_sandbox](../caliche_sandbox/README.md).

## Dripstone

Dripstone is the embedding store and vector search library in `dripstone/`. Caliche uses it for retrieval, and you can also use it on its own. The `VECTOR_BACKEND` environment variable selects a backend for the process. The status values that `caliche doctor` reports are explained in [Environment variables](reference/environment-variables.md#vector-backends).

ColdForge is a Dripstone vector backend that stores vectors in S3 and indexes them with SPFresh. The `vector-store-coldforge` feature compiles it, and `VECTOR_BACKEND=coldforge` selects it.

## Other names

These names appear in flags, features, and settings.

- **AgentFS**: a file system, key-value store, and tool audit log stored in SQLite. The `agentfs` feature compiles it.
- **RepoMap**: a short overview of the repository that `caliche code` adds to the agent's context, within a token budget. `--no-repomap` turns it off for one run.
- **Hashline**: a short content hash on each line that the `fs` tool reads, such as `42:ab12| text`. An edit names the hash, so a stale line number cannot change the wrong line. The `--hashline` options control it for indexed documents.
- **Jev compaction**: an optional filter that runs before compaction summarizes the conversation. It scores tool calls and tool results with an external service and drops or shortens the ones with a low score. See [Jev compaction pre-filter](reference/environment-variables.md#jev-compaction-pre-filter-experimental).

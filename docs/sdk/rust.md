# Rust SDK

The `caliche` crate is the Rust interface to the runtime. The crates are not published to crates.io (`publish = false`), so depend on a local checkout by path:

```toml
[dependencies]
caliche = { path = "/home/user/caliche" }
```

This enables the default `agent` preset. To choose the features yourself, see [Feature flags](../reference/feature-flags.md).

## Running an agent

```rust
use caliche::{AgentBuilder, ProviderType};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut agent = AgentBuilder::new()
        .provider(ProviderType::OpenAI)
        .model("gpt-4o")
        .system_prompt("You are a helpful assistant.")
        .build()
        .await?;

    let reply = agent.chat("Hello").await?;
    println!("{}", reply.text());
    println!("tokens: {:?}", reply.usage());

    Ok(())
}
```

## Middleware and reasoning schedules

```rust
use caliche::agent::hooks::{
    ReasoningPreset, ReasoningScheduleConfig,
};
use caliche::{AgentBuilder, ProviderType};

let agent = AgentBuilder::new()
    .provider(ProviderType::OpenAI)
    .model("gpt-4o")
    .with_default_middleware()
    .with_reasoning_schedule(ReasoningScheduleConfig::preset(ReasoningPreset::Sandwich))
    .build()
    .await?;
```

`with_default_middleware` registers three hooks: a pre-completion checklist, loop detection, and environment onboarding. The reasoning schedule sets `CompletionRequest.reasoning_effort` on each turn for OpenAI-compatible providers, and converts it to a thinking budget for Anthropic and Google.

## Several sessions and interaction modes

```rust
use caliche::agent::modes::InteractionMode;
use caliche::{AgentRuntime, AgentService, HarnessConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut runtime = AgentRuntime::new(HarnessConfig::default());
    let mut service = AgentService::new().await?;
    service.apply_interaction_mode(InteractionMode::Ask).await?;

    let session_id = runtime.create_session_from_service(service, None)?;
    let turn = runtime.run_turn(&session_id, "Explain this module").await?;
    println!("{}", turn.response);
    Ok(())
}
```

`AgentRuntime` holds several harness sessions in one process. In the Plan and Ask modes, `apply_interaction_mode` keeps only the tools for which `tool_registry().is_readonly()` is true; Plan mode also keeps `write_plan`. To change the provider, model, tools, and mode together, use `apply_settings` instead.

## Stream event contracts

A host that sends events to clients over SSE or WebSocket should convert runtime `StreamEvent` values with the SDK helpers, instead of copying fields itself:

```rust
use caliche::messages::{StreamEvent, envelope_value_from_stream_event};

let envelope = envelope_value_from_stream_event(
    StreamEvent::TextChunk {
        content: "hi".into(),
    },
    Some("evt-1".into()),
)?;
```

The SSE framing in `caliche_server` uses these helpers. The wire types are defined in `caliche_contracts`.

## Permission and approval contracts

```rust
use caliche::agent::hooks::{
    parse_permission_mode_str, permission_decision_from_json_value,
};
use caliche::harness::config::PermissionModeConfig;

let mode = parse_permission_mode_str(Some("accept-edits"))?;
assert_eq!(mode, PermissionModeConfig::AcceptEdits);

let decision = permission_decision_from_json_value(&serde_json::json!({
    "decision": "require_approval",
    "reason": "network"
}))?;
assert!(decision.is_approval_required());
```

The wire types are defined in `caliche_contracts::permission`. Hosts should parse permission modes and decisions with these helpers, instead of keeping their own string tables.

An unknown permission mode string returns an error; it does not fall back to `Default`. The table of mode names is `caliche_policy::parse_mode_overlay`, which `parse_permission_mode_str` and `caliche::policy` re-export. Before any hook runs, each tool call goes through `evaluate_tool_gate`, which applies the `trusted` policy preset together with the session's overlay. `CALICHE_POLICY_SHADOW=1` also logs a comparison with a shadow evaluation, without changing the gate result. See [caliche_policy](../../crates/caliche_policy/README.md).

## Turn metrics

`TurnResult` and `TaskResult` have a `discipline` field of type `HarnessDisciplineMetrics`. It contains `tokens_per_turn`, `tool_calls`, `compactions`, `readonly_filter_hits`, and, when the provider's pricing is known, `estimated_cost_usd`. Its `summary_line()` method formats these values as status text for the REPL and the API. Use these values to track cost and latency, instead of computing them in the host.

`messages::public_sse_event_name` and `harness::public_agent_sse_event_name` map stream and agent events to dotted public event names, such as `agent.thinking`, `tool.start`, `tool.end`, and `turn.complete`. The `discipline-budget-tests` feature enables a deterministic budget test for CI.

## Response caching

To cache model responses, give the builder an `LlmCache`:

```rust
use caliche::llm::cache::{CachePolicy, LlmCache};
use std::sync::Arc;

let cache = Arc::new(
    LlmCache::new(256, 300, CachePolicy::SemanticSimilarity { threshold: 0.92 })
        .with_embedding(embedding_provider),
);
let agent = AgentBuilder::new()
    .provider(ProviderType::OpenAI)
    .model("gpt-4o")
    .with_llm_cache(cache)
    .build()
    .await?;
```

## Single completions without an agent

Use `LlmClient` for one-off completions that need no tools, memory, or orchestration.

```rust
use caliche::{LlmClient, ProviderType};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = LlmClient::builder()
        .provider(ProviderType::OpenAI)
        .model("gpt-4o".to_string())
        .build()
        .await?;

    let response = client.complete_text("What is 2 + 2?").await?;
    println!("{response}");

    Ok(())
}
```

## Crates and types

For `0.1.x`, the root `caliche` crate is the supported Rust API. Each type below wraps the one above it. See also [Using Caliche as a Rust library](../../README.md#using-caliche-as-a-rust-library) in the README.

| Type | Adds |
| --- | --- |
| `Agent` | Runs a model with tools. |
| `AgentService` | Settings, a tool registry, and session storage. The CLI and the server use this type. |
| `AgentHarness` | An event stream, checkpoints, and undo for one interactive session. |
| `AgentRuntime` | Many harness sessions in one process. |

To serve agents over HTTP, use the `caliche_server` crate. The Agent Client Protocol is described in [ACP](../integrations/acp.md), and language server support in [Language servers](../integrations/lsp.md).

| Crate | Contents |
| --- | --- |
| `caliche` | The runtime: agents, providers, tools, the harness, MCP, hooks, and orchestration. |
| `caliche_contracts` | Wire types and parsers shared by clients and the server. It does not depend on `caliche`. |
| `caliche_orchestration` | Access control lists, privilege intersection, quarantine, and dispatch to an external agent runtime. It does not depend on `caliche`. |
| `caliche_policy` | Typed policy, including `evaluate_tool_gate`, which checks each tool call before hooks run. It does not depend on `caliche`. |

The JavaScript and Python SDKs load the runtime through native packages (`@caliche/runtime-node` and `caliche-runtime`), and the other SDK packages build on those.

## Memory

Caliche provides four memory strategies. Choose one based on what the session needs to keep:

| Strategy | Use |
| --- | --- |
| `BufferMemory` | Short conversations whose full history fits in the context window. |
| `SummaryMemory` | Longer conversations that can be summarized. |
| `EntityMemory` | Facts about named entities. |
| `VectorMemory` | Recall by meaning over long histories. |

## Local model inference

The `local-inference` feature enables local inference with Candle and ONNX Runtime. For a smaller build, enable only `local-inference-candle` or `local-inference-ort`. GGUF models through llama.cpp require the `local-inference-llamacpp` feature, which `local-inference` does not include. See [Feature flags](../reference/feature-flags.md).

Before you load a model, check the host's memory and which catalog models fit:

```rust
use caliche::llm::{FitOptions, LocalInferenceProvider};

let host = LocalInferenceProvider::probe_host_capabilities(FitOptions::default())?;
let safe = LocalInferenceProvider::safe_catalog_models(FitOptions::default())?;
println!("available RAM: {} bytes; safe models: {}", host.available_ram_bytes, safe.len());
```

These functions call Dripstone's `CapabilityProbe` and `fit_catalog`, which measure RAM and VRAM. In contrast, the `auto` value of `CALICHE_LOCAL_INFERENCE_DEVICE` chooses a device based only on the platform and the compiled-in features.

```rust
use caliche::llm::{InferenceBackendType, LocalInferenceConfig, LocalInferenceProvider};
use dripstone::loader::ModelSource;

let config = LocalInferenceConfig::new(
    ModelSource::huggingface("meta-llama/Llama-3.2-1B", None),
    InferenceBackendType::Candle,
)
.with_temperature(0.7)
.with_max_tokens(512);

let provider = LocalInferenceProvider::with_config(config).await?;
```

`examples/local_inference.rs` contains runnable configurations. From the CLI, use `caliche doctor` and the REPL commands `/local-model list` and `/local-model fit`.

## Security and sandboxing

Enable JWT authentication (`JWT_ENABLED` and `CALICHE_REQUIRE_AUTH`) before you expose the HTTP API in production. Plugin HTTP egress, MCP command allowlists, webhook verification, and sandbox settings are described in [Environment variables](../reference/environment-variables.md) and in `.env.example`.

The CLI runs the `bash` and `fs` tools in the OS sandbox by default, with the `workspace-write` policy. Use `--sandbox` to choose a policy and `--sandbox-backend` to choose a backend. See [Security](../../SECURITY.md).

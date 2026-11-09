# Examples

The runnable Rust examples are in `examples/`. Each one is registered as an `[[example]]` entry in the root `Cargo.toml`. To list the available names, run:

```bash
cargo run --example
```

## Rust examples

The table lists the most commonly used examples. An example with required features only builds when you enable those features with `--features`.

| Example | Purpose | Required features |
| --- | --- | --- |
| `agent` | Basic agent usage | None |
| `llm_client` | A model client without agent behavior | `openai` |
| `streaming` | Streaming responses | None |
| `tools` | Tool integration | `all-tools` |
| `memory` | Memory strategies | None |
| `rag` | Retrieval-augmented generation with vector backends | `tools-rag` |
| `vector_stores` | Vector backend configuration | `vector-memory` |
| `agent_controls` | Lifecycle hooks, permission modes, MCP servers, and session forking | `mcp` |
| `ollama` | Local models through Ollama | `ollama` |
| `structured_output` | Parsing output against a JSON schema | `output-parsing` |
| `workflows` | DAG and state machine workflows | None |
| `orchestration` | Coordinating multiple agents | None |
| `evaluation` | Evaluation against a dataset | None |
| `graph_agent` | Graph query tools with Apache AGE or Neo4j | `tools-graph`, `graph-age`, `graph-neo4j`, `openai` |
| `plugins` | Writing schema-based plugins | `plugins`, `openai` |
| `sandbox` | A mock sandbox and deterministic test providers | `testing` |
| `security` | API keys, role-based access control, rate limits, audit logging, and credential handling | None |
| `local_inference` | Local model inference with Candle or ONNX Runtime (GGUF through `local-inference-llamacpp`), plus helpers that check whether a catalog model fits the host | `local-inference` |
| `failover` | Failover between configured providers | `failover`, `openai`, `anthropic` |
| `compaction` | Context compaction strategies | `compaction`, `openai` |
| `cloudflare_ai_gateway` | Routing through the Cloudflare AI Gateway provider | `cloudflare-ai-gateway` |

Run examples from the workspace root:

```bash
cargo run --example agent
cargo run --example streaming
cargo run --example tools --features all-tools
cargo run --example rag --features tools-rag
cargo run --example plugins --features "plugins openai"
cargo run --example graph_agent --features "tools-graph,graph-age,graph-neo4j,openai"
cargo run --example local_inference --features local-inference
cargo run --example sandbox --features testing
cargo run --example security
```

The provider examples, such as `openai`, `anthropic`, `bedrock`, `azure_openai`, `google`, `cohere`, and `cloudflare_ai_gateway`, also need the credentials for that provider in the environment.

## Dripstone graph example

The Dripstone graph example compares the same operations on Apache AGE and Neo4j:

```bash
cargo run -p dripstone --example graph_age_neo4j_example --features "graph-age,graph-neo4j"
```

With the default example configuration, start the local graph services first:

```bash
docker compose up -d age neo4j
```

## JavaScript and Python examples

- JavaScript and TypeScript examples are in `caliche_js/examples/`.
- Python examples are in `caliche_py/examples/`.

# Feature flags

The `caliche` crate uses Cargo features to control which providers, tools, backends, and server components are compiled into a build. Enable only the features that your build needs.

The crates are not published to crates.io, so the examples depend on a local checkout by path. Features that you list are added to the default `agent` preset unless you also set `default-features = false`. The smallest useful build enables one provider:

```toml
[dependencies]
caliche = { path = "/home/user/caliche", default-features = false, features = ["openai"] }
```

To add tools to that build, list them as well:

```toml
[dependencies]
caliche = { path = "/home/user/caliche", default-features = false, features = ["openai", "tools-rag"] }
```

## Presets

A preset is a feature that enables a tested group of other features. Use a preset instead of listing individual features where one fits.

| Preset | Contents |
| --- | --- |
| `agent` (the default) | A lean local agent: the HTTP providers, the `fs` and `bash` tools with search (`tools-file`, `tools-grep`, and `tools-git`), the sandbox, compaction, provider failover, tiktoken token counting, and templates. |
| `app` | Everything in `agent`, plus the components for applications and servers: MCP, JWT validation, SQL and Redis persistence, the vector stores, routing, session tools, chains, LSP, and Prometheus metrics. It does not enable native local inference. |
| `full` | The build used by CI: everything in `app`, plus `local-inference` (Candle and ONNX Runtime), media tools, the Docker sandbox, [AgentFS](../concepts.md#other-names), scheduling, the [ColdForge](../concepts.md#dripstone) vector store, and Redis durable streams. |
| `full-local` | Everything in `full`, plus local embeddings and reranking. It requires the ONNX Runtime shared libraries. |

Each preset includes the one before it:

```text
agent  ⊂  app  ⊂  full  ⊂  full-local
```

The repository defines Cargo aliases for the presets in `.cargo/config.toml`. `cargo b`, `c`, `t`, `r`, and `cl` (build, check, test, run, and clippy) use `agent`. `cargo ba`, `ca`, `ta`, `ra`, and `cla` use `app`. `cargo bf`, `cf`, `tf`, `rf`, and `clf` use `full`.

## Binding presets

These presets control what the native JavaScript and Python bindings include.

| Feature | Contents |
| --- | --- |
| `bindings-minimal` | The smallest native build: the HTTP providers, the core and HTTP tools, MCP, and compaction. |
| `bindings-agent` | Everything in `bindings-minimal`, plus the grep and git tools and the sandbox. |
| `bindings-app` | The default native build. Everything in `bindings-agent`, plus JWT validation, prompts and chains, document loaders, reasoning patterns, and vector memory. `bindings` is an alias for this preset. |
| `bindings-sql` and `bindings-redis` | Optional SQL and Redis persistence for the bindings. |

## Common features

| Feature | Purpose |
| --- | --- |
| `auth-jwt` | JWT and JWKS validation with `jsonwebtoken`. Included in `app` and `bindings-app`. |
| `tools-grep` | The `grep` and `glob` operations of the `fs` tool. Included in `agent`, `app`, and `full`. |
| `testing` | Mock sandbox utilities and deterministic test providers. See `examples/sandbox.rs`. No preset includes it; the merge gate turns it on for the root crate tests and Clippy. |
| `sandbox` | The OS sandbox backends from `caliche_sandbox`, used by the `bash` and `fs` tools. It also compiles `SandboxOperationTransport`, which runs operations in a sandbox. |
| `sandbox-docker` | The Docker sandbox backend. Requires `sandbox`. |
| `tools-rag` | Retrieval-augmented generation tools. |
| `vector-memory` | Memory stored in a vector backend. It enables the `hnsw` feature of `dripstone`, which the ANN index needs. |
| `agentfs` | [AgentFS](../concepts.md#other-names): a SQLite-backed agent file system, key-value storage, and tool audit records. |
| `profiles` | Agent profiles written in Markdown. |
| `chains` | Chain workflows. |
| `reasoning-patterns` | ReAct, Plan-Execute, Reflexion, and related reasoning patterns. |
| `mcp` | MCP server support. |
| `plugins` | Schema-based plugins, stored in SQLite. |
| `otlp` | OpenTelemetry OTLP export, used when an endpoint is configured. |
| `persistence-sql` | SQL persistence that takes `postgres://` and `sqlite://` URLs: the connection manager, the PostgreSQL conversation and session stores, schema introspection, and the PostgreSQL audit and lifecycle backends. It compiles both the PostgreSQL and SQLite drivers. `sqlx` is the same set, and `tools-sql` adds the SQL tools to it. Included in `app`. |
| `sqlite-store` | The SQLite conversation, session, credential, and task stores. It compiles only the SQLite driver and does not include `persistence-sql`. |
| `durable-streams-sqlite` | Durable event streams stored in SQLite. It compiles only the SQLite driver and does not include `persistence-sql`. |
| `durable-streams-postgres` | Durable event streams stored in PostgreSQL. It compiles only the PostgreSQL driver and does not include `persistence-sql`. |

## Provider features

| Feature | Purpose |
| --- | --- |
| `providers` | The HTTP providers: OpenAI, Anthropic, OpenRouter, Perplexity, LM Studio, Cohere, Google, Ollama, and Cloudflare AI Gateway. |
| `providers-cloud` | Everything in `providers`, plus AWS Bedrock and Azure OpenAI. |
| `providers-all` | Everything in `providers-cloud`, plus `local-inference`. |
| `local-inference-candle` | Local generation with Candle and SafeTensors models. This is the smaller option. |
| `local-inference-ort` | Local generation with ONNX Runtime. It links the native ONNX Runtime library, so enable it only when you need it. |
| `local-inference-llamacpp` | Local generation of GGUF models with llama.cpp. It compiles native code, so enable it only when you need it. |
| `local-inference-llamacpp-metal` | llama.cpp with Metal GPU offload. |
| `local-inference-llamacpp-cuda` | llama.cpp with CUDA GPU offload. |
| `local-inference` | Both `local-inference-candle` and `local-inference-ort`. It does not include llama.cpp. Included in `full`. |
| `embedding-candle` | Text and vision embeddings with Candle. It enables the `vision-candle` feature of `dripstone`. |
| `embedding-ort` | Embeddings with ONNX Runtime. |
| `mysql` | MySQL connection pools and schema introspection. It enables the `mysql` feature of `sqlx`. |
| `video-processing` | Video embeddings for the local vision providers. It enables the `video-processing` feature of `dripstone`. |

Each provider also has its own feature. `providers` enables the first nine, and `providers-cloud` adds the last two.

| Feature | Provider |
| --- | --- |
| `openai` | OpenAI. |
| `anthropic` | Anthropic. |
| `openrouter` | OpenRouter. |
| `perplexity` | Perplexity. |
| `lmstudio` | LM Studio. |
| `cohere` | Cohere. |
| `google` | Google. |
| `ollama` | Ollama. |
| `cloudflare-ai-gateway` | Cloudflare AI Gateway. |
| `bedrock` | AWS Bedrock, through the AWS SDK. |
| `azure-openai` | Azure OpenAI, with Azure identity for sign-in. |

`local-inference-base` is a marker that the Candle, ONNX Runtime and llama.cpp features turn on. Do not enable it alone. `inference-mlx` enables local generation with MLX on Apple silicon through `dripstone`.

## Vector and graph features

| Feature | Purpose |
| --- | --- |
| `vector-store-redis` | The Redis vector backend. |
| `vector-store-postgres` | PostgreSQL with pgvector. |
| `vector-store-qdrant` | Qdrant over HTTP. |
| `vector-store-qdrant-grpc` | Qdrant over gRPC. |
| `vector-store-pinecone` | Pinecone. |
| `vector-store-chroma` | Chroma. |
| `vector-store-weaviate` | Weaviate. |
| `vector-store-milvus` | Milvus. |
| `vector-store-s3` | S3 storage. |
| `vector-store-coldforge` | [ColdForge](../concepts.md#dripstone), which stores vectors in S3 and indexes them with SPFresh. Included in `vector-store-aws` and `full`. |
| `vector-store-dynamodb` | DynamoDB native vector search (`SearchVectors`). |
| `vector-store-aws` | `vector-store-s3`, `vector-store-coldforge`, and `vector-store-dynamodb`, which together provide the S3, ColdForge, and DynamoDB backends. |
| `hybrid-search` | Enables the `rag` feature of `dripstone`. It links ONNX Runtime and is not part of `app`. |
| `tools-graph` | The graph query tool. |
| `graph-age` | The Apache AGE graph backend. |
| `graph-neo4j` | The Neo4j graph backend. |

## Other root features

The tables above do not cover these features of the `caliche` crate.

- `tools-core`: `tools-calculator`, `tools-file` and `tools-data`.
- `tools-extended`: `tools-core`, plus `tools-http` and `tools-document`.
- `all-tools`: `tools-extended`, plus `tools-grep`, `tools-git`, `tools-rag`, `tools-graph`, `tools-ast` and `tools-sql`.
- `tools-calculator`: the `calculator` tool.
- `tools-file`: the `fs` tool.
- `tools-data`: the `data_manipulation` tool.
- `tools-http`: the `http` tool.
- `tools-document`: the `process_document` tool, which reads images and PDF files.
- `tools-media`: audio transcription and image and video description. Included in `full`.
- `tools-ast`: the syntax tree tool. It enables `ast-all` and `tools-graph`.
- `ast`: the tree-sitter parser core.
- `ast-rust`, `ast-python`, `ast-javascript`, `ast-typescript`, `ast-go`, `ast-java`, `ast-c`, `ast-cpp`, `ast-json`, `ast-yaml`, `ast-html`, `ast-markdown` and `ast-csv`: one tree-sitter grammar each.
- `ast-all`: every grammar above.
- `compaction`: context compaction. Included in `agent`.
- `context-engine`: the pluggable context engine. It enables `compaction`.
- `failover`: the model failover chain and auth profile rotation.
- `router`: provider routing with static, weighted and cost strategies.
- `prometheus`: the Prometheus metrics recorder and scrape helpers.
- `lsp`: language server diagnostics and type information.
- `session-tools`: the tools that list, read, message and start other agent sessions.
- `scheduling`: scheduled agent tasks.
- `learning-loop`: self-evaluation, skill authoring and memory nudges. It enables `plugins`.
- `templates`: prompt templates.
- `prompts`: prompt building blocks used by chains and RAG.
- `output-parsing`: structured output parsing with JSON Schema.
- `document-loaders`: CSV, Markdown and HTML loaders.
- `document-loaders-extended`: `document-loaders`, plus PDF, spreadsheet and zip-based document loaders.
- `retrieval-chains`: chains that retrieve from a vector store. It enables `chains`.
- `chain-features`: `prompts`, `output-parsing`, `document-loaders`, `chains` and `retrieval-chains`.
- `token-tiktoken`: token counting with tiktoken.
- `mcp-client`: the MCP client. It enables `mcp`.
- `code-mode`: MCP code execution inside the sandbox. It enables `mcp` and `sandbox`.
- `redis` and `persistence-redis`: Redis persistence through a connection pool.
- `durable-streams-redis`, `durable-streams-mmap` and `durable-streams-s3`: durable event streams stored in Redis, memory-mapped files and S3.
- `s3`: reading files from S3.
- `standard-vector-stores`: `vector-store-redis`.
- `vector-store-primary`: `standard-vector-stores`, plus PostgreSQL and both Qdrant backends.
- `vector-store-http`: Pinecone, Chroma, Weaviate and Milvus.
- `vector-store-all`: `vector-store-primary`, `vector-store-http`, `vector-store-s3`, `vector-store-dynamodb` and `vector-store-mmap`.
- `vector-store-sqlite`: the SQLite vector backend.
- `vector-store-mmap`: the memory-mapped file vector backend.
- `vector-store-s3-backend`: the S3 vector backend that `vector-store-s3` enables.
- `rag-reranking`: reranking for RAG. It enables the `rag` feature of `dripstone`.
- `embeddings`: `embedding-candle` and `embedding-ort`.
- `reqwest`: the HTTP client. Most provider and tool features enable it.
- `discipline-budget-tests`: test fixtures for the harness discipline budget. It needs no provider.

## CLI provider features

The `caliche_cli` crate has its own features for the providers and local backends that its default build leaves out. Each one turns on the root feature with the same name:

| Feature | Purpose |
| --- | --- |
| `bedrock` | AWS Bedrock. |
| `azure-openai` | Azure OpenAI. |
| `local-inference` | Local generation with Candle and ONNX Runtime. |
| `local-inference-llamacpp` | Local generation of GGUF models with llama.cpp. |
| `inference-mlx` | Local generation with MLX on Apple silicon. |

`local-inference-base` is a marker that the local generation features turn on. The CLI shows `/local-model` and the `local-inference` provider only when it is on.

The CLI offers only what its build can run: `/provider`, `/subagent-provider`, `/embedding`, `/local-model download --backend`, the setup pickers and `doctor`. Selecting a provider or backend that the build left out fails with one line that names the feature, for example `bedrock is not in this build. Rebuild with --features bedrock.` Credentials are kept for every known provider in any build, so `creds export` still includes AWS and Azure keys.

## Embedded web UI

`embedded-ui` is a feature of the `caliche_cli` crate, not of `caliche`. It compiles the Dioxus web UI into the CLI server binary through the `embedded-ui` feature of `caliche_server`. The UI and the HTTP API are served from the same listen address.

The feature is off by default. Building with it requires `caliche_web/dist/index.html`, which the `dx` tool produces, and the build fails when that file is missing. `cargo test` and `cargo build -p caliche_cli` do not need the `dx` toolchain. The installation command and the release build of the CLI build the web assets first and then enable the feature:

```bash
./caliche rust install
```

That command runs these two steps:

```bash
./caliche web wasm
cargo install --path caliche_cli --locked --force --features embedded-ui
```

For a debug build from the checkout, run `./caliche rust build-with-ui`, which runs:

```bash
./caliche web wasm
cargo build -p caliche_cli --features embedded-ui
```

Then start the server:

```bash
caliche --server
```

`GET /` serves the UI, on the same origin as `/api/*` and `GET /health`. Do not run the UI on a second port.

A binary built without the feature returns `404` for `GET /`. That `404` means only that the UI or its assets are not in the binary. If `GET /health` succeeds, the API is running. The missing UI does not make the server more secure. `./caliche rust test-embedded-ui` compiles the API with the feature and checks that `GET /` returns HTML. When `caliche_web/dist` is absent, the test uses a fixture page; it does not build the real UI.

The UI uses the same listen address and security defaults as the API; see [HTTP server defaults](../../SECURITY.md#http-server-defaults).

## Wire contract features

`acp` is a feature of the `caliche_contracts` crate, not of `caliche`. It compiles `caliche_contracts::acp`, the Agent Client Protocol wire types, and the `agent-client-protocol-schema` crate. It is off by default. `caliche_server`, `caliche_cli`, and `caliche_acp_client` turn it on. The root crate and the bindings do not use it.

## Root crate feature tests

`./caliche rust test-features` runs the tests of the root crate on macOS with a wide set of features. It does not enable CUDA and does not download ONNX Runtime. The feature list is `ROOT_TEST_FEATURES` in `scripts/caliche/rust`, and it includes `mysql` and `embedding-candle`.

Tests that need a running MySQL server or that download an embedding model are marked as ignored, with a reason. `tests/local_embeddings_test.rs` is ignored because it downloads a Candle model. This test run includes no test against a running MySQL server.

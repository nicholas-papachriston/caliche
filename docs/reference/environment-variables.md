# Environment variables

This page lists the environment variables that Caliche reads, grouped by area. Set only the variables for the providers, vector backends, graph backends, and server features that you use. `.env.example` in the repository root contains the same variables as a template. `caliche` and `caliche_server` read a `.env` file only when you pass `--env-file <path>`. A variable already set in the shell keeps its value.

## Provider credentials

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
OPENROUTER_API_KEY=...
PERPLEXITY_API_KEY=...
GOOGLE_API_KEY=...
COHERE_API_KEY=...
```

## Web search and fetch

`web_fetch` is part of the default coding tool set (`fs`, `bash`, and `web_fetch`). `web_search` is off until you enable it. `web_fetch` reads page text; it is not a browser and does not run JavaScript.

```bash
TAVILY_API_KEY=...
EXA_API_KEY=...
TINYFISH_API_KEY=...
# TINYFISH_SEARCH_API_URL=https://api.search.tinyfish.ai
# TINYFISH_FETCH_API_URL=https://api.fetch.tinyfish.ai
FIRECRAWL_API_KEY=...
# FIRECRAWL_API_URL=https://api.firecrawl.dev
```

`web_fetch` tries TinyFish when `TINYFISH_API_KEY` is set, then Firecrawl when `FIRECRAWL_API_KEY` is set, and then a plain HTTP request. When neither key is set, only the plain HTTP request runs, and the URL is not sent to Firecrawl. `web_search` tries Tavily, then Exa, then TinyFish when its key is set, and then Firecrawl. Firecrawl search works without a key, with requests limited per IP address. Firecrawl fetch runs only when `FIRECRAWL_API_KEY` is set. TinyFish search and fetch both require `TINYFISH_API_KEY`, which is available from agent.tinyfish.ai. The order of the backends is described in more detail in [Tools](../tools.md).

## Provider defaults

```bash
DEFAULT_PROVIDER=openai
DEFAULT_MODEL=gpt-4o
RUST_LOG=info
```

## Image generation

```bash
CALICHE_IMAGE_PROVIDER=openai
CALICHE_IMAGE_MODEL=gpt-image-1
# OPENAI_IMAGE_MODEL=gpt-image-1
# GOOGLE_IMAGE_MODEL=models/gemini-2.5-flash-image
# GEMINI_IMAGE_MODEL=models/gemini-2.5-flash-image
CALICHE_ARTIFACT_DIR=.caliche/artifacts
```

When `CALICHE_IMAGE_PROVIDER` is `openai` or `google`, the agent exposes the `generate_image` tool. Generated images are saved under `.caliche/artifacts/images` by default. A relative `CALICHE_ARTIFACT_DIR` is resolved against the workspace.

## AWS Bedrock

```bash
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v1
```

## Azure OpenAI

```bash
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_RESOURCE_NAME=...
AZURE_OPENAI_DEPLOYMENT_ID=...
```

## Local providers

```bash
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=local-model
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.3
# Seconds to wait for the first token (default 600, clamped to 10..3600).
# The user configuration key local_first_token_timeout_secs wins.
# CALICHE_LOCAL_FIRST_TOKEN_SECS=600
```

## In-process local inference

Local inference is controlled by these Cargo features: `local-inference` (Candle and ONNX Runtime), `local-inference-candle`, `local-inference-ort`, and `local-inference-llamacpp` (GGUF models; opt-in, and not included in `local-inference`).

```bash
# CALICHE_LOCAL_INFERENCE_BACKEND=candle          # candle | ort | mlx | gguf | llamacpp
# CALICHE_LOCAL_INFERENCE_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
# CALICHE_LOCAL_INFERENCE_DOWNLOAD_REPO=TinyLlama/TinyLlama-1.1B-Chat-v1.0
# CALICHE_LOCAL_INFERENCE_DOWNLOAD_REVISION=main
# CALICHE_LOCAL_INFERENCE_MODEL_DIR=./models/tinyllama-1.1b-candle
# CALICHE_LOCAL_INFERENCE_MODELS_DIR=./models
# CALICHE_LOCAL_INFERENCE_HF_CACHE_DIR=
# CALICHE_LOCAL_INFERENCE_BOOTSTRAP=1
# CALICHE_LOCAL_INFERENCE_CONTINUE_ON_ERROR=1
# CALICHE_LOCAL_INFERENCE_DISABLE_CURL_FALLBACK=1
# CALICHE_LOCAL_INFERENCE_MAX_DOWNLOAD_BYTES=21474836480
# CALICHE_LOCAL_INFERENCE_DISK_RESERVE_BYTES=536870912    # bytes that must remain free after download
# CALICHE_LOCAL_INFERENCE_RESERVE_RAM_BYTES=1610612736    # bytes that must remain free after load
# CALICHE_LOCAL_INFERENCE_MAX_TOKENS=256
# CALICHE_LOCAL_INFERENCE_DEVICE=auto             # auto|cpu|metal|cuda|mlx (preference among compiled-in devices)
# HF_TOKEN=
```

`CALICHE_LOCAL_INFERENCE_DEVICE=auto` chooses a device based only on the platform and the features compiled into the binary. `LocalInferenceProvider::fit_options_from_env` applies `CALICHE_LOCAL_INFERENCE_RESERVE_RAM_BYTES`; Dripstone reads its own variable, `DRIPSTONE_RESERVE_RAM_BYTES`. Before you load large model weights, check whether they fit in the host's RAM and VRAM with `LocalInferenceProvider::fit_catalog_models`, with `caliche doctor`, or with the `/local-model fit` REPL command.

## Cloudflare AI Gateway

```bash
CLOUDFLARE_AIG_API_KEY=...
CLOUDFLARE_ACCOUNT_ID=...
CLOUDFLARE_AIG_GATEWAY_ID=default
CLOUDFLARE_AIG_MODEL=your-provider/your-model
# CLOUDFLARE_AIG_AUTH_TOKEN=...
```

`CLOUDFLARE_AIG_MODEL` uses the format `{provider}/{model}` and is validated before the provider is built. Model discovery first tries the OpenAI-compatible `/models` path of the gateway, then the Cloudflare Workers AI model search endpoint for Workers AI model ids. `CLOUDFLARE_ACCOUNT_ID` is required. `CLOUDFLARE_AIG_GATEWAY_ID` defaults to `default`.

## API server

The HTTP server defaults are in [HTTP server defaults](../../SECURITY.md#http-server-defaults). A host that is reachable from the internet must set `CALICHE_ENV=production`, `JWT_ENABLED=true` with a public key or a JWKS URL, non-empty `JWT_AUDIENCE` and `JWT_ISSUER` values that you supply, and `CALICHE_REQUIRE_AUTH=true`, and it must leave `CALICHE_ALLOW_INSECURE_PRODUCTION` unset. The complete list is in the [production authentication checklist](../ops/production-auth-checklist.md).

`API_HOST=0.0.0.0` and `API_HOST=::` mean every network interface. They do not set the CLI listen address, and they are unsafe without the settings above. The CLI `--bind` option defaults to `127.0.0.1` and does not read `API_HOST`. The standalone `caliche_server` binary reads `PORT` (default `3000`) and `BIND_ADDRESS`, which defaults to `127.0.0.1` when unset. The router logs a warning at startup, and still starts, when `API_HOST` or `BIND_ADDRESS` is `0.0.0.0` or `::` while both the production profile and `CALICHE_REQUIRE_AUTH` are off. An explicit `--bind 0.0.0.0` or `--bind ::` produces the same warning and also still starts. Loopback addresses (`127.0.0.1`, `::1`, and `localhost`) produce no warning. Authentication is configured separately; see [Security](../../SECURITY.md).

```bash
API_HOST=127.0.0.1
PORT=3000
JWT_ENABLED=false
JWT_PUBLIC_KEY=
JWT_PUBLIC_KEY_PATH=
JWT_JWKS_URL=
JWT_TENANT_CLAIM=tenant_id
JWT_REQUIRED_CLAIMS=sub
JWT_VALIDATE_EXP=true
# Required when JWT_ENABLED=true. Empty values fail startup.
# There is no built-in audience or issuer.
# JWT_AUDIENCE=
# JWT_ISSUER=
CALICHE_REQUIRE_AUTH=false
CALICHE_ALLOW_UNAUTH_ORCHESTRATION=false
CALICHE_JSON_RPC_ONLY=false
CALICHE_SHARE_DIR=
CORS_PERMISSIVE=false
```

When `CALICHE_SHARE_DIR` is set, redacted session shares are saved as JSON files in that directory, so that `/s/{slug}` links keep working after the process restarts. When it is unset, shares are kept only in process memory.

When `CALICHE_CONFIG_DIR` is set to a non-empty path, that path is used as the `caliche` directory for `server.lock`, `servers/`, `config.toml`, `mcp.json`, and `plugins.db`. An empty value is ignored. When the variable is unset, the lock file and the configuration files stay in the platform configuration directory, and `plugins.db` stays in the platform data directory. On Windows, both platform directories are `%APPDATA%`, and they do not follow `XDG_CONFIG_HOME`. The standalone `caliche_server` binary also stores its `tasks.db` in this directory.

Caliche trusts the contents of this directory: the REPL sends its turns to the server port recorded in `server.lock`, and `server stop` acts on the records in `servers/`. Point the variable only at a directory that you own and that other users cannot write to.

These paths do not follow `CALICHE_CONFIG_DIR`:

| Path | Location |
| --- | --- |
| CLI logs | `dirs::data_dir()/caliche/logs` (`%APPDATA%\caliche\logs` on Windows) |
| Endpoint store | `dirs::data_dir()/caliche/endpoints.json` (same `%APPDATA%` directory on Windows) |
| Gateway cluster DB | `~/.caliche/gateway-cluster.db` |
| Gateway logs | `~/.caliche/logs/gateway` |
| CLI session DB | `~/.caliche/caliche.db`, from `HOME` |
| CLI task ledger | `~/.caliche/tasks.db`, beside the session DB |

`caliche --server` uses the CLI task ledger, so its `tasks.db` follows `HOME` even when `CALICHE_CONFIG_DIR` is set.

The CLI configuration file can also configure session sharing:

```toml
[share]
policy = "manual"          # disabled | manual | auto
backend = "memory"         # memory | remote
visibility = "public"      # public | authenticated
backend_url = "https://example.com"
redact_patterns = ["sk-.*", "Bearer .*"]
```

`policy = "disabled"` rejects requests to create a share. `visibility = "authenticated"` requires an authenticated JWT caller to open `/s/{slug}` or `/api/agent/shares/{slug}`.

`JWT_JWKS_URL` is loaded by `caliche::security::JwksCache`, which `caliche_server` re-exports. Keys are served from the cache until their time to live expires. A token with an unknown `kid` triggers a rate-limited refresh, so that key rotation works before the cache expires. If a refresh fails, the previously loaded keys are kept.

Further rules for the API server variables:

- `CORS_PERMISSIVE` is meant for development. It is refused when `CALICHE_ENV=production`, unless `CALICHE_ALLOW_INSECURE_PRODUCTION=1`.
- `CALICHE_REQUIRE_AUTH=true` requires a valid JWT on most `/api/*` routes. This includes the chat stream, unless `CALICHE_PUBLIC_CHAT=true`.
- The gateway simulation routes require a JWT. Only the signed adapter ingress paths (`/slack`, `/telegram`, `/discord`, `/twilio`, and `/teams`) stay public.
- The production profile requires `JWT_ENABLED` and `CALICHE_REQUIRE_AUTH`, unless `CALICHE_ALLOW_INSECURE_PRODUCTION=1`.
- When JWT is enabled, the operator must set `JWT_AUDIENCE` and `JWT_ISSUER`. The server never fills them in with a built-in value.
- `CALICHE_TRUSTED_PROXY` requires `CALICHE_TRUSTED_PROXY_CIDRS`, and honors `X-Forwarded-For` only from peers in those ranges.

See the [production authentication checklist](../ops/production-auth-checklist.md).

The orchestration routes deny requests by default. `/api/orchestration/*` and `/api/workflows/*` start agents, spend tokens, and run tools, so they refuse any request without an authenticated caller, even while the rest of the API runs without authentication during development. Enable JWT to use them, or set `CALICHE_ALLOW_UNAUTH_ORCHESTRATION=true` for a local run only. That setting is ignored when `CALICHE_ENV=production`.

## Security hardening

```bash
CALICHE_ENV=production
PLUGIN_HTTP_ALLOWLIST=api.example.com,cdn.example.com
CALICHE_ALLOW_INSECURE_PRODUCTION=false
CALICHE_PUBLIC_CHAT=false
CALICHE_TRUSTED_PROXY=false
CALICHE_TRUSTED_PROXY_CIDRS=10.0.0.0/8
CALICHE_PLUGIN_LOAD_POLICY=warn
CALICHE_MCP_COMMAND_ALLOWLIST=npx,uvx
CALICHE_HARNESS_COMMAND_ALLOWLIST=pi,hermes
CALICHE_WEBHOOK_REQUIRE_VERIFICATION=true
CALICHE_PUBLIC_BASE_URL=https://hooks.example.com
CALICHE_TRUST_FORWARDED_HEADERS=false
CALICHE_METRICS_TOKEN=...
TEAMS_WEBHOOK_SECRET=...
```

- `PLUGIN_HTTP_ALLOWLIST`: a comma-separated list of host names for plugin `http` steps, gateway webhook delivery, and scheduled webhook delivery. Private, loopback, CGNAT, link-local, and cloud metadata addresses are always blocked, including after DNS resolution and on each redirect. When the variable is set, only the listed hosts are allowed (`*` allows any public host). The production profile requires this variable unless `CALICHE_ALLOW_INSECURE_PRODUCTION=1`. The HTTP tool always blocks private and metadata addresses, even when no allowlist is set.
- `CALICHE_PLUGIN_LOAD_POLICY`: what happens when the plugin store fails to load. `warn` (the default) logs the error and continues; `fail` stops building the agent.
- `CALICHE_MCP_COMMAND_ALLOWLIST`: required before the API accepts stdio MCP commands, such as `npx` or `uvx`, through the MCP configuration endpoints. A bare name matches only that bare name; a path such as `/tmp/npx` is rejected unless that exact path is listed.
- `CALICHE_HARNESS_COMMAND_ALLOWLIST`: required before the coordinator starts an external harness (Pi, Hermes, or another CLI) when `CALICHE_ENV=production`. A bare name matches only that bare name; `/tmp/pi` is rejected unless that exact path is listed. A shell interpreter is never allowed as the executable.
- `CALICHE_WEBHOOK_REQUIRE_VERIFICATION`: when `true`, every enabled gateway webhook adapter must have its verification secret configured.
- `CALICHE_PUBLIC_BASE_URL`: the public origin of the server (for example `https://hooks.example.com`), used to reconstruct webhook callback URLs. Prefer this setting over forwarded headers.
- `CALICHE_TRUST_FORWARDED_HEADERS`: when `true`, `X-Forwarded-Proto` and `X-Forwarded-Host` are used to reconstruct webhook URLs. The default is `false`, so that clients cannot forge Twilio signatures.
- `CALICHE_METRICS_TOKEN`: a bearer token that `/metrics` accepts when `CALICHE_REQUIRE_AUTH=true`.
- `TEAMS_WEBHOOK_SECRET`: when set, the bearer token that Teams webhook requests must present. It is also required when `CALICHE_WEBHOOK_REQUIRE_VERIFICATION=true`.

The CLI runs the `bash` and `fs` tools in the sandbox by default, with the `workspace-write` policy. If the sandbox backend fails to initialize, the CLI stops at startup, unless you explicitly choose `--sandbox danger-full-access` with `--sandbox-backend noop`.

## SQL tenant filtering

```bash
SQL_TENANT_FILTERING_ENABLED=false
SQL_TENANT_ISOLATION_STRATEGY=row_level
SQL_TENANT_JWT_CLAIM=tenant_id
```

The `sql_execute` tool and plugin `db_query` steps run every query on one shared connection pool. They do not isolate tenants from each other.

- `SQL_TENANT_FILTERING_ENABLED`: when `true`, the SQL tools refuse every query and return an error. They do not run a query without isolation. The default is `false`, which runs queries unchanged. Leave it `false` only when every caller may read every row in the database.
- `SQL_TENANT_ISOLATION_STRATEGY`: selects the message in that error. `row_level`, `schema`, and `database` are not supported, and `none` gives no isolation, so all four refuse. Tenant isolation needs a separate database connection for each tenant, which the SQL tools do not provide. Any other value is refused as not recognized. The default is `row_level`.
- `SQL_TENANT_JWT_CLAIM`: the custom claim that holds the tenant id when the caller's context has no `tenant_id`. The default is `tenant_id`. When filtering is enabled and the request has no tenant id, the error says so. Plugin `db_query` steps run without a caller context, so they always get that error when filtering is enabled.

## Web UI file browser

```bash
CALICHE_FILE_BROWSING=true
CALICHE_WORKSPACE=/path/to/workspace
```

The file browser rejects requests for paths outside `CALICHE_WORKSPACE`.

## Vector backends

```bash
VECTOR_BACKEND=memory
REDIS_URL=redis://localhost:6379
VALKEY_URL=redis://localhost:6379
POSTGRES_URL=postgresql://localhost/vectors
QDRANT_URL=http://localhost:6333
QDRANT_GRPC_URL=http://localhost:6334
PINECONE_API_KEY=...
PINECONE_INDEX_HOST=...
CHROMA_URL=http://localhost:8000
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=...
MILVUS_URL=http://localhost:19530
S3_BUCKET=my-bucket
S3_PREFIX=embeddings/
AWS_REGION=us-east-1
# DynamoDB native vectors (VECTOR_BACKEND=dynamodb; feature vector-store-dynamodb)
DRIPSTONE_DYNAMODB_TABLE=my-vectors
DRIPSTONE_DYNAMODB_INDEX=embeddings
# DRIPSTONE_DYNAMODB_ENDPOINT=
# DRIPSTONE_DYNAMODB_ENSURE=0
VECTOR_BACKEND_DIMENSION=1536
```

`VECTOR_BACKEND` accepts `memory`, `redis`, `valkey`, `postgres`, `qdrant`, `qdrant-grpc`, `qdrant-http`, `pinecone`, `chroma`, `weaviate`, `milvus`, `s3`, `coldforge`, and `dynamodb`. For DynamoDB, also set `DRIPSTONE_DYNAMODB_TABLE` (or `DYNAMODB_TABLE`), and optionally `DRIPSTONE_DYNAMODB_INDEX`, `DRIPSTONE_DYNAMODB_ENDPOINT`, and `DRIPSTONE_DYNAMODB_ENSURE`. Memory-mapped and tiered vector backends are configured through the builder and configuration APIs, not through environment variables.

`caliche doctor` and `GET /api/settings` report the retrieval status as `vector_backend_state`, and the settings tab of the embedded web UI shows it under retrieval:

- `unconfigured`: `VECTOR_BACKEND` is not set. `caliche doctor` also prints an empty `vector_backend`. `VectorBackendConfig::from_env` still builds an in-memory backend when a caller asks it to read the environment and the variable is missing, but that fallback does not count as a configured backend.
- `configured`: this binary can build the backend named by `VECTOR_BACKEND`, because the required feature is compiled in. It does not mean that a remote store is reachable or healthy.
- `unavailable`: the name is unknown, or the required feature is not compiled in.

Dripstone stores and searches the vectors for the process.

## Graph backends

```bash
AGE_URL=postgres://postgres:postgres@127.0.0.1:5455/age
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

To start local graph services:

```bash
docker compose up -d age neo4j
```

## Gateway credentials

```bash
TELEGRAM_BOT_TOKEN=...
TELEGRAM_WEBHOOK_SECRET_TOKEN=...
DISCORD_BOT_TOKEN=...
DISCORD_PUBLIC_KEY=...
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_FROM=...
TWILIO_WEBHOOK_VALIDATE_SIGNATURE=true
SLACK_BOT_TOKEN=...
SLACK_SIGNING_SECRET=...
TEAMS_WEBHOOK_URL=...
TEAMS_WEBHOOK_SECRET=...
```

## Gateway cluster and egress controls

```bash
CALICHE_CLUSTER_REGISTER_TOKEN=...
CALICHE_CLUSTER_HOST_ALLOWLIST=example.com,worker.example.com
CALICHE_CLUSTER_ALLOW_HTTP=false
CALICHE_CLUSTER_ALLOW_PRIVATE=false
CALICHE_GATEWAY_ROUTING_RULES_PATH=./gateway-rules.json
CALICHE_EGRESS_POLICY_PATH=./gateway-egress.toml
CALICHE_EGRESS_MODE=default_deny
CALICHE_EGRESS_ALLOW=api.openai.com,api.anthropic.com
CALICHE_EGRESS_TOOLS=api.github.com
CALICHE_EGRESS_MCP=localhost,mcp.example.com
CALICHE_EGRESS_PROVIDERS=api.openai.com
CALICHE_EGRESS_INCLUDE_DEFAULT_PROVIDERS=true
CALICHE_SUSPEND_DIR=~/.caliche/suspended
```

- `CALICHE_EGRESS_POLICY_PATH`: a TOML egress policy file (see [Gateway](../gateway.md#egress-allowlist)). When unset, local development uses `allow_public`, in which tools are checked only for server-side request forgery.
- `CALICHE_EGRESS_MODE`: `allow_public` or `default_deny`. The production profile defaults to `default_deny` and requires an allowlist.
- `CALICHE_EGRESS_ALLOW`: hosts allowed for every egress class.
- `CALICHE_EGRESS_TOOLS`, `CALICHE_EGRESS_MCP`, and `CALICHE_EGRESS_PROVIDERS`: host allowlists for HTTP tools, HTTP MCP servers, and model providers respectively.
- `CALICHE_EGRESS_INCLUDE_DEFAULT_PROVIDERS`: when `true`, the well-known hosted model APIs, such as `api.openai.com` and `api.anthropic.com`, are allowed.
- `CALICHE_SUSPEND_DIR`: the directory for the snapshots that `/suspend` writes and `/resume` reads.
- `CALICHE_OPERATION_DISPATCH`: when `1`, `true`, `yes`, or `on`, `AgentBuilder::build` installs `LocalOperationManager`. Operations are committed and then executed through `ToolExecutionEngine`, with the same sandbox, cancellation, workspace root, metrics, AgentFS, hook denials, and parallel batching as the default path. The default is off. A `RemoteOperationManager` used with a configured sandbox refuses to run unless its transport is a `SandboxOperationTransport` bound to that sandbox handle. See [Runtime lifecycle](../architecture/runtime-lifecycle.md#operation-dispatch).

## Jev compaction pre-filter (experimental)

```bash
# TypeSafe System One API key. Either name turns the pre-filter on.
TYPESAFE_API_KEY=...
# CALICHE_JEV_API_KEY=...
# CALICHE_JEV_COMPACTION is on when unset. Set 0 or false to turn the pre-filter off.
# Only 1 and true count as on when the variable is set.
# CALICHE_JEV_COMPACTION=1

# Optional customization
# CALICHE_JEV_MODEL=jev-latest
# CALICHE_JEV_BASE_URL=https://api.typesafe.ai/v1/systemone
# CALICHE_JEV_KEEP_THRESHOLD=0.5
# CALICHE_JEV_PRESERVE_RECENT=6
# CALICHE_JEV_MAX_STATE_TOKENS=25000
# CALICHE_JEV_MAX_REQUEST_TOKENS=30000
# CALICHE_JEV_TRUNCATE_HEAD_CHARS=300
# CALICHE_JEV_MIN_REDUCTION_RATIO=0.0
```

The Jev pre-filter is an optional step that runs before context compaction summarizes the conversation. It sends tool calls and tool results that are not pinned to the TypeSafe System One API, which scores them. Tool text with a low score is truncated or dropped. User and assistant messages are always kept, and so are recent turns that you pin.

To turn the pre-filter on, set `TYPESAFE_API_KEY` or `CALICHE_JEV_API_KEY`, or call `AgentBuilder::jev_asker`. `CALICHE_JEV_COMPACTION` counts as on when it is unset. When it is set, only `1` and `true` keep the pre-filter on, and `0` and `false` turn it off. Without a key, the pre-filter does not run. If the network call fails, the key is missing, or the filter does not reduce the prompt enough, Caliche falls back to the normal summary.

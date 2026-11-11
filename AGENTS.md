# caliche

This file is the instruction file for coding agents that work in this repository. It covers the project layout, the build and test commands, the required checks before a push, and the coding conventions.

## Project overview

Caliche is an agent runtime written in Rust. It supports many model providers, runs tools, retrieves context through Dripstone, speaks MCP, orchestrates several agents, provides durable event streams and gateway adapters for chat platforms, and reports logs, metrics, and token usage.

You can run agents locally or as a service, and you can call the runtime from other applications through the Rust crate or the JavaScript and Python bindings.

**Tech stack:**

| Layer | Stack |
| ----- | ----- |
| Core runtime | Rust 1.92, edition 2024, Tokio, `anyhow`/`thiserror` |
| HTTP API | Axum, Tower, WebSocket |
| CLI and REPL | Clap and a terminal REPL |
| Web UI | Dioxus compiled to WebAssembly, embedded in the CLI server with the `embedded-ui` feature |
| Vector search and retrieval | Dripstone (in this repository) |
| JavaScript and TypeScript SDK | NAPI-RS, Bun, oxfmt, and oxlint, with workspace packages under `caliche_js/packages/*` |
| Python SDK | PyO3, Maturin, uv, ruff, and pytest |
| Sandbox | `caliche_sandbox`, with Docker, macOS Seatbelt, Linux Landlock, Windows Job object, WebAssembly, and `noop` backends |
| Prompt templates | Shotput (a Git submodule) |

For `0.1.x`, the root `caliche` crate is the supported Rust API. Three smaller crates do not depend on it: `crates/caliche_contracts` (wire types), `crates/caliche_orchestration` (access control lists, privilege intersection, quarantine, and dispatch to an external agent runtime), and `crates/caliche_policy` (typed policy).

### Tool policy

- Before any hook runs, each tool call goes through `evaluate_tool_gate`, which applies the `trusted` preset together with the session's overlay. A deny from the floor layer always blocks the call.
- In interactive mode, the agent asks before any side effect, including an HTTP GET and an MCP tool call. Hooks still run after an allow, and after an ask that the user has answered.
- `resolve_fs_path` resolves `..` segments and patch paths. Any access to a path outside the workspace, whether a read, list, search, write, delete, or patch, results in an ask. Accept-all and accept-edits cannot lower that ask, and when no approver is available the access is denied. When there is no workspace root, every file system tool is denied. Text in the user's message does not grant access to a path.
- `CALICHE_POLICY_SHADOW=1` logs the differences from an evaluation with the `standard` preset.
- The approval broker records one pending id, accepts one answer from a person, and checks the action binding again before the tool runs. There is no approval UI and no HTTP approval route.
- The POSIX, PowerShell, and cmd denylists run when the facet for that shell dialect is on.
- `policy show` reports when a named host cannot meet the sandbox requirements. A host that is not named does not change the result of `decide`. Running unattended on Windows needs Docker or WSL2, because there is no Windows kernel sandbox.
- A version 1 `policy.toml` selects the preset that `policy show` and Settings > Permissions display. The tool gate still uses `trusted`.
- The `landlock-net` feature denies TCP bind and connect when the kernel's Landlock ABI supports it.

The JavaScript and Python packages are layered in this order: `contracts`, `runtime`, `core`, and then the feature packages. The Rust crates are `caliche`, `caliche_contracts`, `caliche_orchestration`, and `caliche_policy`. Terms are defined in [docs/concepts.md](docs/concepts.md), and the policy presets are described in [crates/caliche_policy/README.md](crates/caliche_policy/README.md).

`shotput` is the only Git submodule. Clone with `--recurse-submodules`, or initialize `shotput` after a plain clone:

```bash
git clone --recurse-submodules https://github.com/nicholas-papachriston/caliche.git
# after a plain clone:
git submodule update --init shotput
```

## Repository structure

| Path | Purpose |
| ---- | ------- |
| `src/` | The root Rust library: the agent builder and service, model providers, tools, orchestration, plugins, MCP, the gateway, and persistence |
| `crates/` | `caliche_contracts` (wire types), `caliche_mcp` (the MCP protocol, pagination, providers, and router), `caliche_orchestration` (access control lists, privilege intersection, quarantine, and dispatch to an external agent runtime), and `caliche_policy` (the floor rules, strict parsing, the shadow comparison, and the tool gate) |
| `caliche_cli/` | The `caliche` binary, the REPL, credential storage, and the server entry points |
| `caliche_server/` | The HTTP API server: REST, WebSocket, JSON-RPC, MCP configuration, and gateway routes |
| `caliche_js/` | The JavaScript and TypeScript bindings (NAPI-RS) and their packages |
| `caliche_py/` | The Python bindings (PyO3) and their packages |
| `caliche_web/` | The Dioxus web UI; the WebAssembly build output goes to `dist/` |
| `caliche_sandbox/` | The sandbox backends for the `bash` and `fs` tools |
| `dripstone/` | Embedding storage, vector search, graph backends, SPFresh and ColdForge, and full-text search |
| `shotput/` | The Shotput template engine (a Git submodule) |
| `examples/` | Runnable Rust examples for providers, tools, retrieval, workflows, MCP, and more |
| `tests/` | Integration tests for the workspace |
| `docs/` | User and reference documentation |
| `templates/` | Prompt and agent profile templates |
| `scripts/` | Version lockstep checks and release helpers |
| `caliche` and `scripts/caliche/` | The developer command script (`./caliche <command>`) |
| `.cargo/config.toml` | Cargo aliases, linker notes, and incremental compilation |
| `docker-compose.yml` | Local services for integration tests, such as Redis, PostgreSQL, and Qdrant |
| `deploy/` | Docker Compose files for local Prometheus and Grafana |
| `Cargo.toml` | The workspace manifest, feature flags, and example targets |
| `.env.example` | The template for environment variables |
| `STYLE_GUIDE.md` | The visual and interaction rules for the web UI and the REPL |
| `README.md` | The quick start and the documentation index |

The main `src/` modules are `agent/` (including the service settings, interaction modes, and `hooks::permission_contracts`), `harness/` (`AgentHarness`, `AgentRuntime`, and turn metrics), `messages/` (including `stream_contracts`, which maps events to the client wire format), `llm/`, `mcp/`, `orchestration/`, `plugins/`, `gateway/`, `rag/`, `streams/`, `security/`, `observability/`, `lsp/`, and `data/`.

## Build commands

### Rust

```bash
cargo build --features full            # library build with the full preset
cargo check --features full            # type check
cargo build --release --features full
cargo run -p caliche_cli -- --help
cargo run -p caliche_cli --            # interactive REPL
cargo run -p caliche_cli -- --server   # HTTP API, with the UI when embedded-ui is on
cargo run -p caliche_cli -- --acp-stdio  # ACP agent over stdio (JSON-RPC)
cargo run --example agent              # cargo run --example lists the examples
cargo clippy --all-targets --features app,learning-loop,plugins,mcp-client,code-mode,discipline-budget-tests,testing,sqlite-store -- -D warnings
cargo fmt --all
cargo doc --features full --no-deps --open
./caliche audit                        # cargo-audit vulnerability check (configured in audit.toml)
```

The default Cargo feature preset is `agent`: a lean local agent with the HTTP providers, the `fs` and `bash` tools with search, compaction, and the sandbox. Use `--features app` for applications and servers (JWT, vector stores, MCP, and persistence, but no local inference), and `--features full` for the complete integration checks, which include local inference. Each preset includes the one before it. Use `full-local` when ONNX Runtime is installed.

For the native bindings, `--features bindings` selects the default native build (`bindings-app`), and `--no-default-features --features bindings-minimal`, `bindings-agent`, or `bindings-app` selects one level explicitly. The binding presets leave out `sqlx` and Redis; add them with `bindings-sql` and `bindings-redis`, which the JavaScript and Python packages expose as `sql` and `redis`. The NAPI and PyO3 crates default to `bindings`.

Release builds of the native modules use the `release-bindings` Cargo profile, which is `release` with `panic = "unwind"` and `strip = false`. Stripped fat-LTO modules sometimes fail to load on macOS, so `./caliche py build`, `./caliche py build-composable`, and `bun run build:release` load each module after building and fail if it does not load.

A Rust panic then raises `PanicException` in Python, rejects an async call in JavaScript, and throws a JavaScript error from a synchronous call, instead of ending the process. Every synchronous `#[napi]` function and method in `caliche_js` carries `catch_unwind`; add it to each new one.

The server crate, `caliche_server`, has a lean default. `--features app` adds the cloud providers, every vector store, and the Redis and PostgreSQL session stores; the Docker image and `./caliche rust build` use it. Without `app`, the server refuses `SESSION_STORE_BACKEND=redis` and `postgres` at startup. The route groups `tools-rag`, `document-loaders`, `chains`, `retrieval-chains`, and `reasoning-patterns` add HTTP routes, so they are off by default and in the Docker image; the merge gate checks, tests, and lints them. The CLI embeds the server without `app`; `--features server-app` adds it. The Cargo aliases `cargo b`, `c`, `t`, `r`, and `cl` use `agent`; `ba`, `ca`, `ta`, `ra`, and `cla` use `app`; and `bf`, `cf`, `tf`, `rf`, and `clf` use `full`. See [docs/reference/feature-flags.md](docs/reference/feature-flags.md).

Use `./caliche build-dev` while you work; `.ci.yml` maps the `build` check to it. Use `./caliche build` only for release artifacts, which use fat LTO and include the WebAssembly UI and the release wheels and native bindings. `CALICHE_WITH_UI=1 ./caliche build-dev` also builds the Dioxus UI and a CLI with `embedded-ui`. `./caliche rust install` installs a binary on `PATH` with the UI embedded: it builds the WebAssembly assets with `dx` and runs `cargo install --features embedded-ui`. `embedded-ui` is not a default feature of the CLI, so `cargo test` and `cargo build` do not need `dx`. `cargo build --profile release-fast --features agent` builds an optimized binary without fat LTO and without the UI.

### Developer command script

```bash
./caliche help           # list all commands
./caliche build-dev      # fast debug build with the agent preset; the build check
./caliche build          # release artifacts (LTO, WebAssembly UI, wheels); slow
CALICHE_WITH_UI=1 ./caliche build-dev  # debug build with the UI and an embedded-ui CLI
./caliche rust install   # install on PATH: WebAssembly UI, then cargo install --features embedded-ui
./caliche test           # Rust, Python, and JavaScript tests (builds the SDK debug bindings first)
./caliche rust test      # root crate, CLI, server, sandbox, embedded UI smoke test, and Dripstone
./caliche rust test-lib  # root crate library tests only: cargo test --lib --features full (needs ONNX Runtime)
./caliche rust test-features  # root crate tests with ROOT_TEST_FEATURES from scripts/caliche/rust (macOS; no CUDA; no ONNX Runtime download)
./caliche py test        # Python tests with uv and pytest
./caliche js test        # JavaScript tests with Bun
./caliche web wasm       # build the Dioxus UI into caliche_web/dist
./caliche rust build-with-ui  # release CLI with the embedded UI
./caliche rust clippy    # Clippy on the root crate, ACP client, CLI, server, caliche_js, sandbox, and Dripstone
./caliche rust clippy-strict  # optional pedantic, nursery, and cargo lint groups
./caliche rust complexity     # McCabe complexity report (lizard)
./caliche rust complexity --gate  # fail above complexity 20 (part of ./caliche ci)
./caliche lint           # format and fix Rust, Python (ruff), and JavaScript and TypeScript (oxfmt and oxlint)
./caliche ci             # local CI: wasm-opt check, fmt-check, Clippy, complexity gate, tests, type checks
./caliche clean          # remove all build artifacts
```

**Tools required by `./caliche build`:** `cargo`, `bun`, `uv`, `dx` (the dioxus-cli version that matches the major and minor version of the `dioxus` crate in `Cargo.lock`, currently 0.7.6), and `wasm-opt` from binaryen 117 or later. `./caliche web wasm` and `./caliche rust install` run `wasm-opt -Oz` with the WebAssembly features that Rust 1.92 enables by default for wasm32 (`--enable-bulk-memory`, multivalue, mutable-globals, nontrapping-float-to-int, reference-types, and sign-ext). binaryen 116 and older fail with `invalid code after misc prefix: 17` (`table.fill`) even with those flags, and the build stops instead of embedding that module. On macOS, install binaryen with `brew install binaryen`. `rust-toolchain.toml` includes the `wasm32-unknown-unknown` target. `./caliche` puts `~/.cargo/bin` first on `PATH`, so that the rustup toolchain is used instead of a Nix `rust-default` without the wasm target; otherwise `dx build --platform web` fails with "Missing rust target wasm32-unknown-unknown".

### Python SDK

```bash
./caliche py build-dev          # uv sync, then maturin develop --profile dev in packages/runtime
./caliche py build              # the same, with --profile release-bindings
./caliche py test               # uv run pytest tests/ -v
```

The native module is `caliche_runtime._caliche_py`, built from `caliche_py/packages/runtime`. The root `caliche` package in `caliche_py` is pure Python, builds with hatchling, and depends on `caliche-runtime`. `./caliche py build-dev` enables the test-only `panic-probe` feature, which `tests/test_panic_unwind.py` uses. Use `uv sync --group dev --all-packages`, not `uv pip install -e ".[dev]"`. The development tools and workspace packages are in `[dependency-groups].dev`, and the extras for users (`mcp`, `langchain`, `hooks`, and `full`) are in `[project.optional-dependencies]`.

### JavaScript SDK

```bash
cd caliche_js && bun install
bun run develop     # debug native binding, TypeScript build, import smoke test
./caliche js build  # release native binding, TypeScript build, import smoke test
./caliche js test   # bun test
```

`packageManager` is pinned to Bun. The NAPI `packageName` is `@caliche/runtime-node`, so the platform packages are named under that scope.

## Code conventions

### Rust

- **Edition and toolchain:** Rust 2024 edition, minimum supported Rust version 1.92 (`rust-toolchain.toml`).
- **Error handling:** never use `unwrap()` or `panic!()` in production code. Use `anyhow::Result<T>` or `thiserror` enums, propagate errors with `?`, and add context with `.context()`.
- **Strings:** use `beef::lean::Cow` for performance-critical strings where the surrounding code already does.
- **Async:** use the Tokio runtime, `#[tokio::main]` for binaries, and `async-trait` for async traits.
- **Imports:** take dependencies from the workspace in the root `Cargo.toml`. Do not use inline imports, except for a documented circular dependency.
- **Features:** providers, tools, vector backends, MCP, LSP, the sandbox, and plugins are behind Cargo features. A new enum variant must be handled exhaustively in the Python (`caliche_py`) and JavaScript (`caliche_js`) bindings.
- **Linting:** the lint policy is in `[workspace.lints]` in the root `Cargo.toml` and in `clippy.toml`. The merge check, `./caliche rust clippy` or `cargo cls` (`-D warnings`), denies risky restriction lints (`unwrap_used`, `expect_used`, `panic`, `todo`, `unreachable`, prints, `disallowed-*`) as well as allocation and async lints (`await_holding_lock`, `needless_collect`, `unnecessary_to_owned`, `cloned_instead_of_copied`, `inefficient_to_string`, `or_fun_call`, `unused_async`, `redundant_clone`, `large_futures`). Use `#[expect(..., reason = "...")]` instead of a bare `allow`. `./caliche rust clippy-strict` adds the `all`, `pedantic`, `nursery`, and `cargo` groups and is optional. Portable crates such as `caliche_contracts` warn on `std_instead_of_core` and `std_instead_of_alloc`. `./caliche rust complexity` reports McCabe cyclomatic complexity with lizard, with a limit of 20; `./caliche rust complexity --gate` fails above that limit and is part of `./caliche ci`. Do not enable `clippy::cognitive_complexity` in the merge check. For TypeScript, the oxlint `complexity` rule enforces the same limit during `./caliche js lint`; for Python, ruff `C901` enforces it during `./caliche py lint`.
- **Documentation:** document public APIs with `///` comments, with an `# Examples` section where it helps.
- **Persistence:** binary codecs use rkyv envelopes with a magic prefix, defined in `src/data/persistence/rkyv_envelope.rs`. Register any new 4-byte prefix there.

### Python

- Use `uv`, not a bare `python`.
- Lint and format with `ruff`, and type-check with `mypy`. Ruff `C901` fails above a McCabe complexity of 20.
- The packages under `caliche_py/packages/*` share one version, which the lockstep scripts check.
- `caliche-contracts` (`packages/contracts`) contains only wire types and parsers, with no native extension. Use it when a consumer needs nothing else.

### JavaScript/TypeScript

- Use Bun for installation, tests, and build scripts.
- Lint and format `caliche_js` with `oxfmt` and `oxlint`. The oxlint `complexity` rule (classic McCabe, maximum 20) runs as part of `./caliche js lint`.
- The NAPI-RS native bindings ship per-platform optional dependencies and a WASI fallback.
- The packages are under `caliche_js/packages/*`. Do not reuse a package name in the workspace.
- `@caliche/contracts` (`packages/contracts`) contains only wire types and parsers, with no NAPI binding. A feature package that only re-exports should keep `sideEffects: false`.
- The packages are ES modules first and also ship CommonJS (`type: module`, with `.cjs` targets for `require`). Write TypeScript imports without file extensions. `tsconfig.build.json` emits CommonJS, and `caliche_js/scripts/prepare-composable-packages.mjs` generates the `.cjs` files and the ES module wrappers. Feature packages should import `@caliche/core/*` subpaths, not the root `@caliche/core` module.

### Web UI and REPL

- The web UI and the REPL follow the monospace operator console style in `STYLE_GUIDE.md`. Keep that style for new features and visual fixes. Do not apply another style or redesign the UI unless the user asks.
- A new UI surface or a redesign needs an explicit request from the user. Record the chosen direction in `STYLE_GUIDE.md` before you implement it.
- The web UI design tokens are in `caliche_web/src/styles/mod.rs`. The REPL uses the same meanings with ANSI colors.
- Rules for mobile browsers: hover styles only under `@media (hover: hover) and (pointer: fine)`, transparent `-webkit-tap-highlight-color`, `dvh` viewport units, 16px minimum font size on inputs, `env(safe-area-inset-*)` padding on fixed chrome, controlled `overscroll-behavior`, and `viewport-fit=cover`. The base CSS is in `caliche_web/assets/main.css`.

## Testing

```bash
# Default Rust tests: root crate, CLI, server, sandbox, embedded UI smoke test, and Dripstone.
./caliche rust test

# Root crate with the full preset (optional; needs ONNX Runtime).
cargo test -p caliche --features full --no-fail-fast

# Python
cd caliche_py && uv run pytest tests/ -v

# JavaScript
cd caliche_js && bun test

# All languages
./caliche test

# Local CI sequence
./caliche ci

# Dripstone wall-clock benchmarks (release, one thread; not in the gate)
./caliche rust bench
```

Latency and throughput bounds do not belong in merge-gate tests. Mark them `#[ignore = "wall-clock bound; run with ./caliche rust bench"]`, add them to `./caliche rust bench`, and record the numbers in [BENCHMARKS.md](BENCHMARKS.md).

- Tests are in `tests/` (workspace integration tests), in each crate's `tests/` folder, and in inline `#[cfg(test)]` modules.
- Use `#[tokio::test]` for async Rust tests, and `serial_test` where tests cannot run in parallel.
- Keep tests deterministic. Mock external services, except in explicit integration tests.
- The examples create their own cache directories and artifacts when they run.
- `./caliche repl smoke` runs the REPL smoke test against the debug binary, which must already be built.

`dripstone` is an ordinary folder in this repository, and the workspace commands check it: `./caliche rust clippy`, `./caliche rust test`, and `./caliche rust complexity --gate`, which also scans `dripstone/src`.

Use `./caliche ci` for the complete local sequence: formatting, Clippy (including Dripstone), the complexity check, the Rust tests (including Dripstone), and the JavaScript and Python tests.

## Environment and configuration

Copy `.env.example` to `.env`. `caliche` and `caliche_server` load it with `dotenvy` only when you pass `--env-file <path>`; neither loads a `.env` file from the working directory on its own. Configure only the providers and backends that you use.

Common variables:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
DEFAULT_PROVIDER=openai
DEFAULT_MODEL=gpt-4o
VECTOR_BACKEND=memory
RUST_LOG=info
API_HOST=127.0.0.1
PORT=3000
```

**Provider credentials:** OpenAI, Anthropic, AWS Bedrock, OpenRouter, Perplexity, Google, Cohere, Azure OpenAI, LM Studio, Ollama, and Cloudflare AI Gateway. See `.env.example` and `docs/reference/environment-variables.md`.

**Vector backends:** `memory`, `redis`, `postgres`, `qdrant`, `pinecone`, `chroma`, `weaviate`, `milvus`, `s3`, `coldforge`, and `dynamodb` (native `SearchVectors`), plus tiered combinations. When `VECTOR_BACKEND` is unset, the backend is reported as unconfigured. See `VECTOR_BACKEND` in `.env.example`.

**Security settings for production:**

| Variable | Purpose |
| --- | --- |
| `JWT_ENABLED`, and `JWT_PUBLIC_KEY` or `JWT_JWKS_URL` | JWT authentication for the API. When it is enabled, requests that fail validation are rejected. |
| `CALICHE_REQUIRE_AUTH` | Requires a JWT on most `/api/*` routes. |
| `CALICHE_ENV=production` | Requires JWT and `CALICHE_REQUIRE_AUTH`, unless `CALICHE_ALLOW_INSECURE_PRODUCTION=1`. Every public host must set it. See `docs/ops/production-auth-checklist.md`. |
| `PLUGIN_HTTP_ALLOWLIST` | The host names allowed for plugin and webhook HTTP requests. Required in production. |
| `CALICHE_MCP_COMMAND_ALLOWLIST` | The stdio MCP commands that the API accepts. |
| `CALICHE_PLUGIN_LOAD_POLICY` | `warn` or `fail` when plugins fail to load. |
| `CALICHE_METRICS_TOKEN` | A bearer token for `/metrics` when authentication is required. |

**Image generation:** `CALICHE_IMAGE_PROVIDER=openai` or `google` enables the `generate_image` tool. Images are saved under `.caliche/artifacts/images` by default; `CALICHE_ARTIFACT_DIR` changes the location.

**Docker Compose:** `docker-compose.yml` starts local Redis, PostgreSQL, Qdrant, Weaviate, and other services for integration tests. Their passwords are the `STACK_*` variables in `.env.example`.

The complete references are `docs/reference/environment-variables.md`, `docs/configuration.md`, and `docs/reference/feature-flags.md`.

## Continuous integration

This repository has no hosted CI for pushes or pull requests. Run the checks below before every push; nothing else runs them.

### Agent gate (required before push)

Run these steps from the repository root, in order, and stop at the first failure.

```bash
cd "$(git rev-parse --show-toplevel)"
( ./caliche rust fmt-check && ./caliche rust clippy && ./caliche rust complexity --gate )  # lint
( ./caliche check )                              # check
( ./caliche test )                               # test
( ./caliche build-dev )                          # build
```

The stages are `lint`, `check`, `test`, and `build`.

`.ci.yml` maps `build` to `./caliche build-dev`; use `./caliche build` only for release artifacts. `./caliche test` runs the Rust tests (the root crate, the CLI, the server, the sandbox, the embedded UI smoke test, and Dripstone), followed by the Python and JavaScript tests. `./caliche rust clippy` lints:

- the root crate with `app,learning-loop,plugins,mcp-client,code-mode,discipline-budget-tests,testing,sqlite-store`;
- the server twice, once with its default features and once with `app,tools-rag,document-loaders,chains,retrieval-chains,reasoning-patterns`;
- `caliche_js` with all targets, its default features, and `panic-probe`.

`cargo clippy --all-targets --features full` and `cargo check --features full` still work; the Clippy command links ONNX Runtime and needs `libonnxruntime`.

`./caliche rust test` covers:

- `cargo test --workspace` for the member crates (contracts, policy, orchestration, MCP, the ACP client, the web UI, and the JavaScript and Python native tests), each with its default features. The root crate, the CLI, the server, the sandbox, and Dripstone are excluded from this pass and tested separately, as listed below.
- the root crate tests with `app,learning-loop,plugins,mcp-client,code-mode,discipline-budget-tests,testing,sqlite-store`. `testing` runs the mock-provider tests in `tests/sandbox_tests.rs`, and `sqlite-store` runs `tests/task_ledger_e2e.rs`. `app` does not include `hybrid-search`, which links ONNX Runtime.
- the CLI tests with `lsp,scheduling,test-utils`
- the server tests with `app,tools-rag,document-loaders,chains,retrieval-chains,reasoning-patterns`
- the sandbox tests with `all,landlock-net`
- the Dripstone `--lib --tests` pass and a separate `--doc` pass, with `memory-backend,mmap-backend,sqlite-backend,s3-backend,coldforge-backend,corpus,ast,simd,search-advanced,rust-stemmers,hnsw`
- the embedded UI smoke test, which checks that `GET /` returns HTML

These test runs are optional, because they need a service, a GPU, a model download, or another operating system:

- PostgreSQL, Qdrant, Redis, Milvus, Weaviate, and Chroma: start the service with `docker compose up`, then run `cargo test -p dripstone --features postgres-backend` (or `qdrant-backend`, `redis-backend`, and the other backend features)
- Docker sandbox: `cargo test -p caliche_sandbox --features docker`
- llama.cpp: `cargo test -p caliche --features local-inference-llamacpp` and `cargo test -p dripstone --features inference-llamacpp`
- ONNX Runtime, `hybrid-search`, and the `full` preset: `cargo test -p caliche --features full` when `libonnxruntime` is installed. The CLI `local-inference` feature is the same opt-in: `cargo test -p caliche_cli --features local-inference,lsp,scheduling,test-utils`. Candle and ONNX model downloads: `cargo test -p dripstone --features embedding-ort` or `embedding-candle`, and `cargo test --features full-local` for the root local-embedding tests
- CUDA builds: the `candle-cuda` and `inference-llamacpp-cuda` features
- JavaScript RAG tests: `cargo test -p caliche_js --features tools-rag`, then rebuild the native binding with that feature and run `bun test tests/rag.test.ts`
- Live LM Studio: `cargo test -- --ignored` with `CALICHE_RUN_LMSTUDIO_INTEGRATION=true`
- Live S3: `cargo test -p dripstone --features s3-backend -- --ignored` with `DRIPSTONE_INT_S3_BUCKET` set
- JavaScript live model tests: `CALICHE_LIVE=1 bun test`
- Linux sandbox tests (`tests/linux.rs`, `tests/linux_harden.rs`, Landlock, cgroup) run on Linux
- Windows sandbox tests and `caliche_cli/tests/windows_server_registry.rs` run on Windows

A test that was skipped because its variable is unset has not passed. Ignored tests print the reason they are ignored.

The command for each stage comes from `.ci.yml`. When Nix is available, run the commands inside `nix develop`, which uses `flake.nix`.

### Commit convention

Every commit, by an agent or a person, follows Conventional Commits:
`type(scope)?: summary`, where the type is one of `build`, `chore`, `ci`,
`docs`, `feat`, `fix`, `perf`, `refactor`, `revert`, `style`, or `test`.
Make one logical change per commit. Write the summary in the
imperative mood, starting with a lowercase letter and without a trailing
period. Aim for a 50-character header and never exceed 72. Leave a blank
line before the body, wrap the body at 72 columns, and explain what
changed and why. Do not use emojis, and do not push `wip`, `fixup!`, or
`squash!` commits to shared branches. Commits written by an agent add an
`Assisted-by: <agent> (<model>)` trailer. Never credit an AI tool with
`Co-authored-by:`.

## Documentation

| File | Role |
| --- | --- |
| `AGENTS.md` | This file: the instructions for coding agents, including the required checks, conventions, and environment. It is the only instruction file for agents. |
| `README.md` | The quick start and the documentation index |
| `STYLE_GUIDE.md` | The visual and interaction rules for the web UI and the REPL |

The reference documentation is in `docs/`, as plain Markdown. Read this file before you edit code.

## Notes

- **Workspace layout:** a Cargo workspace with the root library, the ordinary folders (`caliche_cli`, `caliche_server`, `caliche_js`, `caliche_py`, `caliche_web`, `caliche_sandbox`, `dripstone`), the `shotput` submodule, and composable crates under `crates/`.
- **Feature flags:** the presets are described under [Rust](#rust) and in [docs/reference/feature-flags.md](docs/reference/feature-flags.md).
- **Default tools:** `enable_all_default_tools()` registers `fs`, `bash`, and `web_fetch`. What happens when a tool call fails repeatedly is described in [docs/tools.md](docs/tools.md).
- **Skills:** the system prompt lists each discovered skill by name, description, and location, and the model reads the full skill with `fs op=read`. A skill applied explicitly with `/skill`, and a prebuilt skill, are still added to the prompt in full. The CLI selects tools with `--tools coding|readonly|extended|none` and `--exclude-tools`.
- **Subagents:** the `/subagent-provider` and `/subagent-model` settings are saved, and apply to `@mention` and `sessions_spawn`. The `--subagent-provider` and `--subagent-model` options and the `[subagent]` configuration section set the same values. `--complex` runs a loop that investigates and then verifies; in the API, use `POST /api/orchestration/coordinator` with `complex: true`. See [docs/architecture/orchestration.md](docs/architecture/orchestration.md).
- **Server registry:** `caliche --server` writes `~/.config/caliche/servers/<id>.json` and `server.lock`. How servers are stopped and how `CALICHE_CONFIG_DIR` moves these files is described in [docs/gateway.md](docs/gateway.md) and [docs/configuration.md](docs/configuration.md).
- **ACP:** `--acp-stdio` and `caliche_acp_client` are documented in [docs/integrations/acp.md](docs/integrations/acp.md).
- **Plugins (`plugins` feature):** plugins are defined by a schema, in `src/plugins/`. The plugin types include tool, skill, hook, embedding, evaluator, document_loader, stream_backend, and output_parser. `.with_plugins_store(path)` gives an agent the `plugin` tool.
- **LSP (`lsp` feature):** the `lsp_diagnostics`, `lsp_symbols`, and `lsp_hover` tools are available when the language server is on `PATH`. No environment variable turns them on or off. See [docs/integrations/lsp.md](docs/integrations/lsp.md).
- **Security:** the HTTP server defaults are in [HTTP server defaults](SECURITY.md#http-server-defaults), and a public host follows the [production authentication checklist](docs/ops/production-auth-checklist.md). The default sandbox policy is `workspace-write`. See [SECURITY.md](SECURITY.md).
- **Build speed:** `.cargo/config.toml` enables incremental compilation. macOS uses the system `ld64` linker, because Homebrew `lld` fails to resolve `-lclang_rt.osx`. To use `zld`, pass `cargo --config .cargo/config.local.toml`; Cargo does not load that file on its own. Linux uses the system linker; `mold` can be enabled in the same local file, but forcing `-fuse-ld=mold` fails the build when `mold` is not installed. With sccache, set `CARGO_INCREMENTAL=0`.
- **Dripstone:** route all vector search and retrieval through the Dripstone APIs in this repository, instead of duplicating them in the core crate. S3 Vector Buckets use `S3VectorBackendConfig::with_sidecar_s3_bucket`. DynamoDB native vectors use `DynamoDbVectorBackend` with `VECTOR_BACKEND=dynamodb` and the `vector-store-dynamodb` feature. Tiered search should query the hot index. For late-interaction MaxSim scoring on a store without `native_maxsim`, use `search_with_maxsim_fallback` or `LateInteractionIndexedBackend`.
- **Examples:** each example is declared as an `[[example]]` in the root `Cargo.toml`, with its `required-features`. Run one with `cargo run --example <name>`. See [docs/examples.md](docs/examples.md).
- **Jev compaction:** an optional filter that runs before compaction summarizes the conversation ([definition](docs/concepts.md#other-names)). Its keys and `CALICHE_JEV_COMPACTION` are described in [docs/reference/environment-variables.md](docs/reference/environment-variables.md#jev-compaction-pre-filter-experimental).
- **Documentation:** start at [README.md](README.md) and [docs/concepts.md](docs/concepts.md).

## Contributor conventions

- The Caliche packages share one version, currently `0.1.0`, across the root `Cargo.toml`, `crates/*`, the CLI, the server, the web UI, the sandbox, the JavaScript packages, and the Python packages. `dripstone` and the `shotput` submodule keep their own versions.
- Set the shared edition, license, and authors in `[workspace.package]`. Take release versions from the manifests (`Cargo.toml`, `package.json`, and `pyproject.toml`), not from Git tags.
- Verify changes with `./caliche build-dev` and `./caliche ci`. Use `uv` or `uvx` for Python, and do not run project tooling with a bare `python`.
- Do not reuse a package or project name in the uv or npm workspaces.
- The `shotput` submodule must stay cloneable with `git clone --recurse-submodules` or `git submodule update --init shotput`.
- Do not commit `.env` files, credentials, or private keys. Copy `.env.example` for local development.

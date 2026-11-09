# Contributing to Caliche

This guide covers repository setup, the developer commands, the checks to run before you push, the `shotput` submodule, and the coding conventions.

## Repository setup

The CLI, the HTTP API, the web UI, the language bindings, the sandbox, and the vector store are ordinary folders in this repository. To change any of them, edit the folder and open one pull request against `caliche`. `shotput`, the prompt template engine, is the only Git submodule.

Clone the repository together with the submodule:

```bash
git clone --recurse-submodules https://github.com/nicholas-papachriston/caliche.git
```

If you already cloned the repository without the submodule, initialize it:

```bash
git submodule update --init shotput
```

The repository contains these folders:

| Path | Contents |
| --- | --- |
| `shotput` | The prompt template engine. This is a Git submodule; see [Changing shotput](#changing-shotput). |
| `dripstone` | Vector storage, search backends, and retrieval for retrieval-augmented generation. |
| `caliche_cli` | The command-line interface, the REPL, and the binary entry points. |
| `caliche_server` | The HTTP API server, WebSocket handlers, and route definitions. |
| `caliche_js` | The JavaScript and TypeScript bindings (NAPI-RS) and their packages. |
| `caliche_py` | The Python bindings (PyO3) and their packages. |
| `caliche_web` | The Dioxus web UI, compiled to WebAssembly. |
| `caliche_sandbox` | The sandbox backends for the shell and file system tools. |

## Developer commands

The `./caliche` script in the repository root runs the build, test, and lint steps for every part of the project. Each command is a script in [scripts/caliche/](scripts/caliche/). `./caliche help` lists them, and `./caliche <command> help` describes the subcommands of one command.

### General commands

| Command | What it does |
| --- | --- |
| `./caliche help` | Lists the available commands. |
| `./caliche build-dev` | Builds fast debug artifacts of the agent runtime. This is the build step of the pre-push checks. |
| `./caliche build` | Builds release artifacts, with link-time optimization, the WebAssembly UI, and release wheels. |
| `./caliche check` | Type-checks the Rust, JavaScript, and Python code without building it. |
| `./caliche lint` | Formats the code and applies automatic lint fixes: `cargo fmt` for Rust, `ruff` for Python, and `oxfmt` and `oxlint` for JavaScript and TypeScript. |
| `./caliche test` | Runs the Rust, Python, and JavaScript tests. |
| `./caliche test quick` | Runs the root crate library tests with `cargo test --lib --features full`, which needs ONNX Runtime, followed by the JavaScript and Python tests. |
| `./caliche test full` | Runs `./caliche rust test-features`, followed by the JavaScript and Python tests. |
| `./caliche ci` | Runs the local continuous integration sequence described in [Local CI sequence](#local-ci-sequence). |
| `./caliche ci --full` | Runs the same sequence, with `./caliche rust test-features` in place of `./caliche rust test`. |
| `./caliche clean` | Removes the Rust, Python, and JavaScript build artifacts. |
| `./caliche install` | Installs the Python dependencies with `uv` and the JavaScript dependencies with `bun`. |
| `./caliche version` | Prints the Rust, Python, and JavaScript package versions. |
| `./caliche info` | Prints tool versions, workspace paths, and build status. |
| `./caliche audit` | Checks the Rust dependencies for known vulnerabilities with `cargo-audit`. |

### Rust commands

| Command | What it does |
| --- | --- |
| `./caliche rust build-dev` | Builds the agent in debug mode. Set `CALICHE_WITH_UI=1` to also build the WebAssembly UI. |
| `./caliche rust build-with-ui` | Builds the release CLI binary with the embedded web UI. |
| `./caliche rust install` | Builds the WebAssembly UI, then installs `caliche` with `cargo install` and the `embedded-ui` feature. |
| `./caliche rust test` | Tests the root crate, the CLI, the server, the sandbox, the embedded web UI smoke test, and Dripstone. |
| `./caliche rust test-lib` | Runs only the root crate library tests, with `cargo test --lib --features full`. This needs ONNX Runtime. |
| `./caliche rust test-features` | Runs the root crate tests with the feature list in `ROOT_TEST_FEATURES` in [scripts/caliche/rust](scripts/caliche/rust). It runs on macOS, and does not enable CUDA or download ONNX Runtime. See [Feature flags](docs/reference/feature-flags.md#root-crate-feature-tests). |
| `./caliche rust clippy` | Runs Clippy with `-D warnings` on the root crate, the `caliche_acp_client` crate, the CLI, the server, the JavaScript bindings crate (`caliche_js`), the sandbox, and Dripstone. The root crate is linted with `app,learning-loop,plugins,mcp-client,code-mode,discipline-budget-tests,testing,sqlite-store`, which need no external services. The server is linted with its default features and again with `app,tools-rag,document-loaders,chains,retrieval-chains,reasoning-patterns`. `caliche_js` is linted with all targets, its default features, and the test-only `panic-probe` feature. Linting with `--features full` needs `libonnxruntime`. |
| `./caliche rust clippy-strict` | Runs additional Clippy passes with the `pedantic`, `nursery`, and `cargo` lint groups. It is optional. |
| `./caliche rust complexity` | Reports McCabe cyclomatic complexity with lizard. With `--gate`, it fails when a function exceeds 20. |

### Python, JavaScript, and web UI commands

| Command | What it does |
| --- | --- |
| `./caliche py test` | Runs the Python tests with `uv run pytest`. |
| `./caliche py lint` | Lints the Python code with `ruff`. |
| `./caliche py typecheck` | Type-checks the Python code with `mypy`. |
| `./caliche js test` | Runs the JavaScript and TypeScript tests with `bun test`. |
| `./caliche js lint` | Formats the JavaScript and TypeScript code with `oxfmt`, applies `oxlint --fix`, and then fails if any formatting or lint problems remain. |
| `./caliche js typecheck` | Type-checks the TypeScript code. |
| `./caliche web wasm` | Compiles the Dioxus UI to WebAssembly in `caliche_web/dist`. |

## Checks before you push

No Git hook is installed, so run the checks yourself and make sure that they pass before you push a commit. Run these four steps, in order, from the repository root:

```bash
./caliche rust fmt-check && ./caliche rust clippy   # lint
./caliche check                                     # type check
./caliche test                                      # test
./caliche build-dev                                 # build
```

`./caliche test` runs the Rust tests (the root crate, the CLI, the server, the sandbox, the embedded web UI smoke test, and Dripstone), followed by the Python and JavaScript tests. Tests that need PostgreSQL, Qdrant, Redis, Docker, CUDA, llama.cpp, an ONNX or Candle model download, a running LM Studio server, or a live S3 bucket are not run by default. Tests specific to Linux or Windows run only on those systems. The same steps are listed in the "Agent gate" section of [AGENTS.md](AGENTS.md).

### Local CI sequence

`./caliche ci` ([scripts/caliche/ci](scripts/caliche/ci)) runs a longer sequence:

1. `bash scripts/check_wasm_opt.sh` checks that the binaryen `wasm-opt` tool meets the requirements.
2. `./caliche rust fmt-check` checks the Rust formatting.
3. `./caliche rust clippy` runs Clippy with `-D warnings`.
4. `./caliche rust complexity --gate` fails when a function exceeds a McCabe cyclomatic complexity of 20.
5. `./caliche rust test` runs the Rust tests listed above. With `--full`, `./caliche rust test-features` runs instead.
6. `./caliche js typecheck` type-checks the TypeScript code.
7. `./caliche test js` builds the JavaScript debug bindings and runs the JavaScript tests with Bun.
8. `./caliche test py` builds the Python debug bindings and runs the Python tests with pytest.

## Where changes land

`caliche_server`, `caliche_cli`, `caliche_js`, `caliche_py`, `caliche_sandbox`, `caliche_web`, and `dripstone` are ordinary folders. Edit them and open one pull request against `caliche`.

### Changing shotput

`shotput` is a Git submodule, so changes to it are made in the shotput repository. Push the change there first, then record the new commit in this repository:

```bash
git submodule update --remote shotput
git add shotput
git commit -m "chore: bump shotput gitlink"
```

`.gitmodules` sets `branch = main` for `shotput`. Record only a shotput commit that is already on the `main` branch of that repository.

## Coding conventions

[AGENTS.md](AGENTS.md) contains the detailed technical rules and architecture decisions. The main conventions are:

- **Commit messages:** follow Conventional Commits, `type(scope)?: summary`, where the type is one of `build`, `chore`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`, `revert`, `style`, or `test`. Write the summary in the imperative mood, starting with a lowercase letter. Keep the header under 72 characters, and aim for 50. Wrap the body at 72 columns. Do not use emojis. Make one logical change per commit, and do not push `wip`, `fixup!`, or `squash!` commits to shared branches. Commits written by an agent add an `Assisted-by: <agent> (<model>)` trailer.
- **Rust:** the crates use the Rust 2024 edition, with a minimum supported Rust version of 1.92. Do not use `unwrap()` or `panic!()` in production code; propagate errors with `?`, and add context with `anyhow` or `thiserror`. Clippy warnings are errors (`-D warnings`). The McCabe cyclomatic complexity of a function must not exceed 20.
- **Python:** use `uv` to manage the environment. Lint and format with `ruff` (rule `C901` with a maximum complexity of 20), and type-check with `mypy`.
- **JavaScript and TypeScript:** use `bun` to manage the packages. Format with `oxfmt` and lint with `oxlint`. The maximum complexity is 20.
- **Versions:** the Caliche packages share one version, currently `0.1.0`, across the root `Cargo.toml`, `crates/*`, the CLI, the server, the web UI, the sandbox, the JavaScript packages, and the Python packages. `dripstone` and the `shotput` submodule keep their own versions.

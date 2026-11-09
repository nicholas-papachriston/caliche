# Caliche Python SDK

This folder contains the Python packages for Caliche. The native runtime package, `caliche-runtime`, calls the Rust runtime through bindings built with PyO3 and Maturin. The `caliche` package is pure Python and depends on `caliche-runtime`.

## Packages

| Package | Contents |
| --- | --- |
| `caliche` | The SDK package that most programs import (`import caliche`). It is pure Python. |
| `caliche-runtime` | The native runtime wheel, built with PyO3. |
| `caliche-core` | The core API. |
| `caliche-contracts` | Wire types and parsers, with no native extension. |
| `caliche-mcp` | MCP helpers. |
| `caliche-hooks` | Lifecycle hooks and session controls. |
| `caliche-integrations-langchain` | Adapters for LangChain. |

## Installation

The packages are not published to a package index, so build them from this checkout. From the repository root, run:

```bash
./caliche py build-dev
```

This installs the workspace packages with `uv` and builds the native module in `packages/runtime`. The manual steps are under [Developing from source](#developing-from-source).

## Requirements

- Python 3.9 or later. `caliche-integrations-langchain` requires Python 3.10 or later.
- A Rust toolchain, to build the native module.
- [`uv`](https://docs.astral.sh/uv/) for development.

## Quick start

```python
from caliche import AgentBuilder, ProviderType, create_calculator_tool

agent = (
    AgentBuilder()
    .provider(ProviderType.OpenAI)
    .model("gpt-4o")
    .system_prompt("You are a helpful assistant.")
    .enable_tool("calculator")
    .build()
)

response, usage, _metrics = agent.chat("What is 2 + 2?")
print(response)

# A built-in tool can also be called directly:
tool = create_calculator_tool()
print(tool.execute({"operation": "add", "a": 2, "b": 2}))
```

With asyncio:

```python
import asyncio
from caliche.async_agent import ProviderType, create_agent_async

async def main():
    agent = await create_agent_async(
        provider=ProviderType.OpenAI,
        model="gpt-4o",
    )

    response, usage, _metrics = await agent.chat("Hello")
    print(response)

asyncio.run(main())
```

## Providers

`ProviderType` has the same providers as the Rust runtime: `OpenAI`, `Anthropic`, `Bedrock`, `Google`, `Cohere`, `AzureOpenAI`, `OpenRouter`, `Perplexity`, `CloudflareAIGateway`, `LMStudio`, `Ollama`, `External`, and `LocalInference`.

When the native module is built with the `local-inference` feature, `probe_local_inference_host`, `fit_local_inference_catalog`, `fit_local_inference_model`, and `safe_local_inference_models` report which catalog models fit in the host's RAM before you load one. See `examples/local_inference_fit.py`.

## Streaming and hooks

Tool calls produce `tool_call_start` and `tool_call_result` streaming events. Python lifecycle hooks deny the tool call when they fail: if `pre_tool_use` raises an error or returns a malformed result, the tool does not run.

## Documentation

The documentation is in the `docs/` folder of the repository:

- [Python SDK](../docs/sdk/python.md)
- [MCP integration](../docs/integrations/mcp.md)
- [Feature flags](../docs/reference/feature-flags.md)
- [Environment variables](../docs/reference/environment-variables.md)

## Examples

The examples are in `examples/`:

```bash
uv run examples/agent.py
uv run examples/streaming.py
uv run examples/mcp_server.py
```

## Developing from source

Run these commands in `caliche_py`:

```bash
uv sync --group dev --all-packages
cd packages/runtime && uvx --from maturin maturin develop --profile dev && cd ../..
uv run pytest
uv run ruff check python/ packages/*/python tests/
```

The native module is built in `packages/runtime`, because the root `caliche` package is pure Python. `--all-packages` installs the workspace packages (`core`, `hooks`, `mcp`, and the others), so that imports such as `from caliche.hooks import ...` work. `./caliche py build-dev`, run from the repository root, does the same steps.

For an optimized build of the native module, use the `release-bindings` profile:

```bash
uv sync --group dev --all-packages
cd packages/runtime && uvx --from maturin maturin develop --profile release-bindings && cd ../..
```

To install the optional extras:

```bash
uv sync --group dev --all-packages --extra mcp
uv sync --group dev --all-packages --extra langchain
```

| Command | What it does |
| --- | --- |
| `uv sync --group dev --all-packages` | Installs the project, the workspace packages, and the development tools. |
| `cd packages/runtime && uvx --from maturin maturin develop --profile dev` | Builds the native module and installs it in editable mode. |
| `uv run pytest` | Runs the tests. |
| `uv run ruff check` and `uv run ruff format` | Lint and format the code. |
| `uv run mypy` and `uv run ty check` | Type-check the code. |

## License

Licensed under either of [Apache-2.0](../LICENSE-APACHE) or [MIT](../LICENSE-MIT), at your option.

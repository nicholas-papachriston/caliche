# Python SDK

The Python SDK calls the Rust runtime through native bindings built with PyO3 and Maturin. The source build uses `uv`.

## Packages

| Package | Contents |
| --- | --- |
| `caliche` | The SDK package that most programs import. |
| `caliche-runtime` | The native runtime wheel, built with PyO3. |
| `caliche-core` | The core API. |
| `caliche-contracts` | Wire types and parsers, with no native extension. |
| `caliche-mcp` | MCP helpers. |
| `caliche-hooks` | Lifecycle hooks and session controls. |
| `caliche-integrations-langchain` | Adapters for LangChain. |

## Installation

Build the packages from a checkout. From the repository root, run:

```bash
./caliche py build-dev
```

This runs `uv sync --group dev --all-packages` in `caliche_py`, then builds the native module with `maturin develop --profile dev --features panic-probe` in `caliche_py/packages/runtime`, which installs it as `caliche_runtime._caliche_py`. `panic-probe` adds a test-only function that `tests/test_panic_unwind.py` calls. For an optimized build of the native module, run `./caliche py build` instead, which uses the `release-bindings` profile and leaves `panic-probe` off. It then imports `caliche_runtime._caliche_py` and fails if the module does not load.

To run the same steps by hand:

```bash
cd caliche_py
uv sync --group dev --all-packages
cd packages/runtime
uvx --from maturin maturin develop --profile dev
```

The `caliche` package is pure Python and builds with hatchling. It imports the native module from `caliche-runtime` and declares `caliche-runtime` as a dependency, so installing `caliche` also installs the native module. Release builds of `caliche-runtime` use the `release-bindings` Cargo profile, so a Rust panic raises `PanicException` in Python and does not end the process. `./caliche py build-composable` imports the native module from the built wheel and fails if it does not load.

## Running an agent

```python
from caliche import AgentBuilder, ProviderType

agent = (
    AgentBuilder()
    .provider(ProviderType.OpenAI)
    .model("gpt-4o")
    .system_prompt("You are a helpful assistant.")
    .build()
)

response, usage, _metrics = agent.chat("Hello")
print(response)
```

## Running an agent with asyncio

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

    async for chunk in agent.stream_chat("Tell me a story"):
        if chunk.get("type") == "text_chunk":
            print(chunk["content"], end="", flush=True)

asyncio.run(main())
```

## Builder configuration

```python
agent = (
    AgentBuilder()
    .provider(ProviderType.OpenAI)
    .model("gpt-4o")
    .api_key("sk-...")
    .temperature(0.7)
    .max_tokens(2048)
    .enable_tool("calculator")
    .enable_all_default_tools()
    .json_mode()
    .build()
)
```

## Structured output

```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
}

agent = (
    AgentBuilder()
    .provider(ProviderType.OpenAI)
    .model("gpt-4o")
    .json_schema("Person", schema)
    .build()
)
```

## Single completions without an agent

Use `LlmClient` for one-off completions that need no tools or memory.

```python
from caliche import LlmClient, ProviderType

client = (
    LlmClient.builder()
    .provider(ProviderType.OpenAI)
    .model("gpt-4o")
    .api_key("sk-...")
    .build()
)

response = client.complete_text("What is 2 + 2?")
print(response)
```

## Attachments

```python
from caliche import LlmClient, ProviderType, Attachment
import base64

client = (
    LlmClient.builder()
    .provider(ProviderType.OpenAI)
    .model("gpt-4o")
    .api_key("sk-...")
    .build()
)

with open("recording.mp3", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("utf-8")

attachments = [
    Attachment(
        attachment_type="audio",
        data_base64=audio_b64,
        media_type="audio/mpeg",
        filename="recording.mp3",
    )
]

response = client.complete_with_attachments(
    "Transcribe this audio and summarize action items.",
    attachments,
)
```

## Memory

```python
from caliche import (
    BufferMemoryConfig,
    EntityMemoryConfig,
    SummaryMemoryConfig,
    VectorMemoryConfig,
)

builder.buffer_memory(BufferMemoryConfig(
    max_tokens=4096,
    min_messages=4,
    preserve_system=True,
    summary_placeholder=True,
))

builder.summary_memory(SummaryMemoryConfig(
    max_tokens=8192,
    summarize_threshold_pct=0.8,
    chunk_size=12,
    max_summaries=8,
))

builder.vector_memory(VectorMemoryConfig(
    max_tokens=8192,
    max_memories=1000,
    top_k=5,
    similarity_threshold=0.7,
    buffer_tokens=4096,
))
```

## Vector backends

```python
builder.set_vector_backend_memory(ttl_secs=86400, max_entries=100000)
builder.set_vector_backend_redis(url="redis://localhost:6379", prefix="my_app:")
builder.set_vector_backend_qdrant(url="http://localhost:6333", vector_dimension=1536)
builder.set_vector_backend_qdrant_grpc(grpc_url="http://localhost:6334", vector_dimension=1536)
builder.set_vector_backend_pinecone(base_url=host, api_key=key, dimension=1536)
builder.set_vector_backend_chroma(base_url="http://localhost:8000")
builder.set_vector_backend_weaviate(url="http://localhost:8080", vector_dimension=1536, api_key=key)
builder.set_vector_backend_milvus(endpoint="http://localhost:19530", vector_dimension=1536)
builder.set_vector_backend_dynamodb(table_name="vectors", dimension=1536, index_name="embeddings")
builder.vector_backend_from_env()
```

A tiered backend keeps frequently used vectors in a fast tier and the rest in slower tiers:

```python
builder.set_vector_backend_tiered(
    [
        (
            {"type": "memory", "ttl_secs": 3600, "max_entries": 10_000},
            {"name": "hot", "max_vectors": 10_000, "promotion_threshold": 1},
        ),
        (
            {"type": "redis", "url": "redis://localhost:6379", "prefix": "tier:"},
            {"name": "warm", "max_vectors": 100_000, "promotion_threshold": 2},
        ),
    ],
    write_strategy="write_to_source_and_hot",
    background_sync=True,
    sync_interval_secs=60,
)
```

## Custom tools

```python
from caliche import ToolResult
from caliche.tools import Tool
import json

class MyTool(Tool):
    def name(self) -> str:
        return "my_tool"

    def description(self) -> str:
        return "Tool description"

    def parameters(self) -> dict:
        return {"type": "object", "properties": {"input": {"type": "string"}}}

    def execute(self, arguments: dict) -> ToolResult:
        return ToolResult(success=True, content=json.dumps(arguments), error=None)

builder.add_tool(MyTool())
```

## Permission modes and allowed tools

```python
from caliche.hooks import PermissionMode

agent = (
    AgentBuilder()
    .provider(ProviderType.OpenAI)
    .model("gpt-4o")
    .permission_mode(PermissionMode.accept_edits())
    .allowed_tools(["calculator", "fs"])
    .build()
)
```

## Running an MCP server

```python
from caliche.mcp import McpServerBuilder

server = (
    McpServerBuilder()
    .name("example-mcp-server")
    .version("1.0.0")
    .enable_tools(True)
    .build()
)

server.run_stdio()
```

## Choosing the synchronous or asynchronous API

Use the synchronous API for scripts and notebooks. Use the asynchronous API for web applications, concurrent work, and streaming responses.

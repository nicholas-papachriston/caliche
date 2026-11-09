# Language servers

When built with the `lsp` Cargo feature, Caliche can start a language server that is on `PATH` and attach its diagnostics to the results of file tools. The `lsp` feature is part of the `app` preset, and therefore also of `full`.

Caliche knows how to start these language servers:

| Language | Server | Command |
| --- | --- | --- |
| Rust | rust-analyzer | `rust-analyzer` |
| Python | Pyright | `pyright-langserver` |
| TypeScript and JavaScript | typescript-language-server | `typescript-language-server` |
| Go | gopls | `gopls` |
| C and C++ | clangd | `clangd` |

To build with language server support:

```bash
cargo build --features lsp
```

When `LspConfig.enabled` is `true`, a successful `fs` read, write, or edit, and a successful `apply_patch`, append the diagnostics that the language server reports for the file.

The following tools let the model query the language server directly. `enable_lsp_tools()` is only available when the `lsp` feature is enabled.

| Tool | Arguments |
| --- | --- |
| `lsp_diagnostics` | `{"path": "src/main.rs"}` |
| `lsp_symbols` | `{"query": "name", "path": "optional/path"}` |
| `lsp_hover` | `{"path": "src/main.rs", "line": 10, "column": 5}` |

```rust
use caliche::lsp::LspConfig;
use caliche::{AgentBuilder, ProviderType};

let mut lsp = LspConfig::default();
lsp.enabled = true;

let agent = AgentBuilder::new()
    .provider(ProviderType::OpenAI)
    .model("gpt-4o")
    .with_lsp_config(lsp)
    .enable_lsp_tools()
    .build()
    .await?;
```

No environment variable controls the language server settings. Set `LspConfig` in code, or set the `lsp` section of the CLI configuration file. See [Configuration](../configuration.md).

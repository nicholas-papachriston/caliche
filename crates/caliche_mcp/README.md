# caliche_mcp

`caliche_mcp` contains [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) types and helpers for Rust. It does not depend on the Caliche engine, so other hosts can reuse the same wire schema and the same list and call patterns.

## Modules

| Module | Contents |
| --- | --- |
| `protocol` | JSON-RPC and MCP wire types, and the supported protocol versions. |
| `revision` | Request admission for protocol `2026-07-28`: the `_meta` protocol version check, the protocol errors, and the cache fields on results. |
| `negotiate` | Parsing of `YYYY-MM-DD` protocol version strings into comparable dates. |
| `pagination` | Opaque cursor paging for `tools/list`, `prompts/list`, `resources/list` and `resources/templates/list`. |
| `providers` | The `ToolProvider`, `PromptProvider` and `ResourceProvider` traits that a host implements. |
| `router` | `ProviderRouter`, a JSON-RPC method router backed by the provider traits. |
| `cow_serde` | Serde helpers for `beef::lean::Cow` string fields. |

The crate has one feature, `server`, which is on by default and gates no code.

## Protocol support

The crate supports one protocol revision, `2026-07-28`. `SUPPORTED_PROTOCOL_VERSIONS` and `LATEST_PROTOCOL_VERSION` both name it. Every request except `server/discover` must carry that version in `_meta`, and a request with any other version gets an `UnsupportedProtocolVersionError`. The server does not accept `initialize`.

The optional fields `title`, `icons`, `outputSchema`, `structuredContent` and `_meta` are serialized when they are set. The Tasks extension, OAuth client registration, MCP Apps and cancelling a tool that has already started are not implemented. The [MCP integration guide](../../docs/integrations/mcp.md) describes the server's behavior in full.

## Example

```rust
use caliche_mcp::{LATEST_PROTOCOL_VERSION, paginate, parse_protocol_date};

assert_eq!(LATEST_PROTOCOL_VERSION, "2026-07-28");
assert_eq!(parse_protocol_date("2026-07-28"), Some(20_260_728));

let items = vec!["a", "b", "c"];
let page = paginate(&items, None, Some(2)).expect("page");
assert_eq!(page.items, ["a", "b"]);
assert!(page.next_cursor.is_some());
```

## Use in Caliche

`caliche::mcp` re-exports this crate and adapts Caliche's tool registry, templates and RAG collections to the provider traits.

## License

Licensed under either of [Apache-2.0](../../LICENSE-APACHE) or [MIT](../../LICENSE-MIT), at your option.

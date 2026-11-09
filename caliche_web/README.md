# caliche_web

`caliche_web` is the Caliche web UI, written with Dioxus. When the CLI is built with the `embedded-ui` feature, the HTTP server serves this UI at `/`, from the same process and address as the HTTP API. The crate also builds as a desktop app. See [HTTP sessions](../docs/sessions/sessions.md) and [Feature flags](../docs/reference/feature-flags.md#embedded-web-ui).

The UI streams chat replies, shows tool calls and token usage, renders Markdown, and can attach files from a file browser. Its panes cover sessions, agents and external agent runtimes, workflows, chains, reasoning, documents, plugins, metrics, and permissions. When the connection to the API drops, the UI reconnects on its own.

## Layout

| Path | Contents |
| --- | --- |
| `src/api/` | The HTTP client, the request and response types, and the browser WebSocket client. |
| `src/app/` | The root component, the reconnect loop, stream handling, and settings merging. |
| `src/commands/` | Slash command parsing. |
| `src/components/` | The views: chat, header, status bar, session panel, history, settings, permissions, and the other panes. |
| `src/harness/` | Presets and form checks for external agent runtimes. |
| `src/hooks/` | Dioxus hooks. |
| `src/state/` | Chat, configuration, agents, history, and streaming state. |
| `src/styles/` | The style sheets, written in Rust. |
| `src/context.rs` | Parsing of `@` references in the chat input. |

## Requirements

- Rust 1.92 or later.
- For a web build, the `wasm32-unknown-unknown` target: `rustup target add wasm32-unknown-unknown`.
- The Dioxus CLI (`dx`) for the development server. Install it with `cargo install dioxus-cli`.
- A running Caliche HTTP server. The UI connects to `http://localhost:3000` by default.

## Features

| Feature | Build |
| --- | --- |
| `desktop` | The desktop app. This is the default. |
| `web` | The browser build for `wasm32-unknown-unknown`. |
| `mobile` | The Dioxus mobile renderer. |
| `test-utils` | Exports the API client helpers that the tests in `tests/` use. |

## Running

Run the desktop app:

```bash
cargo run -p caliche_web
```

Build the browser version:

```bash
cargo build -p caliche_web --features web --target wasm32-unknown-unknown
```

Start a development server with hot reload, which `dx serve` turns on by default:

```bash
cd caliche_web
dx serve --platform web
```

Use `--platform desktop` for the desktop app.

## Configuration

The API address is `http://localhost:3000` unless you change it in the settings view. When the server requires a JWT (`JWT_ENABLED` and `CALICHE_REQUIRE_AUTH`), enter the bearer token in the settings view.

The file browser lists and reads files on the server, and the UI sends the selected files as context with your message. Turn it on in the server's environment:

```bash
export CALICHE_FILE_BROWSING=true
export CALICHE_WORKSPACE=/path/to/your/project
```

`CALICHE_WORKSPACE` defaults to the server's working directory, and the server refuses paths outside it. When file browsing is on, the chat input shows a `[files]` button.

## API routes

The UI uses these routes:

| Route | Method | Purpose |
| --- | --- | --- |
| `/health` | `GET` | Checks that the server is running. |
| `/api/models` | `GET` | Lists providers and models, and whether each is configured or local. |
| `/api/chat/stream` | `POST` | Streams a chat reply as server-sent events. |
| `/api/files/enabled` | `GET` | Reports whether file browsing is on. |
| `/api/files/list` | `GET` | Lists a directory. |
| `/api/files/read` | `GET` | Reads a file. |

A chat request carries the earlier messages and, when set, the tool allowlist:

```json
{
  "message": "Hello!",
  "provider": "openai",
  "model": "gpt-4o",
  "messages": [
    { "role": "user", "content": "Earlier question" },
    { "role": "assistant", "content": "Earlier answer" }
  ],
  "tools": ["fs", "bash"]
}
```

Each frame of the reply has an `event:` line with the event name, such as `text_chunk`, `tool.start`, `tool.end`, or `turn.complete`, and a `data:` line with a JSON envelope. The envelope has `version`, `type`, and `payload` fields.

## Testing

The tests live in `tests/` and next to the modules they cover. Run them with the `test-utils` feature:

```bash
cargo test -p caliche_web --features test-utils
```

These tests check chat streaming over the browser socket without a development server:

```bash
cargo test -p caliche_web --features test-utils --lib http_session_is_adopted
cargo test -p caliche_web --features test-utils --lib second_send_retargets
cargo test -p caliche_web --features test-utils --lib stop_generation_clears
```

To check streaming by hand, start a server with the UI built in (`--features embedded-ui`, or `./caliche rust install`):

```bash
caliche --profile lean --server --bind 127.0.0.1 --port 3080
```

Open `http://127.0.0.1:3080`, then:

1. Send a message. The agent row shows `[streaming...]` and then the reply.
2. Send a second message in the same session. The second agent row shows its reply.
3. Send another message and press `[stop]` while it streams. The row shows `[Generation stopped by user]`, and the chat input accepts a new message.

## License

Licensed under either of [Apache-2.0](../LICENSE-APACHE) or [MIT](../LICENSE-MIT), at your option.

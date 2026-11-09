# Logging and observability

Caliche writes runtime logs with the `tracing` crate. Depending on how you run it, it can also report token usage, cost, and request metadata, and it can export traces from the HTTP server over OpenTelemetry.

## Logging from Rust

Initialize logging with human-readable output:

```rust
use caliche::observability::init_logging;

init_logging(false)?;
```

When another system reads the logs, use JSON output and set a level:

```rust
use caliche::observability::init_logging_with_level;
use tracing::Level;

init_logging_with_level(true, Some(Level::INFO))?;
```

## Forwarding log entries

To receive each log entry in your own code, register a callback before you initialize logging:

```rust
use caliche::observability::{set_log_callback, LogEntry};
use std::sync::Arc;

let callback = Arc::new(|entry: &LogEntry| {
    println!("[{}] {}: {}", entry.level, entry.target, entry.message);
});

set_log_callback(callback);
init_logging(false)?;
```

## Log levels

- `trace`: detailed runtime debugging.
- `debug`: tool arguments and other development details.
- `info`: normal runtime operations.
- `warn`: recoverable problems and invalid tool calls.
- `error`: failed operations.

## Setting the level from the environment

```bash
RUST_LOG=info caliche
RUST_LOG=caliche=debug,reqwest=warn caliche
```

## Tracing the HTTP server

To export the spans of the API server through OpenTelemetry, build with the `otlp` feature and set `OTEL_EXPORTER_OTLP_ENDPOINT`. The exporter is set up in `caliche_server`.

```bash
cargo run -p caliche_cli --features otlp -- --server
```

When the feature is not enabled, or the endpoint is not set, traces stay inside the process and OpenTelemetry uses a provider that does nothing.

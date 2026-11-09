//! CLI output helpers for non-interactive runs.

pub mod stream_format;

pub use stream_format::{eprint_multimodal_banner, process_oneshot_stream, run_multimodal_oneshot};

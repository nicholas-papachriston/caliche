//! Streaming utilities for LLM providers
//!
//! Provides zero-copy, allocation-efficient abstractions for
//! processing Server-Sent Events (SSE) and streaming responses.

mod byte_line_buffer;
mod line_buffer;

pub use byte_line_buffer::ByteLineBuffer;
pub use line_buffer::LineBuffer;

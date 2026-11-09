//! Efficient containers for memory management
//!
//! Provides specialized data structures optimized for
//! conversation history and message buffer management.

mod ring_buffer;

pub use ring_buffer::RingBuffer;

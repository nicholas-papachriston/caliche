//! Durable event streams for replay, resume, and cross-node observability.
//!
//! Events can be persisted to Redis/Valkey Streams, an in-memory ring buffer,
//! or a memory-mapped append log (`STRMV001` + length-prefixed rkyv records when
//! `durable-streams-mmap` is enabled). Consumers can replay from a position or
//! subscribe to history then live events.

mod backend_trait;
#[cfg(feature = "durable-streams-mmap")]
mod codec;
mod memory;
#[cfg(feature = "durable-streams-mmap")]
mod mmap;
#[cfg(feature = "durable-streams-postgres")]
mod postgres;
mod reader;
#[cfg(feature = "durable-streams-redis")]
mod redis;
#[cfg(feature = "durable-streams-s3")]
mod s3;
#[cfg(feature = "durable-streams-sqlite")]
mod sqlite;
mod types;
mod writer;

pub use backend_trait::DurableStreamBackend;
pub use memory::MemoryDurableStreamBackend;
#[cfg(feature = "durable-streams-mmap")]
pub use mmap::MmapDurableStreamBackend;
#[cfg(feature = "durable-streams-postgres")]
pub use postgres::PostgresDurableStreamBackend;
pub use reader::DurableStreamReader;
#[cfg(feature = "durable-streams-redis")]
pub use redis::RedisDurableStreamBackend;
#[cfg(feature = "durable-streams-s3")]
pub use s3::{S3DurableStreamBackend, S3DurableStreamConfig};
#[cfg(feature = "durable-streams-sqlite")]
pub use sqlite::SqliteDurableStreamBackend;
pub use types::{
    DurableEvent, DurableEventMetadata, DurableEventPayload, STREAM_START, TrimPolicy,
};
pub use writer::DurableStreamWriter;

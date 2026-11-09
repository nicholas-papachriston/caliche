//! DynamoDB native vector search backend.
//!
//! Uses DynamoDB vector indexes (`SearchVectors`) for server-side ANN while storing
//! full [`crate::core::Chunk`] envelopes on table items. Implements [`crate::traits::Backend`]
//! so it works standalone, inside [`crate::backend::tiered::TieredChainBuilder`], and with
//! decorators such as [`crate::backend::indexed::IndexedBackend`] and
//! [`crate::backend::late_interaction::LateInteractionIndexedBackend`].
//!
//! # Item schema
//!
//! | Attribute | Type | Role |
//! |-----------|------|------|
//! | `pk` | S | Namespace (vector-index HASH) |
//! | `sk` | S | Chunk id |
//! | `embedding` | L\<N\> | Dense vector |
//! | `envelope` | S | Chunk JSON envelope |
//! | `doc_id` | S | Document id (INLINE_FILTER) |
//! | `multi_vectors` | S | Optional MaxSim token vectors JSON |
//!
//! # Feature
//!
//! Enable with `dynamodb-backend`.

mod attrs;
mod backend;
mod client;
mod config;

pub use backend::DynamoDbVectorBackend;
pub use config::DynamoDbVectorConfig;

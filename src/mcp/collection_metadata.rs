//! Document collection metadata type for MCP resources.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Document collection metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionMetadata {
    pub name: String,
    pub document_count: usize,
    pub created_at: DateTime<Utc>,
}

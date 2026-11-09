//! Configuration for the DynamoDB native vector backend.

use crate::core::Metric;

/// Configuration for [`super::DynamoDbVectorBackend`].
#[derive(Debug, Clone)]
pub struct DynamoDbVectorConfig {
    /// DynamoDB table name (must already exist, or set `ensure_table`).
    pub table_name: String,
    /// Vector index name on the table.
    pub index_name: String,
    /// Embedding dimensionality (must match the vector index).
    pub dimension: usize,
    /// Distance metric configured on the vector index.
    pub distance_metric: Metric,
    /// AWS region (default credentials chain / env when unset).
    pub region: Option<String>,
    /// Optional custom endpoint (LocalStack / DynamoDB local).
    ///
    /// When set, `SearchVectors` uses this host too. Live AWS leaves this
    /// unset and calls `https://search-dynamodb.{region}.api.aws`.
    pub endpoint_url: Option<String>,
    /// Optional override for the `SearchVectors` host.
    pub search_endpoint_url: Option<String>,
    /// Explicit access key (optional; default chain otherwise).
    pub access_key_id: Option<String>,
    /// Explicit secret key (optional; default chain otherwise).
    pub secret_access_key: Option<String>,
    /// Optional session token.
    pub session_token: Option<String>,
    /// Partition-key attribute name (default `pk` = namespace).
    pub pk_attribute: String,
    /// Sort-key attribute name (default `sk` = chunk id).
    pub sk_attribute: String,
    /// Vector attribute name (default `embedding`).
    pub vector_attribute: String,
    /// Create the table (+ vector index) when missing.
    pub ensure_table: bool,
    /// Create the vector index when missing on an existing table.
    pub ensure_index: bool,
    /// Batch size for `BatchWriteItem` (max 25).
    pub batch_size: usize,
    /// Hard cap for DynamoDB `TopK` (service max is 100).
    pub top_k_cap: usize,
    /// Oversample factor when client-side metadata filters are applied.
    pub filter_oversample_factor: usize,
}

impl DynamoDbVectorConfig {
    /// Build a config for an existing table/index.
    pub fn new(
        table_name: impl Into<String>,
        index_name: impl Into<String>,
        dimension: usize,
    ) -> Self {
        Self {
            table_name: table_name.into(),
            index_name: index_name.into(),
            dimension,
            distance_metric: Metric::Cosine,
            region: None,
            endpoint_url: None,
            search_endpoint_url: None,
            access_key_id: None,
            secret_access_key: None,
            session_token: None,
            pk_attribute: "pk".to_string(),
            sk_attribute: "sk".to_string(),
            vector_attribute: "embedding".to_string(),
            ensure_table: false,
            ensure_index: false,
            batch_size: 25,
            top_k_cap: 100,
            filter_oversample_factor: 4,
        }
    }

    #[must_use]
    pub const fn with_distance_metric(mut self, metric: Metric) -> Self {
        self.distance_metric = metric;
        self
    }

    #[must_use]
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    #[must_use]
    pub fn with_endpoint_url(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint_url = Some(endpoint.into());
        self
    }

    #[must_use]
    pub fn with_search_endpoint_url(mut self, endpoint: impl Into<String>) -> Self {
        self.search_endpoint_url = Some(endpoint.into());
        self
    }

    #[must_use]
    pub fn with_credentials(
        mut self,
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
        session_token: Option<String>,
    ) -> Self {
        self.access_key_id = Some(access_key_id.into());
        self.secret_access_key = Some(secret_access_key.into());
        self.session_token = session_token;
        self
    }

    #[must_use]
    pub const fn with_ensure_table(mut self, enabled: bool) -> Self {
        self.ensure_table = enabled;
        self
    }

    #[must_use]
    pub const fn with_ensure_index(mut self, enabled: bool) -> Self {
        self.ensure_index = enabled;
        self
    }

    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.clamp(1, 25);
        self
    }

    #[must_use]
    pub fn with_top_k_cap(mut self, top_k_cap: usize) -> Self {
        self.top_k_cap = top_k_cap.clamp(1, 100);
        self
    }

    #[must_use]
    pub fn with_filter_oversample_factor(mut self, factor: usize) -> Self {
        self.filter_oversample_factor = factor.max(1);
        self
    }
}

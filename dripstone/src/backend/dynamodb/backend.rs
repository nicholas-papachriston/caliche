//! DynamoDB native vector search backend (`SearchVectors`).

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use super::attrs::{
    ATTR_DOC_ID, ATTR_EMBEDDING, ATTR_PK, ATTR_SK, AttributeMap, attribute_map_from_value,
    attribute_map_to_value, chunk_to_item, distance_function_name, encode_search_vector,
    item_filter_metadata, item_to_chunk, map_dynamodb_score, namespace_key,
};
use super::client::DynamoDbJsonClient;
use super::config::DynamoDbVectorConfig;
use crate::core::{Capabilities, Chunk, Metric, SearchQuery, SearchResult};
use crate::errors::DripstoneError;
use crate::search::Filter;
use crate::traits::Backend;

/// Native DynamoDB vector-index backend.
///
/// Standalone `Backend` using PutItem / GetItem / `SearchVectors`. Works with
/// [`crate::backend::tiered::TieredChainBuilder`], [`crate::backend::indexed::IndexedBackend`],
/// [`crate::backend::late_interaction::LateInteractionIndexedBackend`], and
/// [`crate::traits::search_with_maxsim_fallback`] (`native_maxsim: false`).
#[derive(Debug, Clone)]
pub struct DynamoDbVectorBackend {
    client: Arc<DynamoDbJsonClient>,
    config: DynamoDbVectorConfig,
}

impl DynamoDbVectorBackend {
    /// Create a backend using the default AWS credentials chain.
    ///
    /// # Errors
    ///
    /// Returns when credentials/config cannot be loaded or table/index ensure fails.
    pub async fn new(config: DynamoDbVectorConfig) -> Result<Self, DripstoneError> {
        distance_function_name(config.distance_metric)?;
        let search_endpoint = config
            .search_endpoint_url
            .clone()
            .or_else(|| std::env::var("DRIPSTONE_DYNAMODB_SEARCH_ENDPOINT").ok());
        let client = DynamoDbJsonClient::connect(
            config.region.as_deref(),
            config.endpoint_url.as_deref(),
            search_endpoint.as_deref(),
            config.access_key_id.as_deref(),
            config.secret_access_key.as_deref(),
            config.session_token.as_deref(),
        )
        .await?;
        Self::with_client(config, Arc::new(client)).await
    }

    /// Create with an existing JSON client (internal / tests).
    ///
    /// # Errors
    ///
    /// Returns when optional table/index ensure fails.
    pub(crate) async fn with_client(
        config: DynamoDbVectorConfig,
        client: Arc<DynamoDbJsonClient>,
    ) -> Result<Self, DripstoneError> {
        distance_function_name(config.distance_metric)?;
        let backend = Self { client, config };
        if backend.config.ensure_table {
            backend.ensure_table().await?;
        } else if backend.config.ensure_index {
            backend.ensure_vector_index().await?;
        }
        Ok(backend)
    }

    #[must_use]
    pub const fn config(&self) -> &DynamoDbVectorConfig {
        &self.config
    }

    fn vector_index_definition(&self) -> Result<Value, DripstoneError> {
        Ok(json!({
            "IndexName": self.config.index_name,
            "VectorAttribute": {
                "AttributeName": self.config.vector_attribute
            },
            "SearchSchema": [
                {
                    "AttributeName": self.config.pk_attribute,
                    "SearchSchemaElementType": "HASH"
                },
                {
                    "AttributeName": ATTR_DOC_ID,
                    "SearchSchemaElementType": "INLINE_FILTER"
                }
            ],
            "Projection": { "ProjectionType": "ALL" },
            "Dimensions": self.config.dimension,
            "DistanceFunction": distance_function_name(self.config.distance_metric)?
        }))
    }

    async fn ensure_table(&self) -> Result<(), DripstoneError> {
        match self.describe_table().await {
            Ok(_) => {
                if self.config.ensure_index {
                    self.ensure_vector_index().await?;
                }
                Ok(())
            }
            Err(err) if err.is_not_found() => {
                let body = json!({
                    "TableName": self.config.table_name,
                    "BillingMode": "PAY_PER_REQUEST",
                    "AttributeDefinitions": [
                        { "AttributeName": self.config.pk_attribute, "AttributeType": "S" },
                        { "AttributeName": self.config.sk_attribute, "AttributeType": "S" }
                    ],
                    "KeySchema": [
                        { "AttributeName": self.config.pk_attribute, "KeyType": "HASH" },
                        { "AttributeName": self.config.sk_attribute, "KeyType": "RANGE" }
                    ],
                    "VectorIndexes": [self.vector_index_definition()?]
                });
                self.client.call("CreateTable", &body).await?;
                self.wait_for_table_active().await?;
                self.wait_for_index_active().await
            }
            Err(err) => Err(err),
        }
    }

    async fn ensure_vector_index(&self) -> Result<(), DripstoneError> {
        let description = self.describe_table().await?;
        let exists = description
            .pointer("/Table/VectorIndexes")
            .and_then(Value::as_array)
            .is_some_and(|indexes| {
                indexes.iter().any(|idx| {
                    idx.get("IndexName").and_then(Value::as_str)
                        == Some(self.config.index_name.as_str())
                })
            });
        if exists {
            return Ok(());
        }
        let body = json!({
            "TableName": self.config.table_name,
            "VectorIndexUpdates": [{
                "Create": self.vector_index_definition()?
            }]
        });
        self.client.call("UpdateTable", &body).await?;
        self.wait_for_index_active().await
    }

    async fn describe_table(&self) -> Result<Value, DripstoneError> {
        self.client
            .call(
                "DescribeTable",
                &json!({ "TableName": self.config.table_name }),
            )
            .await
    }

    async fn wait_for_table_active(&self) -> Result<(), DripstoneError> {
        for _ in 0..60 {
            let description = self.describe_table().await?;
            if description
                .pointer("/Table/TableStatus")
                .and_then(Value::as_str)
                == Some("ACTIVE")
            {
                return Ok(());
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
        Err(DripstoneError::storage(
            "timed out waiting for DynamoDB table to become ACTIVE",
        ))
    }

    async fn wait_for_index_active(&self) -> Result<(), DripstoneError> {
        for _ in 0..120 {
            let description = self.describe_table().await?;
            let active = description
                .pointer("/Table/VectorIndexes")
                .and_then(Value::as_array)
                .is_some_and(|indexes| {
                    indexes.iter().any(|idx| {
                        idx.get("IndexName").and_then(Value::as_str)
                            == Some(self.config.index_name.as_str())
                            && idx.get("IndexStatus").and_then(Value::as_str) == Some("ACTIVE")
                    })
                });
            if active {
                return Ok(());
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
        Err(DripstoneError::storage(
            "timed out waiting for DynamoDB vector index to become ACTIVE",
        ))
    }

    fn insert_field(body: &mut Value, key: &str, value: Value) -> Result<(), DripstoneError> {
        let obj = body.as_object_mut().ok_or_else(|| {
            DripstoneError::storage("DynamoDB request body must be a JSON object")
        })?;
        obj.insert(key.to_string(), value);
        Ok(())
    }

    fn key_for(&self, id: &str, namespace: Option<&str>) -> AttributeMap {
        let mut key = AttributeMap::new();
        key.insert(
            self.config.pk_attribute.clone(),
            json!({ "S": namespace_key(namespace) }),
        );
        key.insert(self.config.sk_attribute.clone(), json!({ "S": id }));
        key
    }

    fn remap_item_keys(&self, item: AttributeMap) -> AttributeMap {
        let mut out = item;
        if self.config.pk_attribute != ATTR_PK
            && let Some(v) = out.remove(&self.config.pk_attribute)
        {
            out.insert(ATTR_PK.to_string(), v);
        }
        if self.config.sk_attribute != ATTR_SK
            && let Some(v) = out.remove(&self.config.sk_attribute)
        {
            out.insert(ATTR_SK.to_string(), v);
        }
        if self.config.vector_attribute != ATTR_EMBEDDING
            && let Some(v) = out.remove(&self.config.vector_attribute)
        {
            out.insert(ATTR_EMBEDDING.to_string(), v);
        }
        out
    }

    fn prepare_write_item(&self, chunk: &Chunk) -> Result<AttributeMap, DripstoneError> {
        let mut item = chunk_to_item(chunk, self.config.dimension)?;
        if self.config.pk_attribute != ATTR_PK
            && let Some(v) = item.remove(ATTR_PK)
        {
            item.insert(self.config.pk_attribute.clone(), v);
        }
        if self.config.sk_attribute != ATTR_SK
            && let Some(v) = item.remove(ATTR_SK)
        {
            item.insert(self.config.sk_attribute.clone(), v);
        }
        if self.config.vector_attribute != ATTR_EMBEDDING
            && let Some(v) = item.remove(ATTR_EMBEDDING)
        {
            item.insert(self.config.vector_attribute.clone(), v);
        }
        Ok(item)
    }

    fn requested_top_k(&self, limit: usize, has_filter: bool) -> usize {
        let base = limit.max(1);
        let oversampled = if has_filter {
            base.saturating_mul(self.config.filter_oversample_factor.max(1))
        } else {
            base
        };
        oversampled.min(self.config.top_k_cap.max(1)).min(100)
    }

    fn sort_results(metric: Metric, results: &mut [SearchResult]) {
        if metric.higher_is_better() {
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            results.sort_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i;
        }
    }

    async fn query_namespace_keys(
        &self,
        namespace: &str,
    ) -> Result<Vec<AttributeMap>, DripstoneError> {
        let mut keys = Vec::new();
        let mut exclusive_start_key = None::<Value>;
        loop {
            let mut body = json!({
                "TableName": self.config.table_name,
                "KeyConditionExpression": "#pk = :ns",
                "ExpressionAttributeNames": {
                    "#pk": self.config.pk_attribute,
                    "#sk": self.config.sk_attribute
                },
                "ExpressionAttributeValues": {
                    ":ns": { "S": namespace }
                },
                "ProjectionExpression": "#pk, #sk"
            });
            if let Some(start) = exclusive_start_key.take() {
                Self::insert_field(&mut body, "ExclusiveStartKey", start)?;
            }
            let output = self.client.call("Query", &body).await?;
            if let Some(items) = output.get("Items").and_then(Value::as_array) {
                for item in items {
                    keys.push(attribute_map_from_value(item)?);
                }
            }
            match output.get("LastEvaluatedKey") {
                Some(last) if !last.is_null() => exclusive_start_key = Some(last.clone()),
                _ => break,
            }
        }
        Ok(keys)
    }

    async fn scan_items(&self, limit: Option<usize>) -> Result<Vec<AttributeMap>, DripstoneError> {
        let mut items = Vec::new();
        let mut exclusive_start_key = None::<Value>;
        loop {
            let mut body = json!({ "TableName": self.config.table_name });
            if let Some(start) = exclusive_start_key.take() {
                Self::insert_field(&mut body, "ExclusiveStartKey", start)?;
            }
            let output = self.client.call("Scan", &body).await?;
            if let Some(batch) = output.get("Items").and_then(Value::as_array) {
                for item in batch {
                    items.push(attribute_map_from_value(item)?);
                    if limit.is_some_and(|n| items.len() >= n) {
                        return Ok(items);
                    }
                }
            }
            match output.get("LastEvaluatedKey") {
                Some(last) if !last.is_null() => exclusive_start_key = Some(last.clone()),
                _ => break,
            }
        }
        Ok(items)
    }
}

#[async_trait]
impl Backend for DynamoDbVectorBackend {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            native_hybrid: false,
            native_full_text: false,
            native_hybrid_full_text: false,
            native_filter: true,
            requires_exact_filter_pass: true,
            supports_namespaces: true,
            native_maxsim: false,
            max_dimension: Some(4096),
            max_batch: Some(self.config.batch_size.min(25)),
            supports_streaming: false,
        }
    }

    async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        if chunks.is_empty() {
            return Ok(());
        }
        let batch_size = self.config.batch_size.clamp(1, 25);
        for batch in chunks.chunks(batch_size) {
            let requests: Vec<Value> = batch
                .iter()
                .map(|chunk| {
                    let item = self.prepare_write_item(chunk)?;
                    Ok(json!({
                        "PutRequest": { "Item": attribute_map_to_value(&item) }
                    }))
                })
                .collect::<Result<_, DripstoneError>>()?;
            let body = json!({
                "RequestItems": {
                    self.config.table_name.clone(): requests
                }
            });
            self.client.call("BatchWriteItem", &body).await?;
        }
        Ok(())
    }

    async fn get(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        let body = json!({
            "TableName": self.config.table_name,
            "Key": attribute_map_to_value(&self.key_for(id, namespace))
        });
        let output = self.client.call("GetItem", &body).await?;
        match output.get("Item") {
            Some(item) if !item.is_null() => {
                let mapped = self.remap_item_keys(attribute_map_from_value(item)?);
                Ok(Some(item_to_chunk(&mapped)?))
            }
            _ => Ok(None),
        }
    }

    async fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
        let body = json!({
            "TableName": self.config.table_name,
            "Key": attribute_map_to_value(&self.key_for(id, namespace))
        });
        self.client.call("DeleteItem", &body).await?;
        Ok(())
    }

    async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        let ns = namespace_key(namespace);
        let mut deleted = 0usize;
        let mut exclusive_start_key = None::<Value>;
        loop {
            let mut body = json!({
                "TableName": self.config.table_name,
                "KeyConditionExpression": "#pk = :ns",
                "FilterExpression": "#doc = :doc",
                "ExpressionAttributeNames": {
                    "#pk": self.config.pk_attribute,
                    "#sk": self.config.sk_attribute,
                    "#doc": ATTR_DOC_ID
                },
                "ExpressionAttributeValues": {
                    ":ns": { "S": ns },
                    ":doc": { "S": doc_id }
                },
                "ProjectionExpression": "#pk, #sk"
            });
            if let Some(start) = exclusive_start_key.take() {
                Self::insert_field(&mut body, "ExclusiveStartKey", start)?;
            }
            let output = self.client.call("Query", &body).await?;
            if let Some(items) = output.get("Items").and_then(Value::as_array) {
                for item in items {
                    let del = json!({
                        "TableName": self.config.table_name,
                        "Key": item
                    });
                    self.client.call("DeleteItem", &del).await?;
                    deleted += 1;
                }
            }
            match output.get("LastEvaluatedKey") {
                Some(last) if !last.is_null() => exclusive_start_key = Some(last.clone()),
                _ => break,
            }
        }
        Ok(deleted)
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        if query.vector.len() != self.config.dimension {
            return Err(DripstoneError::storage(format!(
                "query vector dimension mismatch: expected {}, got {}",
                self.config.dimension,
                query.vector.len()
            )));
        }
        if query.metric != self.config.distance_metric {
            return Err(DripstoneError::storage(format!(
                "query metric {:?} does not match DynamoDB index metric {:?}",
                query.metric, self.config.distance_metric
            )));
        }

        let has_filter = query.filter.is_some();
        let top_k = self.requested_top_k(query.limit, has_filter);
        let ns = namespace_key(query.namespace.as_deref());

        let mut names = json!({ "#pk": self.config.pk_attribute });
        let mut values = json!({ ":ns": { "S": ns } });
        let mut condition = "#pk = :ns".to_string();

        if let Some(Filter::Equal { field, value }) = query.filter.as_ref()
            && field == ATTR_DOC_ID
            && let Some(doc_id) = value.as_str()
        {
            Self::insert_field(&mut names, "#doc", json!(ATTR_DOC_ID))?;
            Self::insert_field(&mut values, ":doc", json!({ "S": doc_id }))?;
            condition.push_str(" AND #doc = :doc");
        }

        let body = json!({
            "TableName": self.config.table_name,
            "IndexName": self.config.index_name,
            "SearchVector": encode_search_vector(&query.vector),
            "TopK": top_k,
            "SearchConditionExpression": condition,
            "ExpressionAttributeNames": names,
            "ExpressionAttributeValues": values
        });
        let output = self.client.call("SearchVectors", &body).await?;

        let mut results = Vec::new();
        let hits = output
            .get("SearchResults")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for (rank, hit) in hits.iter().enumerate() {
            let Some(item_value) = hit.get("Item") else {
                continue;
            };
            let mapped = self.remap_item_keys(attribute_map_from_value(item_value)?);
            if let Some(filter) = query.filter.as_ref() {
                let meta = item_filter_metadata(&mapped);
                if !filter.matches(&meta) {
                    continue;
                }
            }
            let chunk = item_to_chunk(&mapped)?;
            let score = hit.get("Score").and_then(Value::as_f64).unwrap_or(0.0);
            results.push(SearchResult::new(
                chunk,
                map_dynamodb_score(query.metric, score),
                rank,
            ));
            if results.len() >= query.limit {
                break;
            }
        }

        Self::sort_results(query.metric, &mut results);
        results.truncate(query.limit);
        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i;
        }
        Ok(results)
    }

    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        let items = self.scan_items(None).await?;
        let mut set = HashSet::new();
        for item in items {
            let mapped = self.remap_item_keys(item);
            if let Some(ns) = mapped
                .get(ATTR_PK)
                .and_then(|v| v.get("S"))
                .and_then(Value::as_str)
            {
                set.insert(ns.to_string());
            }
        }
        let mut out: Vec<String> = set.into_iter().collect();
        out.sort();
        Ok(out)
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        let keys = self.query_namespace_keys(namespace).await?;
        let batch_size = self.config.batch_size.clamp(1, 25);
        for batch in keys.chunks(batch_size) {
            let requests: Vec<Value> = batch
                .iter()
                .map(|key| {
                    json!({
                        "DeleteRequest": { "Key": attribute_map_to_value(key) }
                    })
                })
                .collect();
            if requests.is_empty() {
                continue;
            }
            let body = json!({
                "RequestItems": {
                    self.config.table_name.clone(): requests
                }
            });
            self.client.call("BatchWriteItem", &body).await?;
        }
        Ok(())
    }

    async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError> {
        if let Some(ns) = namespace {
            let mut total = 0usize;
            let mut exclusive_start_key = None::<Value>;
            loop {
                let mut body = json!({
                    "TableName": self.config.table_name,
                    "KeyConditionExpression": "#pk = :ns",
                    "ExpressionAttributeNames": { "#pk": self.config.pk_attribute },
                    "ExpressionAttributeValues": { ":ns": { "S": ns } },
                    "Select": "COUNT"
                });
                if let Some(start) = exclusive_start_key.take() {
                    Self::insert_field(&mut body, "ExclusiveStartKey", start)?;
                }
                let output = self.client.call("Query", &body).await?;
                total += output.get("Count").and_then(Value::as_u64).unwrap_or(0) as usize;
                match output.get("LastEvaluatedKey") {
                    Some(last) if !last.is_null() => exclusive_start_key = Some(last.clone()),
                    _ => break,
                }
            }
            return Ok(total);
        }

        let mut total = 0usize;
        let mut exclusive_start_key = None::<Value>;
        loop {
            let mut body = json!({
                "TableName": self.config.table_name,
                "Select": "COUNT"
            });
            if let Some(start) = exclusive_start_key.take() {
                Self::insert_field(&mut body, "ExclusiveStartKey", start)?;
            }
            let output = self.client.call("Scan", &body).await?;
            total += output.get("Count").and_then(Value::as_u64).unwrap_or(0) as usize;
            match output.get("LastEvaluatedKey") {
                Some(last) if !last.is_null() => exclusive_start_key = Some(last.clone()),
                _ => break,
            }
        }
        Ok(total)
    }

    async fn list(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Chunk>, DripstoneError> {
        let mut chunks = Vec::new();
        if let Some(ns) = namespace {
            let mut exclusive_start_key = None::<Value>;
            loop {
                let mut body = json!({
                    "TableName": self.config.table_name,
                    "KeyConditionExpression": "#pk = :ns",
                    "ExpressionAttributeNames": { "#pk": self.config.pk_attribute },
                    "ExpressionAttributeValues": { ":ns": { "S": ns } }
                });
                if let Some(start) = exclusive_start_key.take() {
                    Self::insert_field(&mut body, "ExclusiveStartKey", start)?;
                }
                let output = self.client.call("Query", &body).await?;
                if let Some(items) = output.get("Items").and_then(Value::as_array) {
                    for item in items {
                        let mapped = self.remap_item_keys(attribute_map_from_value(item)?);
                        chunks.push(item_to_chunk(&mapped)?);
                        if limit.is_some_and(|n| chunks.len() >= n) {
                            return Ok(chunks);
                        }
                    }
                }
                match output.get("LastEvaluatedKey") {
                    Some(last) if !last.is_null() => exclusive_start_key = Some(last.clone()),
                    _ => break,
                }
            }
            return Ok(chunks);
        }

        for item in self.scan_items(limit).await? {
            let mapped = self.remap_item_keys(item);
            chunks.push(item_to_chunk(&mapped)?);
        }
        Ok(chunks)
    }
}

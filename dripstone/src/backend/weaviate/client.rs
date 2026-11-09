use super::convert::{
    CREATED_KEY, DEFAULT_BATCH_SIZE, DEFAULT_CLASS_PREFIX, UPDATED_KEY, build_weaviate_headers,
    encode_metadata, extract_record, generate_weaviate_uuid, similarity_metric_to_weaviate,
};
use serde_json::{Map as JsonMap, Value, json};
use std::time::Duration;

use crate::backend::http::HttpClient;
use crate::embedding::{EmbeddingRecord, SimilarityMetric};
use crate::errors::DripstoneError;

pub struct WeaviateEmbeddingBackend {
    pub(crate) http: HttpClient,
    pub(crate) class_prefix: String,
    pub(crate) vector_dimension: usize,
    pub(crate) batch_size: usize,
}

#[derive(Debug)]
pub struct WeaviateBackendOptions {
    pub base_url: String,
    pub class_prefix: Option<String>,
    pub vector_dimension: usize,
    pub api_key: Option<String>,
    pub timeout: Option<Duration>,
    pub batch_size: Option<usize>,
    pub retry_config: Option<crate::backend::http::RetryConfig>,
}

impl WeaviateBackendOptions {
    pub fn new(base_url: impl Into<String>, vector_dimension: usize) -> Self {
        Self {
            base_url: base_url.into(),
            class_prefix: None,
            vector_dimension,
            api_key: None,
            timeout: None,
            batch_size: None,
            retry_config: None,
        }
    }
}

impl WeaviateEmbeddingBackend {
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn new(options: WeaviateBackendOptions) -> Result<Self, DripstoneError> {
        if options.vector_dimension == 0 {
            return Err(DripstoneError::Misconfiguration(
                "weaviate vector dimension must be greater than zero".to_string(),
            ));
        }

        let headers = options
            .api_key
            .as_ref()
            .map(|key| build_weaviate_headers(key))
            .transpose()?;

        let mut http_config = crate::backend::http::HttpClientConfig {
            timeout: options.timeout,
            default_headers: headers,
            ..Default::default()
        };
        if let Some(retry) = options.retry_config {
            http_config.retry_config = Some(retry);
        }
        let http = HttpClient::with_config(options.base_url, http_config)?;
        let prefix = options
            .class_prefix
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| DEFAULT_CLASS_PREFIX.to_string());
        let batch_size = options.batch_size.unwrap_or(DEFAULT_BATCH_SIZE).max(1);
        let backend = Self {
            http,
            class_prefix: prefix,
            vector_dimension: options.vector_dimension,
            batch_size,
        };
        backend.health_check().await?;
        Ok(backend)
    }

    pub(crate) fn class_name(&self, namespace: &str) -> String {
        // Weaviate class names must start with uppercase and be CamelCase
        let ns = namespace.trim();
        if ns.is_empty() {
            let prefix = self.normalized_class_prefix();
            return format!("{prefix}Default");
        }
        // Convert namespace to CamelCase
        let camel_case: String = ns
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| {
                let mut chars = s.chars();
                chars.next().map_or_else(String::new, |first| {
                    first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase()
                })
            })
            .collect();
        let prefix = self.normalized_class_prefix();
        format!("{prefix}{camel_case}")
    }

    fn normalized_class_prefix(&self) -> String {
        self.class_prefix.chars().next().map_or_else(
            || self.class_prefix.clone(),
            |first| {
                if first.is_uppercase() {
                    self.class_prefix.clone()
                } else {
                    first.to_uppercase().collect::<String>()
                        + &self.class_prefix[first.len_utf8()..]
                }
            },
        )
    }

    pub(crate) async fn health_check(&self) -> Result<(), DripstoneError> {
        let response = self.http.get("/v1/.well-known/ready").await?;
        if response.status().is_success() {
            return Ok(());
        }
        Err(DripstoneError::storage(format!(
            "weaviate health check failed with status {}",
            response.status()
        )))
    }

    pub(crate) async fn ensure_class(&self, namespace: &str) -> Result<(), DripstoneError> {
        let class_name = self.class_name(namespace);
        let schema_path = format!("/v1/schema/{class_name}");
        let class_exists = match self.http.get(schema_path.as_str()).await {
            Ok(response) => {
                if response.status() == reqwest::StatusCode::OK {
                    return Ok(());
                }
                if response.status() != reqwest::StatusCode::NOT_FOUND {
                    return Err(DripstoneError::storage(format!(
                        "failed to check weaviate class {}: status {}",
                        class_name,
                        response.status()
                    )));
                }
                // 404 - class doesn't exist
                false
            }
            Err(e) => {
                // Check if the error is a 404 (class doesn't exist)
                // The HTTP client wraps 404s as permanent errors
                let error_msg = format!("{e}");
                if error_msg.contains("404")
                    || error_msg.contains("Not found")
                    || error_msg.contains("doesn't exist")
                {
                    // Class doesn't exist
                    false
                } else {
                    return Err(e);
                }
            }
        };

        if class_exists {
            return Ok(());
        }

        // Create the class schema
        // Note: 'id' is a reserved property in Weaviate, so we don't include it in properties
        // The object ID is set at the top level when creating objects
        // Metadata is stored as a JSON string since Weaviate requires nested properties for object types
        let payload = json!({
            "class": class_name,
            "description": format!("Dripstone embedding class for namespace: {}", namespace),
            "vectorizer": "none",
            "properties": [
                {
                    "name": "metadata",
                    "dataType": ["text"],
                    "indexInverted": false
                },
                {
                    "name": CREATED_KEY,
                    "dataType": ["date"],
                    "indexInverted": false
                },
                {
                    "name": UPDATED_KEY,
                    "dataType": ["date"],
                    "indexInverted": false
                }
            ],
            "vectorIndexConfig": {
                "distance": similarity_metric_to_weaviate(SimilarityMetric::Cosine)
            }
        });

        match self.http.post_json("/v1/schema", &payload).await {
            Ok(create_response) => {
                let create_status = create_response.status();
                let create_status_code = create_status.as_u16();
                if !create_status.is_success() {
                    let body = create_response.text().await.unwrap_or_default();
                    // Check if class already exists (422 error with various message formats)
                    if create_status_code == 422
                        && (body.contains("already exists")
                            || body.contains("class already exists")
                            || body.contains("TYPE_ADD_CLASS"))
                    {
                        // Class already exists, which is fine
                        return Ok(());
                    }
                    return Err(DripstoneError::storage(format!(
                        "failed to create weaviate class {class_name}: status {create_status_code} body {body}"
                    )));
                }
                Ok(())
            }
            Err(e) => {
                // Check if the error is a 422 (class already exists)
                // The HTTP client wraps 422s as permanent errors
                let error_msg = format!("{e}");
                if error_msg.contains("422")
                    && (error_msg.contains("already exists")
                        || error_msg.contains("class already exists")
                        || error_msg.contains("TYPE_ADD_CLASS"))
                {
                    // Class already exists, which is fine
                    Ok(())
                } else {
                    Err(e)
                }
            }
        }
    }

    pub(crate) async fn upsert_objects(
        &self,
        namespace: &str,
        records: Vec<EmbeddingRecord>,
    ) -> Result<(), DripstoneError> {
        if records.is_empty() {
            return Ok(());
        }

        self.ensure_class(namespace).await?;
        let class_name = self.class_name(namespace);

        // Weaviate batch API expects objects array
        let mut objects = Vec::with_capacity(records.len());
        for mut record in records {
            if record.vector.len() != self.vector_dimension {
                return Err(DripstoneError::InvalidVectorLength {
                    expected: self.vector_dimension,
                    actual: record.vector.len(),
                });
            }
            record.touch();

            let mut properties = JsonMap::new();
            // Store the original record ID in a property so we can retrieve it later
            // Weaviate's object ID is a UUID, but we need the original ID for lookups
            properties.insert(
                "record_id".to_string(),
                Value::String(record.id.to_string()),
            );

            let (metadata, created_at, updated_at) =
                encode_metadata(record.metadata.take(), record.created_at, record.updated_at);
            if let Some(meta) = metadata {
                // Store metadata as JSON string since Weaviate schema uses text type
                properties.insert(
                    "metadata".to_string(),
                    Value::String(
                        serde_json::to_string(&meta).unwrap_or_else(|_| "{}".to_string()),
                    ),
                );
            }
            properties.insert(
                CREATED_KEY.to_string(),
                Value::String(EmbeddingRecord::format_timestamp(&created_at)?),
            );
            properties.insert(
                UPDATED_KEY.to_string(),
                Value::String(EmbeddingRecord::format_timestamp(&updated_at)?),
            );

            // Weaviate requires UUID format for object IDs
            // Generate deterministic UUID v5 from namespace + record ID
            let uuid = generate_weaviate_uuid(namespace, record.id.as_ref());

            objects.push(json!({
                "class": class_name,
                "id": uuid.to_string(),
                "properties": properties,
                "vector": record.vector
            }));
        }

        // Use batch API for better performance
        let payload = json!({ "objects": objects });
        let response = self.http.post_json("/v1/batch/objects", &payload).await?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(DripstoneError::storage(format!(
                "failed to upsert weaviate objects: status {status} body {body}"
            )));
        }

        // Check for errors in batch response
        let body: Value = response.json().await.map_err(DripstoneError::from)?;
        if let Some(result) = body.get("result").and_then(Value::as_object) {
            if let Some(errors) = result.get("errors").and_then(Value::as_object)
                && !errors.is_empty()
            {
                return Err(DripstoneError::storage(format!(
                    "weaviate batch upsert had errors: {}",
                    serde_json::to_string(errors).unwrap_or_default()
                )));
            }
            // Check for error objects array
            if let Some(error_objects) = result.get("objects").and_then(Value::as_array) {
                for obj in error_objects {
                    if let Some(error_obj) = obj.as_object()
                        && error_obj.contains_key("errors")
                    {
                        return Err(DripstoneError::storage(format!(
                            "weaviate batch upsert had object errors: {}",
                            serde_json::to_string(error_obj).unwrap_or_default()
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    pub(crate) async fn get_object(
        &self,
        namespace: &str,
        id: &str,
    ) -> Result<Option<EmbeddingRecord>, DripstoneError> {
        let class_name = self.class_name(namespace);
        // Convert record ID to Weaviate UUID format
        let uuid = generate_weaviate_uuid(namespace, id);
        // Include vector in the response
        let path = format!("/v1/objects/{class_name}/{uuid}?include=vector");

        match self.http.get(path.as_str()).await {
            Ok(response) => {
                let status = response.status();
                if status == reqwest::StatusCode::NOT_FOUND {
                    return Ok(None);
                }
                if !status.is_success() {
                    let body = response.text().await.unwrap_or_default();
                    return Err(DripstoneError::storage(format!(
                        "failed to get weaviate object: status {status} body {body}"
                    )));
                }
                let body: Value = response.json().await.map_err(DripstoneError::from)?;
                extract_record(namespace, &body)
            }
            Err(e) => {
                // Check if the error is a 404 (object or class doesn't exist)
                let error_msg = format!("{e}");
                if error_msg.contains("404")
                    || error_msg.contains("Not found")
                    || error_msg.contains("doesn't exist")
                {
                    return Ok(None);
                }
                Err(e)
            }
        }
    }

    pub(crate) async fn delete_object(
        &self,
        namespace: &str,
        id: &str,
    ) -> Result<(), DripstoneError> {
        self.ensure_class(namespace).await?;
        let class_name = self.class_name(namespace);
        // Convert record ID to Weaviate UUID format
        let uuid = generate_weaviate_uuid(namespace, id);
        let path = format!("/v1/objects/{class_name}/{uuid}");

        let response = self.http.delete(path.as_str()).await?;
        let status = response.status();

        if status == reqwest::StatusCode::NOT_FOUND {
            return Ok(());
        }

        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(DripstoneError::storage(format!(
                "failed to delete weaviate object: status {status} body {body}"
            )));
        }
        Ok(())
    }

    pub(crate) async fn query_class(
        &self,
        namespace: &str,
        vector: &[f32],
        limit: usize,
        where_filter: Option<String>,
    ) -> Result<Value, DripstoneError> {
        self.ensure_class(namespace).await?;
        let class_name = self.class_name(namespace);

        // Build GraphQL query for similarity search
        let where_clause =
            where_filter.map_or_else(String::new, |clause| format!("where: {{ {clause} }}"));

        // Weaviate GraphQL expects vector as inline JSON array
        // Format: [0.1, 0.2, 0.3, ...]
        let vector_str = format!(
            "[{}]",
            vector
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        );

        let query = format!(
            r"{{
                Get {{
                    {class_name}(
                        limit: {limit}
                        nearVector: {{
                            vector: {vector_str}
                        }}
                        {where_clause}
                    ) {{
                        _additional {{
                            id
                            distance
                            vector
                        }}
                        record_id
                        metadata
                        {CREATED_KEY}
                        {UPDATED_KEY}
                    }}
                }}
            }}"
        );

        let payload = json!({ "query": query });

        let response = self.http.post_json("/v1/graphql", &payload).await?;
        let status = response.status();

        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            // Check for GraphQL errors in response body
            if let Ok(body_json) = serde_json::from_str::<Value>(&body)
                && let Some(errors) = body_json.get("errors")
            {
                return Err(DripstoneError::storage(format!(
                    "weaviate GraphQL query errors: {}",
                    serde_json::to_string(errors).unwrap_or_default()
                )));
            }
            return Err(DripstoneError::storage(format!(
                "failed to query weaviate class: status {status} body {body}"
            )));
        }

        let body: Value = response.json().await.map_err(DripstoneError::from)?;

        // Check for GraphQL errors even if HTTP status is 200
        if let Some(errors) = body.get("errors") {
            return Err(DripstoneError::storage(format!(
                "weaviate GraphQL query errors: {}",
                serde_json::to_string(errors).unwrap_or_default()
            )));
        }

        Ok(body)
    }

    pub(crate) async fn count_class(&self, namespace: &str) -> Result<usize, DripstoneError> {
        let class_name = self.class_name(namespace);

        // Check if class exists first - if not, return 0
        let schema_path = format!("/v1/schema/{class_name}");
        let class_exists = match self.http.get(schema_path.as_str()).await {
            Ok(response) => response.status() == reqwest::StatusCode::OK,
            Err(e) => {
                // Check if the error is a 404 (class doesn't exist)
                let error_msg = format!("{e}");
                if error_msg.contains("404")
                    || error_msg.contains("Not found")
                    || error_msg.contains("doesn't exist")
                {
                    return Ok(0);
                }
                return Err(e);
            }
        };

        if !class_exists {
            return Ok(0);
        }

        // Use GraphQL aggregate query to count
        let query = format!(
            r"{{
                Aggregate {{
                    {class_name} {{
                        meta {{
                            count
                        }}
                    }}
                }}
            }}"
        );

        let payload = json!({ "query": query });
        let response = self.http.post_json("/v1/graphql", &payload).await?;
        let status = response.status();

        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(DripstoneError::storage(format!(
                "failed to count weaviate class: status {status} body {body}"
            )));
        }

        let body: Value = response.json().await.map_err(DripstoneError::from)?;

        // Debug: log the response if count is 0
        if let Some(errors) = body.get("errors") {
            return Err(DripstoneError::storage(format!(
                "weaviate count query had errors: {}",
                serde_json::to_string(errors).unwrap_or_default()
            )));
        }

        let count = body
            .get("data")
            .and_then(|d| d.get("Aggregate"))
            .and_then(|a| a.get(class_name))
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|obj| obj.get("meta"))
            .and_then(|m| m.get("count"))
            .and_then(Value::as_u64)
            .unwrap_or(0);

        usize::try_from(count)
            .map_err(|_| DripstoneError::storage("weaviate count exceeds addressable size"))
    }

    pub(crate) async fn list_classes(&self) -> Result<Vec<String>, DripstoneError> {
        let response = self.http.get("/v1/schema").await?;
        let status = response.status();

        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(DripstoneError::storage(format!(
                "failed to list weaviate classes: status {status} body {body}"
            )));
        }

        let body: Value = response.json().await.map_err(DripstoneError::from)?;
        let empty_vec: Vec<Value> = Vec::new();
        let classes = body
            .get("classes")
            .and_then(Value::as_array)
            .unwrap_or(&empty_vec);

        let mut namespaces = Vec::new();
        for class in classes {
            if let Some(class_name) = class.get("class").and_then(Value::as_str)
                && let Some(stripped) = class_name.strip_prefix(&self.class_prefix)
            {
                // Convert CamelCase back to namespace
                let namespace = stripped
                    .chars()
                    .enumerate()
                    .flat_map(|(i, c)| {
                        if i > 0 && c.is_uppercase() {
                            vec!['_', c.to_lowercase().next().unwrap_or(c)]
                        } else {
                            vec![c.to_lowercase().next().unwrap_or(c)]
                        }
                    })
                    .collect::<String>();
                namespaces.push(namespace);
            }
        }
        Ok(namespaces)
    }

    pub(crate) async fn list_class_objects(
        &self,
        namespace: &str,
    ) -> Result<Vec<EmbeddingRecord>, DripstoneError> {
        self.ensure_class(namespace).await?;
        let class_name = self.class_name(namespace);

        // Use GraphQL to list all objects
        let query = format!(
            r"{{
                Get {{
                    {class_name}(
                        limit: 10000
                    ) {{
                        _additional {{
                            id
                            vector
                        }}
                        record_id
                        metadata
                        {CREATED_KEY}
                        {UPDATED_KEY}
                    }}
                }}
            }}"
        );

        let payload = json!({ "query": query });
        let response = self.http.post_json("/v1/graphql", &payload).await?;
        let status_code = response.status().as_u16();

        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(DripstoneError::storage(format!(
                "failed to list weaviate objects: status {status_code} body {body}"
            )));
        }

        let body: Value = response.json().await.map_err(DripstoneError::from)?;
        let empty_vec: Vec<Value> = Vec::new();
        let objects = body
            .get("data")
            .and_then(|d| d.get("Get"))
            .and_then(|g| g.get(class_name))
            .and_then(Value::as_array)
            .unwrap_or(&empty_vec);

        let mut records = Vec::new();
        for obj in objects {
            if let Ok(Some(record)) = extract_record(namespace, obj) {
                records.push(record);
            }
        }
        Ok(records)
    }
}

#[cfg(test)]
mod class_name_tests {
    use super::WeaviateEmbeddingBackend;
    use crate::backend::http::HttpClient;

    #[test]
    fn class_name_upper_cases_a_multibyte_first_letter() {
        let backend = WeaviateEmbeddingBackend {
            http: HttpClient::with_config("http://127.0.0.1:1".to_string(), Default::default())
                .expect("client"),
            class_prefix: "élan".to_string(),
            vector_dimension: 3,
            batch_size: 1,
        };
        assert_eq!(backend.class_name("docs"), "ÉlanDocs");
    }
}

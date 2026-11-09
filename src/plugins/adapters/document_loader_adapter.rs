//! Schema-defined document loader adapter: implements DocumentLoader via the step DSL.
//!
//! Parsing steps receive args: content (raw file content). Final step output is mapped
//! to Vec<Document> via documents_path, content_path, metadata_path.

#[cfg(all(feature = "plugins", feature = "document-loaders"))]
mod impl_ {
    use crate::data::loaders::document::{Document, DocumentLoader, LoaderError};
    use crate::plugins::runtime::{AgentCapabilities, StepRuntime, resolve_path};
    use crate::plugins::schema::{DocumentLoaderPluginSpec, PluginDefinition, PluginSpec};
    use async_trait::async_trait;
    use std::collections::HashMap;
    use std::sync::Arc;

    const LOADER_TIMEOUT_SECS: u64 = 60;

    #[derive(Debug)]
    pub struct SchemaDocumentLoader {
        definition: Arc<PluginDefinition>,
        spec: DocumentLoaderPluginSpec,
        runtime: Arc<StepRuntime>,
        capabilities: Arc<AgentCapabilities>,
        content: String,
    }

    impl SchemaDocumentLoader {
        pub fn new(
            definition: Arc<PluginDefinition>,
            runtime: Arc<StepRuntime>,
            capabilities: Arc<AgentCapabilities>,
            content: String,
        ) -> Option<Self> {
            let spec = match &definition.spec {
                PluginSpec::DocumentLoader(s) => s.clone(),
                _ => return None,
            };
            Some(Self {
                definition,
                spec,
                runtime,
                capabilities,
                content,
            })
        }

        /// Build a loader for the given content (e.g. file bytes as string). Call load() to run parsing.
        pub fn from_content(
            definition: Arc<PluginDefinition>,
            runtime: Arc<StepRuntime>,
            capabilities: Arc<AgentCapabilities>,
            content: String,
        ) -> Option<Self> {
            Self::new(definition, runtime, capabilities, content)
        }

        fn parse_documents(&self, value: serde_json::Value) -> Result<Vec<Document>, LoaderError> {
            let documents_path = self.spec.documents_path.as_deref().unwrap_or("documents");
            let content_path = self.spec.content_path.as_deref().unwrap_or("content");
            let metadata_path = self.spec.metadata_path.as_deref().unwrap_or("metadata");

            let arr = resolve_path(&value, documents_path)
                .map_err(|e| LoaderError::ParseError(format!("documents path: {}", e)))?;
            let arr = arr.as_array().ok_or_else(|| {
                LoaderError::ParseError("documents path must be an array".to_string())
            })?;

            let mut documents = Vec::with_capacity(arr.len());
            for item in arr {
                let content_val = resolve_path(item, content_path)
                    .map_err(|e| LoaderError::ParseError(format!("content path: {}", e)))?;
                let content = content_val
                    .as_str()
                    .map(String::from)
                    .unwrap_or_else(|| content_val.to_string());

                let metadata: HashMap<String, String> = resolve_path(item, metadata_path)
                    .ok()
                    .and_then(|m| m.as_object().cloned())
                    .map(|obj| {
                        obj.into_iter()
                            .filter_map(|(k, v)| v.as_str().map(|s| (k, s.to_string())))
                            .collect()
                    })
                    .unwrap_or_default();

                documents.push(Document::new(content).with_metadata_map(metadata));
            }
            Ok(documents)
        }
    }

    #[async_trait]
    impl DocumentLoader for SchemaDocumentLoader {
        async fn load(&self) -> Result<Vec<Document>, LoaderError> {
            let args = serde_json::json!({ "content": self.content });
            let value = self
                .runtime
                .execute_steps(
                    &self.spec.parsing_steps,
                    args,
                    &self.capabilities,
                    LOADER_TIMEOUT_SECS,
                )
                .await
                .map_err(|e| LoaderError::ParseError(e.to_string()))?;
            self.parse_documents(value)
        }

        fn name(&self) -> &str {
            &self.definition.name
        }
    }
}

#[cfg(all(feature = "plugins", feature = "document-loaders"))]
pub use impl_::SchemaDocumentLoader;

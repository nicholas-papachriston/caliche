use crate::agent::tools::common::params;
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
use crate::data::storage::vector::Backend;
use crate::llm::embedding::{EmbeddingProvider, EmbeddingRequest};
use crate::llm::provider::Provider;
use crate::messages::{CompletionRequest, Message};
use anyhow::{Context, Result};
use async_trait::async_trait;
use dripstone::{Metric, SearchQuery};
use serde_json::json;
use std::sync::Arc;

const SYSTEM_PROMPT: &str = "You are an expert SQL engineer. Generate safe, efficient SQL \
queries that follow the provided database dialect. Use only the tables and columns from \
the supplied schema context. Return queries that are ready to run without additional explanation. \
If a task cannot be satisfied with the available schema, respond with a comment explaining the issue.";

pub struct SqlGenerationTool {
    provider: Arc<dyn Provider>,
    embedding_provider: Arc<dyn EmbeddingProvider>,
    backend: Arc<dyn Backend + Send + Sync>,
    default_namespace: String,
    default_top_k: usize,
}

impl std::fmt::Debug for SqlGenerationTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqlGenerationTool")
            .field("provider", &"<dyn Provider>")
            .field("embedding_provider", &"<dyn EmbeddingProvider>")
            .field("backend", &"<dyn Backend>")
            .field("default_namespace", &self.default_namespace)
            .field("default_top_k", &self.default_top_k)
            .finish()
    }
}

impl SqlGenerationTool {
    pub fn new(
        provider: Arc<dyn Provider>,
        embedding_provider: Arc<dyn EmbeddingProvider>,
        backend: Arc<dyn Backend + Send + Sync>,
    ) -> Self {
        Self {
            provider,
            embedding_provider,
            backend,
            default_namespace: "default".to_string(),
            default_top_k: 5,
        }
    }

    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.default_namespace = namespace.into();
        self
    }

    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.default_top_k = top_k.clamp(1, 12);
        self
    }

    fn resolve_namespace(&self, namespace: &str) -> String {
        let trimmed = namespace.trim();
        if trimmed.is_empty() {
            return format!("schema::{}", self.default_namespace);
        }

        if trimmed.starts_with("schema::") {
            trimmed.to_string()
        } else {
            format!("schema::{trimmed}")
        }
    }

    fn build_prompt(&self, task: &str, dialect: &str, context_sections: &[String]) -> String {
        let mut prompt = String::new();
        prompt.push_str("Task:\n");
        prompt.push_str(task);
        prompt.push_str("\n\nDatabase Dialect:\n");
        prompt.push_str(dialect);
        prompt.push_str("\n\nSchema Context:\n");

        if context_sections.is_empty() {
            prompt.push_str(
                "- No schema context was retrieved. Avoid guessing table names. \
If the request cannot be satisfied, explain why in a SQL comment.",
            );
        } else {
            for section in context_sections {
                prompt.push_str("- ");
                prompt.push_str(section);
                prompt.push('\n');
            }
        }

        prompt.push_str("\nOutput Instructions:\n");
        prompt.push_str(
            "- Return only the SQL query, optionally prefixed with line comments for caveats.\n\
- Avoid destructive statements (DROP, TRUNCATE, etc.).\n\
- Prefer parameterization for dynamic values using double curly braces (e.g., {{customer_id}}) when literals are unknown.\n\
- If multiple statements are required, separate them with semicolons and comments.",
        );

        prompt
    }

    async fn fetch_schema_context(
        &self,
        namespace: &str,
        description: &str,
        top_k: usize,
    ) -> Result<(Vec<String>, Option<crate::messages::TokenUsage>)> {
        let request = EmbeddingRequest {
            texts: vec![description.to_string()],
            model: None,
        };

        let response = self.embedding_provider.embed(request).await?;
        let embeddings = response.embeddings;
        let embedding_usage = response.usage.map(|usage| crate::messages::TokenUsage {
            prompt_tokens: None,
            completion_tokens: None,
            embedding_tokens: usage.prompt_tokens.or(usage.total_tokens),
            total_tokens: usage.total_tokens,
            context_window: None,
            cache_creation_tokens: None,
            cache_read_tokens: None,
            last_prompt_tokens: None,
        });

        let vector = embeddings
            .into_iter()
            .next()
            .context("Embedding provider did not return a vector")?;

        let query = SearchQuery::new(vector)
            .limit(top_k)
            .metric(Metric::Cosine)
            .namespace(namespace.to_string());

        let results = self.backend.search(&query).await.with_context(|| {
            format!("Failed to retrieve schema context for namespace {namespace}")
        })?;

        let mut sections = Vec::new();
        for result in results {
            let metadata = &result.chunk.metadata;
            let summary = metadata
                .get("summary")
                .and_then(|value| value.as_str())
                .unwrap_or("");
            let schema_name = metadata
                .get("schema")
                .and_then(|value| value.as_str())
                .unwrap_or("unknown");
            let table_name = metadata
                .get("table")
                .and_then(|value| value.as_str())
                .unwrap_or("unknown");

            if summary.is_empty() {
                sections.push(format!(
                    "{}.{} (similarity {:.3})",
                    schema_name, table_name, result.score
                ));
            } else {
                sections.push(format!(
                    "{}.{} (similarity {:.3})\n{}",
                    schema_name, table_name, result.score, summary
                ));
            }
        }

        Ok((sections, embedding_usage))
    }
}

const NAME: &str = "sql_generate";
const DESCRIPTION: &str = "Generate safe, dialect-aware SQL using vectorized schema context. \
Provide a natural language task and optional namespace/dialect. The tool embeds the task, \
retrieves relevant schema snippets, and uses the LLM to author the final SQL query.";

impl ToolMeta for SqlGenerationTool {
    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(DESCRIPTION)
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(NAME)
    }

    fn name(&self) -> &str {
        NAME
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Natural language description of the SQL you need. Example: \"List the top 20 customers by revenue for the past quarter\""
                },
                "dialect": {
                    "type": "string",
                    "description": "SQL dialect to target (postgres, mysql, snowflake, mssql, etc.)",
                    "default": "postgres"
                },
                "namespace": {
                    "type": "string",
                    "description": "Schema embedding namespace to query. When omitted, uses the default namespace configured for the tool."
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 12,
                    "description": "Number of schema chunks to retrieve for context (default depends on configuration)."
                }
            },
            "required": ["task"]
        })
    }
}

#[async_trait]
impl Tool for SqlGenerationTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let task = match params::extract_string(&arguments, "task", None) {
            Ok(task) => task,
            Err(e) => return ToolResult::error(e),
        };

        let dialect = params::extract_string_or(&arguments, "dialect", "postgres");
        let user_namespace =
            params::extract_string_or(&arguments, "namespace", self.default_namespace.as_str());
        let namespace = self.resolve_namespace(&user_namespace);

        let requested_top_k =
            params::extract_number_or(&arguments, "top_k", self.default_top_k as u64) as usize;
        let top_k = requested_top_k.clamp(1, 12);

        let (context_sections, embedding_usage) =
            match self.fetch_schema_context(&namespace, &task, top_k).await {
                Ok(result) => result,
                Err(err) => return ToolResult::error(err.to_string()),
            };

        let prompt = self.build_prompt(&task, &dialect, &context_sections);
        let request = CompletionRequest {
            messages: Arc::new(vec![Message::system(SYSTEM_PROMPT), Message::user(prompt)]),
            max_tokens: Some(512),
            temperature: Some(0.2),
            reasoning_effort: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        };

        let completion = match self.provider.complete(request).await {
            Ok(response) => response,
            Err(err) => return ToolResult::error(format!("SQL generation failed: {err}")),
        };

        let generated_sql = completion.content.as_text().to_string();

        let payload = json!({
            "task": task,
            "dialect": dialect,
            "namespace": namespace,
            "schema_context": context_sections,
            "sql": generated_sql.trim(),
        });

        let mut combined_usage = completion.usage;
        if let Some(embedding_tokens) = embedding_usage {
            combined_usage = match combined_usage {
                Some(existing) => Some(existing + embedding_tokens),
                None => Some(embedding_tokens),
            };
        }

        let payload_str = payload.to_string();
        match combined_usage {
            Some(usage) => ToolResult::success_with_usage(payload_str, usage),
            None => ToolResult::success(payload_str),
        }
    }
}

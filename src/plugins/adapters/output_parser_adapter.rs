//! Schema-defined output parser adapter: implements OutputParser<Output = Value> via the step DSL.
//!
//! Extraction steps receive args: text (LLM output). Final step output is used as parsed value
//! (or a path within it via output_path).

#[cfg(all(feature = "plugins", feature = "output-parsing"))]
mod impl_ {
    use crate::output::parser::{OutputParser, ParseError};
    use crate::plugins::runtime::{AgentCapabilities, StepRuntime, resolve_path};
    use crate::plugins::schema::{OutputParserPluginSpec, PluginDefinition, PluginSpec};
    use async_trait::async_trait;
    use std::sync::Arc;

    const PARSER_TIMEOUT_SECS: u64 = 30;

    #[derive(Debug)]
    pub struct SchemaOutputParserAdapter {
        definition: Arc<PluginDefinition>,
        spec: OutputParserPluginSpec,
        runtime: Arc<StepRuntime>,
        capabilities: Arc<AgentCapabilities>,
    }

    impl SchemaOutputParserAdapter {
        pub fn new(
            definition: Arc<PluginDefinition>,
            runtime: Arc<StepRuntime>,
            capabilities: Arc<AgentCapabilities>,
        ) -> Option<Self> {
            let spec = match &definition.spec {
                PluginSpec::OutputParser(s) => s.clone(),
                _ => return None,
            };
            Some(Self {
                definition,
                spec,
                runtime,
                capabilities,
            })
        }
    }

    #[async_trait]
    impl OutputParser for SchemaOutputParserAdapter {
        type Output = serde_json::Value;

        async fn parse(&self, text: &str) -> Result<Self::Output, ParseError> {
            let args = serde_json::json!({ "text": text });
            let value = self
                .runtime
                .execute_steps(
                    &self.spec.extraction_steps,
                    args,
                    &self.capabilities,
                    PARSER_TIMEOUT_SECS,
                )
                .await
                .map_err(|e| ParseError::ParsingFailed(e.to_string()))?;
            let out = match &self.spec.output_path {
                Some(path) => resolve_path(&value, path)
                    .map_err(|e| ParseError::ParsingFailed(e.to_string()))?,
                None => value,
            };
            Ok(out)
        }

        fn get_format_instructions(&self) -> String {
            self.spec.format_instructions.clone()
        }

        fn name(&self) -> &str {
            &self.definition.name
        }
    }
}

#[cfg(all(feature = "plugins", feature = "output-parsing"))]
pub use impl_::SchemaOutputParserAdapter;

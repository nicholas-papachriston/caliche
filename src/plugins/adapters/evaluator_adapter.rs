//! Schema-defined evaluator adapter: implements Evaluator via the step DSL.
//!
//! Steps run with args: output, expected, context (optional). The final step output
//! is parsed for score, passed, and optional details.

use crate::evaluation::{EvaluationResult, Evaluator};
use crate::plugins::runtime::{AgentCapabilities, StepRuntime};
use crate::plugins::schema::{EvaluatorPluginSpec, PluginDefinition, PluginSpec};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

const EVALUATOR_TIMEOUT_SECS: u64 = 60;

#[derive(Debug)]
pub struct SchemaEvaluatorAdapter {
    definition: Arc<PluginDefinition>,
    spec: EvaluatorPluginSpec,
    runtime: Arc<StepRuntime>,
    capabilities: Arc<AgentCapabilities>,
}

impl SchemaEvaluatorAdapter {
    pub fn new(
        definition: Arc<PluginDefinition>,
        runtime: Arc<StepRuntime>,
        capabilities: Arc<AgentCapabilities>,
    ) -> Option<Self> {
        let spec = match &definition.spec {
            PluginSpec::Evaluator(s) => s.clone(),
            _ => return None,
        };
        Some(Self {
            definition,
            spec,
            runtime,
            capabilities,
        })
    }

    /// Build an evaluator from a plugin definition for use with `evaluate_dataset`.
    pub fn evaluator_from_plugin(
        definition: Arc<PluginDefinition>,
        runtime: Arc<StepRuntime>,
        capabilities: Arc<AgentCapabilities>,
    ) -> Option<Arc<dyn Evaluator>> {
        Self::new(definition, runtime, capabilities).map(|a| Arc::new(a) as Arc<dyn Evaluator>)
    }

    fn args_for_evaluate(
        output: &str,
        expected: &str,
        context: Option<&HashMap<String, serde_json::Value>>,
    ) -> serde_json::Value {
        let context_value = context.map(|m| {
            serde_json::to_value(m)
                .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()))
        });
        serde_json::json!({
            "output": output,
            "expected": expected,
            "context": context_value.unwrap_or(serde_json::Value::Null),
        })
    }

    fn parse_result(&self, value: serde_json::Value) -> Result<EvaluationResult> {
        let score_path = self.spec.score_path.as_deref().unwrap_or("score");
        let passed_path = self.spec.passed_path.as_deref().unwrap_or("passed");
        let details_path = self.spec.details_path.as_deref().unwrap_or("details");

        let root = value.as_object().cloned().unwrap_or_default();
        let root_value = serde_json::Value::Object(root);

        let score = crate::plugins::runtime::resolve_path(&root_value, score_path)
            .ok()
            .and_then(|v| v.as_f64().or_else(|| v.as_i64().map(|i| i as f64)))
            .unwrap_or(0.0);
        let passed = crate::plugins::runtime::resolve_path(&root_value, passed_path)
            .ok()
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let details: HashMap<String, serde_json::Value> =
            crate::plugins::runtime::resolve_path(&root_value, details_path)
                .ok()
                .and_then(|v| v.as_object().cloned())
                .map(|m| m.into_iter().collect::<HashMap<_, _>>())
                .unwrap_or_default();

        Ok(EvaluationResult::new(score, passed).with_details(details))
    }
}

#[async_trait]
impl Evaluator for SchemaEvaluatorAdapter {
    async fn evaluate(
        &self,
        output: &str,
        expected: &str,
        context: Option<&HashMap<String, serde_json::Value>>,
    ) -> Result<EvaluationResult> {
        let args = Self::args_for_evaluate(output, expected, context);
        let value = self
            .runtime
            .execute_steps(
                &self.spec.steps,
                args,
                &self.capabilities,
                EVALUATOR_TIMEOUT_SECS,
            )
            .await
            .map_err(|e| anyhow!("evaluator plugin error: {}", e))?;
        self.parse_result(value)
    }

    fn name(&self) -> &str {
        &self.definition.name
    }
}

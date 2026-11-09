//! JavaScript bindings for Caliche Evaluation Framework
#![expect(
    clippy::trailing_empty_array,
    reason = "NAPI callback code generation emits the zero-sized ABI container"
)]

use caliche::evaluation::{
    ContainsEvaluator as RustContainsEvaluator, EvaluationDataset, Evaluator,
    ExactMatchEvaluator as RustExactMatchEvaluator, LLMEvaluator as RustLLMEvaluator,
};
use napi_derive::napi;
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// Evaluators
// ============================================================================

/// JavaScript wrapper for `ExactMatchEvaluator`
#[napi]
pub struct ExactMatchEvaluator {
    inner: Arc<dyn Evaluator>,
}

#[napi]
impl ExactMatchEvaluator {
    /// Create `ExactMatchEvaluator`
    #[napi(constructor, catch_unwind)]
    pub fn new(case_sensitive: Option<bool>) -> Self {
        Self {
            inner: Arc::new(RustExactMatchEvaluator::new(case_sensitive.unwrap_or(true)))
                as Arc<dyn Evaluator>,
        }
    }

    /// Evaluate with `ExactMatchEvaluator`
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn evaluate(
        &self,
        output: String,
        expected: String,
        context: Option<serde_json::Value>,
    ) -> napi::Result<EvaluationResult> {
        let context_map = context.and_then(|v| {
            if let serde_json::Value::Object(map) = v {
                let mut hashmap = HashMap::new();
                for (k, v) in map {
                    hashmap.insert(k, v);
                }
                Some(hashmap)
            } else {
                None
            }
        });

        let result = self
            .inner
            .evaluate(&output, &expected, context_map.as_ref())
            .await
            .map_err(|e| napi::Error::from_reason(format!("Evaluation failed: {e}")))?;

        Ok(EvaluationResult {
            score: result.score,
            passed: result.passed,
            details: serde_json::json!(result.details),
            error: result.error,
        })
    }
}

/// JavaScript wrapper for `ContainsEvaluator`
#[napi]
pub struct ContainsEvaluator {
    inner: Arc<dyn Evaluator>,
}

#[napi]
impl ContainsEvaluator {
    /// Create `ContainsEvaluator`
    #[napi(constructor, catch_unwind)]
    pub fn new(case_sensitive: Option<bool>) -> Self {
        Self {
            inner: Arc::new(RustContainsEvaluator::new(case_sensitive.unwrap_or(false)))
                as Arc<dyn Evaluator>,
        }
    }

    /// Evaluate with `ContainsEvaluator`
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn evaluate(
        &self,
        output: String,
        expected: String,
        context: Option<serde_json::Value>,
    ) -> napi::Result<EvaluationResult> {
        let context_map = context.and_then(|v| {
            if let serde_json::Value::Object(map) = v {
                let mut hashmap = HashMap::new();
                for (k, v) in map {
                    hashmap.insert(k, v);
                }
                Some(hashmap)
            } else {
                None
            }
        });

        let result = self
            .inner
            .evaluate(&output, &expected, context_map.as_ref())
            .await
            .map_err(|e| napi::Error::from_reason(format!("Evaluation failed: {e}")))?;

        Ok(EvaluationResult {
            score: result.score,
            passed: result.passed,
            details: serde_json::json!(result.details),
            error: result.error,
        })
    }
}

/// JavaScript wrapper for `LLMEvaluator`
#[napi]
pub struct LLMEvaluator {
    inner: Arc<dyn Evaluator>,
}

#[napi]
impl LLMEvaluator {
    /// Create `LLMEvaluator`
    #[napi(constructor, catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when `provider` is not a known provider name.
    /// `provider` is optional here but required by `evaluate`, which fails
    /// without one instead of picking a provider from the environment.
    pub fn new(
        criteria: String,
        provider: Option<String>,
        model: Option<String>,
    ) -> napi::Result<Self> {
        let mut evaluator = RustLLMEvaluator::new(criteria);
        if let Some(provider) = provider {
            let provider = crate::parse_provider_type(&provider)?;
            evaluator = evaluator.with_provider(provider);
        }
        if let Some(model) = model {
            evaluator = evaluator.with_model(model);
        }

        Ok(Self {
            inner: Arc::new(evaluator) as Arc<dyn Evaluator>,
        })
    }

    /// Evaluate with `LLMEvaluator`
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn evaluate(
        &self,
        output: String,
        expected: String,
        context: Option<serde_json::Value>,
    ) -> napi::Result<EvaluationResult> {
        let context_map = context.and_then(|v| {
            if let serde_json::Value::Object(map) = v {
                let mut hashmap = HashMap::new();
                for (k, v) in map {
                    hashmap.insert(k, v);
                }
                Some(hashmap)
            } else {
                None
            }
        });

        let result = self
            .inner
            .evaluate(&output, &expected, context_map.as_ref())
            .await
            .map_err(|e| napi::Error::from_reason(format!("Evaluation failed: {e}")))?;

        Ok(EvaluationResult {
            score: result.score,
            passed: result.passed,
            details: serde_json::json!(result.details),
            error: result.error,
        })
    }
}

// ============================================================================
// Dataset
// ============================================================================

/// Load dataset from JSONL file
#[napi]
///
/// # Errors
///
/// Returns a NAPI error when validation, synchronization, serialization,
/// or an underlying runtime operation fails.
pub async fn load_dataset_from_jsonl(path: String) -> napi::Result<EvaluationDatasetJs> {
    let dataset = EvaluationDataset::from_jsonl(&path)
        .await
        .map_err(|e| napi::Error::from_reason(format!("Failed to load dataset: {e}")))?;

    Ok(EvaluationDatasetJs {
        name: dataset.name,
        description: dataset.description,
        test_cases: dataset
            .test_cases
            .into_iter()
            .map(|tc| TestCaseJs {
                input: tc.input,
                expected_output: tc.expected_output,
                context: tc.context.map(|c| serde_json::json!(c)),
                metadata: tc.metadata.map(|m| serde_json::json!(m)),
            })
            .collect(),
    })
}

/// Load dataset from JSON file
#[napi]
///
/// # Errors
///
/// Returns a NAPI error when validation, synchronization, serialization,
/// or an underlying runtime operation fails.
pub async fn load_dataset_from_json(path: String) -> napi::Result<EvaluationDatasetJs> {
    let dataset = EvaluationDataset::from_json(&path)
        .await
        .map_err(|e| napi::Error::from_reason(format!("Failed to load dataset: {e}")))?;

    Ok(EvaluationDatasetJs {
        name: dataset.name,
        description: dataset.description,
        test_cases: dataset
            .test_cases
            .into_iter()
            .map(|tc| TestCaseJs {
                input: tc.input,
                expected_output: tc.expected_output,
                context: tc.context.map(|c| serde_json::json!(c)),
                metadata: tc.metadata.map(|m| serde_json::json!(m)),
            })
            .collect(),
    })
}

#[napi(object)]
pub struct EvaluationResult {
    pub score: f64,
    pub passed: bool,
    pub details: serde_json::Value,
    pub error: Option<String>,
}

#[napi(object)]
pub struct EvaluationDatasetJs {
    pub name: String,
    pub description: Option<String>,
    pub test_cases: Vec<TestCaseJs>,
}

#[napi(object)]
pub struct TestCaseJs {
    pub input: String,
    pub expected_output: String,
    pub context: Option<serde_json::Value>,
    pub metadata: Option<serde_json::Value>,
}

//! Dataset loading and evaluation orchestration

use crate::evaluation::{EvaluationResult, Evaluator};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// A single test case in an evaluation dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    pub input: String,
    pub expected_output: String,
    pub context: Option<HashMap<String, serde_json::Value>>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Evaluation dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationDataset {
    pub name: String,
    pub description: Option<String>,
    pub test_cases: Vec<TestCase>,
}

impl EvaluationDataset {
    /// Load dataset from JSONL file (one JSON object per line)
    pub async fn from_jsonl<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = tokio::fs::read_to_string(path.as_ref())
            .await
            .with_context(|| format!("Failed to read dataset file: {:?}", path.as_ref()))?;

        let mut test_cases = Vec::new();
        for (line_num, line) in content.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }

            let test_case: TestCase = serde_json::from_str(line)
                .with_context(|| format!("Failed to parse line {} in dataset", line_num + 1))?;
            test_cases.push(test_case);
        }

        Ok(Self {
            name: path
                .as_ref()
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("dataset")
                .to_string(),
            description: None,
            test_cases,
        })
    }

    /// Load dataset from JSON file (array of test cases)
    pub async fn from_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = tokio::fs::read_to_string(path.as_ref())
            .await
            .with_context(|| format!("Failed to read dataset file: {:?}", path.as_ref()))?;

        let test_cases: Vec<TestCase> = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse JSON dataset file: {:?}", path.as_ref()))?;

        Ok(Self {
            name: path
                .as_ref()
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("dataset")
                .to_string(),
            description: None,
            test_cases,
        })
    }
}

/// Results from running evaluation on a dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReport {
    pub dataset_name: String,
    pub evaluator_name: String,
    pub total_cases: usize,
    pub passed_cases: usize,
    pub failed_cases: usize,
    pub average_score: f64,
    pub results: Vec<(usize, EvaluationResult)>,
    pub duration_ms: u64,
}

/// Run evaluation on a dataset
pub async fn evaluate_dataset(
    dataset: &EvaluationDataset,
    evaluator: &dyn Evaluator,
    agent_fn: impl Fn(
        String,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<String, anyhow::Error>> + Send>,
    >,
) -> Result<EvaluationReport> {
    let start_time = std::time::Instant::now();
    let mut results = Vec::new();
    let mut passed_count = 0;
    let mut total_score = 0.0;

    for (idx, test_case) in dataset.test_cases.iter().enumerate() {
        // Get agent output
        let output = match agent_fn(test_case.input.clone()).await {
            Ok(out) => out,
            Err(e) => {
                let error_result = EvaluationResult::new(0.0, false)
                    .with_error(format!("Agent execution failed: {}", e));
                results.push((idx, error_result));
                continue;
            }
        };

        // Evaluate output
        let eval_result = evaluator
            .evaluate(
                &output,
                &test_case.expected_output,
                test_case.context.as_ref(),
            )
            .await?;

        if eval_result.passed {
            passed_count += 1;
        }
        total_score += eval_result.score;

        results.push((idx, eval_result));
    }

    let duration_ms = start_time.elapsed().as_millis() as u64;
    let average_score = if dataset.test_cases.is_empty() {
        0.0
    } else {
        total_score / dataset.test_cases.len() as f64
    };

    Ok(EvaluationReport {
        dataset_name: dataset.name.clone(),
        evaluator_name: evaluator.name().to_string(),
        total_cases: dataset.test_cases.len(),
        passed_cases: passed_count,
        failed_cases: dataset.test_cases.len() - passed_count,
        average_score,
        results,
        duration_ms,
    })
}

//! Evaluation framework for agent outputs
//!
//! Provides evaluators for scoring agent responses against expected outputs,
//! detecting hallucinations, and measuring helpfulness.

pub mod dataset;
pub mod evaluators;

// Explicit exports from dataset module
pub use dataset::{EvaluationDataset, EvaluationReport, TestCase, evaluate_dataset};

// Explicit exports from evaluators module
pub use evaluators::{
    ContainsEvaluator, EvaluationResult, Evaluator, ExactMatchEvaluator,
    LLM_EVALUATOR_NEEDS_PROVIDER, LLMEvaluator, RegexEvaluator,
};

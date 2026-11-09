//! Evaluation Framework Example
//!
//! This example demonstrates:
//! - Creating evaluation datasets
//! - Using different evaluator types (exact match, contains, regex, LLM)
//! - Running evaluations against agent outputs
//! - Generating evaluation reports
//!
//! Run with:
//! ```bash
//! cargo run --example evaluation
//! ```

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use caliche::AgentService;
use caliche::evaluation::{
    dataset::{EvaluationDataset, TestCase, evaluate_dataset},
    evaluators::{ContainsEvaluator, ExactMatchEvaluator},
};
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    println!("=== Evaluation Framework Examples ===\n");

    // Create a simple agent service for testing
    let agent_service = AgentService::new().await?;
    let agent_service_arc = Arc::new(Mutex::new(agent_service));

    // Example 1: Exact Match Evaluator
    println!("1. Exact Match Evaluator");
    let dataset = EvaluationDataset {
        name: "math_questions".to_string(),
        description: Some("Simple math questions".to_string()),
        test_cases: vec![
            TestCase {
                input: "What is 2 + 2? Reply with ONLY the number, no other text.".to_string(),
                expected_output: "4".to_string(),
                context: None,
                metadata: None,
            },
            TestCase {
                input: "What is 5 * 3? Reply with ONLY the number, no other text.".to_string(),
                expected_output: "15".to_string(),
                context: None,
                metadata: None,
            },
        ],
    };

    let evaluator = ExactMatchEvaluator::case_insensitive();
    let agent_fn = create_agent_fn(agent_service_arc.clone());

    let report = evaluate_dataset(&dataset, &evaluator, agent_fn).await?;
    println!("   Total Cases: {}", report.total_cases);
    println!("   Passed: {}", report.passed_cases);
    println!("   Failed: {}", report.failed_cases);
    println!("   Average Score: {:.2}", report.average_score);
    println!("   Duration: {}ms\n", report.duration_ms);

    // Example 2: Contains Evaluator
    println!("2. Contains Evaluator");
    let dataset = EvaluationDataset {
        name: "factual_responses".to_string(),
        description: Some("Factual question responses".to_string()),
        test_cases: vec![
            TestCase {
                input: "Hello".to_string(),
                expected_output: "hello".to_string(), // Just check if response contains greeting
                context: None,
                metadata: None,
            },
            TestCase {
                input: "What is the capital of France?".to_string(),
                expected_output: "Paris".to_string(), // Check if response contains the capital city
                context: None,
                metadata: None,
            },
        ],
    };

    let evaluator = ContainsEvaluator::case_insensitive();
    let agent_fn = create_agent_fn(agent_service_arc.clone());

    let report = evaluate_dataset(&dataset, &evaluator, agent_fn).await?;
    println!("   Total Cases: {}", report.total_cases);
    println!("   Passed: {}", report.passed_cases);
    println!("   Average Score: {:.2}\n", report.average_score);

    // Example 3: Loading from JSON file (commented out - would need actual file)
    println!("3. Loading Dataset from File");
    println!("   Dataset files can be JSON or JSONL format:");
    println!("   JSON format: [{{ \"input\": \"...\", \"expected_output\": \"...\" }}, ...]");
    println!("   JSONL format: One JSON object per line");
    println!("\n   Example JSONL file:");
    println!("   {{\"input\": \"What is 2+2?\", \"expected_output\": \"4\"}}");
    println!("   {{\"input\": \"What is 5*3?\", \"expected_output\": \"15\"}}");
    println!("\n   Load with:");
    println!("   let dataset = EvaluationDataset::from_jsonl(\"test_cases.jsonl\").await?;");

    // Example 4: LLM Evaluator (commented out - requires API key and is slower)
    println!("\n4. LLM Evaluator");
    println!("   The LLM evaluator uses an LLM to grade responses based on custom criteria.");
    println!("   This is useful for evaluating complex, subjective responses.");
    println!("\n   Example:");
    println!("   let evaluator = LLMEvaluator::new(\"Check if response is helpful and accurate\")");
    println!("       .with_provider(ProviderType::OpenAI)");
    println!("       .with_model(\"gpt-4o\".to_string());");

    println!("\n=== CLI Usage ===");
    println!("Evaluate with CLI:");
    println!("  caliche evaluate --dataset test_cases.jsonl --evaluator exact_match");
    println!("  caliche evaluate --dataset test_cases.jsonl --evaluator contains");
    println!("  caliche evaluate --dataset test_cases.jsonl --evaluator regex --pattern '\\d+'");
    println!("  caliche evaluate --dataset test_cases.jsonl --evaluator llm \\");
    println!("    --criteria 'Check if response is helpful' \\");
    println!("    --provider openai --model gpt-4o");

    Ok(())
}

fn create_agent_fn(
    service: Arc<Mutex<AgentService>>,
) -> impl Fn(
    String,
) -> std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<String, anyhow::Error>> + Send>,
> {
    move |input: String| -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<String, anyhow::Error>> + Send>,
    > {
        let service = service.clone();
        Box::pin(async move {
            let mut service = {
                let guard = service.lock().await;
                guard.clone_for_request()
            };
            let response = service.chat(&input).await?;
            Ok(response.text().to_string())
        })
    }
}

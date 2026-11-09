//! Evaluation and testing command handlers.
//!
//! Handles dataset evaluation and agent testing commands.

use crate::commands::TestAction;
use crate::errors::{CliError, CliResult};
use crate::traits::command::{CommandContext, CommandHandler, CommandResult};
use anyhow::Result;
use async_trait::async_trait;
use std::path::PathBuf;
use std::sync::Arc;

impl EvaluateArgs {
    pub fn from_command(
        dataset: &str,
        evaluator: &str,
        pattern: &Option<String>,
        criteria: &Option<String>,
        provider: &Option<String>,
        model: &Option<String>,
        output_format: &str,
    ) -> Self {
        Self {
            dataset: dataset.to_owned(),
            evaluator: evaluator.to_owned(),
            pattern: pattern.clone(),
            criteria: criteria.clone(),
            provider: provider.clone(),
            model: model.clone(),
            output_format: output_format.to_owned(),
        }
    }
}

/// Arguments for the evaluate command.
#[derive(Debug, Clone)]
pub struct EvaluateArgs {
    pub dataset: String,
    pub evaluator: String,
    pub pattern: Option<String>,
    pub criteria: Option<String>,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub output_format: String,
}

/// Arguments for the test command.
#[derive(Debug, Clone)]
pub struct TestArgs {
    pub action: TestAction,
}

impl From<TestAction> for TestArgs {
    fn from(action: TestAction) -> Self {
        Self { action }
    }
}

/// Handler for the `evaluate` command.
pub struct EvaluateHandler;

#[async_trait]
impl CommandHandler for EvaluateHandler {
    type Args = EvaluateArgs;

    async fn execute(
        &self,
        args: Self::Args,
        _ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult> {
        run_evaluate_command(
            args.dataset,
            args.evaluator,
            args.pattern,
            args.criteria,
            args.provider,
            args.model,
            args.output_format,
        )
        .await
        .map_err(|e| CliError::Command(e.to_string()))?;

        Ok(CommandResult::Success(None))
    }

    fn name(&self) -> &'static str {
        "evaluate"
    }

    fn description(&self) -> &'static str {
        "Evaluate agent performance on a dataset"
    }

    fn help(&self) -> Option<&'static str> {
        Some(
            "Run evaluation on a dataset using various evaluators.\n\
             Supports exact match, contains, regex, and LLM-based evaluation.",
        )
    }

    fn examples(&self) -> &[&'static str] {
        &[
            "caliche evaluate dataset.json --evaluator exact",
            "caliche evaluate dataset.json --evaluator llm --criteria \"accuracy\" --provider ollama",
        ]
    }
}

/// Handler for the `test` command.
pub struct TestHandler;

#[async_trait]
impl CommandHandler for TestHandler {
    type Args = TestArgs;

    async fn execute(
        &self,
        args: Self::Args,
        _ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult> {
        Box::pin(run_test_command(args.action))
            .await
            .map_err(|e| CliError::Command(e.to_string()))?;

        Ok(CommandResult::Success(None))
    }

    fn name(&self) -> &'static str {
        "test"
    }

    fn description(&self) -> &'static str {
        "Run agent tests and benchmarks"
    }

    fn help(&self) -> Option<&'static str> {
        Some(
            "Run tests against agent templates.\n\
             Supports running tests, benchmarks, recording, and replay.",
        )
    }

    fn examples(&self) -> &[&'static str] {
        &[
            "caliche test run --template my_template",
            "caliche test benchmark --template my_template --iterations 10",
        ]
    }

    fn requires_credentials(&self) -> bool {
        true
    }
}

/// Run the evaluate command implementation.
pub async fn run_evaluate_command(
    dataset: String,
    evaluator: String,
    pattern: Option<String>,
    criteria: Option<String>,
    provider: Option<String>,
    model: Option<String>,
    output_format: String,
) -> Result<()> {
    use caliche::evaluation::dataset::evaluate_dataset;
    use std::sync::Arc;

    let dataset = load_evaluation_dataset(&dataset).await?;
    let evaluator = build_evaluator(&evaluator, pattern, criteria, provider, model)?;
    let mut agent_service = caliche::agent::service::AgentService::new().await?;
    crate::bootstrap::finalize_cli_service(&mut agent_service, false);
    let agent_service_arc = Arc::new(tokio::sync::Mutex::new(agent_service));
    let report = evaluate_dataset(
        &dataset,
        evaluator.as_ref(),
        evaluation_agent(agent_service_arc),
    )
    .await?;
    print_evaluation_report(&report, &output_format)
}

async fn load_evaluation_dataset(
    dataset: &str,
) -> Result<caliche::evaluation::dataset::EvaluationDataset> {
    use caliche::evaluation::dataset::EvaluationDataset;
    if dataset.ends_with(".jsonl") {
        EvaluationDataset::from_jsonl(dataset).await
    } else {
        EvaluationDataset::from_json(dataset).await
    }
}

fn build_evaluator(
    evaluator: &str,
    pattern: Option<String>,
    criteria: Option<String>,
    provider: Option<String>,
    model: Option<String>,
) -> Result<Arc<dyn caliche::evaluation::Evaluator>> {
    use caliche::evaluation::evaluators::{ContainsEvaluator, ExactMatchEvaluator, RegexEvaluator};
    match evaluator {
        "exact_match" => Ok(Arc::new(ExactMatchEvaluator::case_insensitive())),
        "contains" => Ok(Arc::new(ContainsEvaluator::case_insensitive())),
        "regex" => {
            let pattern_str =
                pattern.ok_or_else(|| anyhow::anyhow!("Pattern required for regex evaluator"))?;
            Ok(Arc::new(RegexEvaluator::new(&pattern_str)?))
        }
        "llm" => Ok(Arc::new(build_llm_evaluator(criteria, provider, model)?)),
        _ => anyhow::bail!("Unknown evaluator type: {evaluator}"),
    }
}

/// Error when `--evaluator llm` has no `--provider`.
const LLM_EVALUATOR_NEEDS_PROVIDER_FLAG: &str =
    "--evaluator llm needs --provider (for example --provider ollama); it does not pick one";

fn build_llm_evaluator(
    criteria: Option<String>,
    provider: Option<String>,
    model: Option<String>,
) -> Result<caliche::evaluation::evaluators::LLMEvaluator> {
    use caliche::evaluation::evaluators::LLMEvaluator;
    use caliche::llm::ProviderType;
    let criteria_str = criteria
        .unwrap_or_else(|| "Evaluate if the output is correct, helpful, and accurate.".to_string());
    let Some(prov_str) = provider else {
        anyhow::bail!(LLM_EVALUATOR_NEEDS_PROVIDER_FLAG);
    };
    let prov = prov_str
        .parse::<ProviderType>()
        .map_err(|e| anyhow::anyhow!("Invalid provider: {e}"))?;
    let mut llm_eval = LLMEvaluator::new(criteria_str).with_provider(prov);
    if let Some(model_str) = model {
        llm_eval = llm_eval.with_model(model_str);
    }
    Ok(llm_eval)
}

fn evaluation_agent(
    agent_service: Arc<tokio::sync::Mutex<caliche::agent::service::AgentService>>,
) -> impl Fn(
    String,
) -> std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<String, anyhow::Error>> + Send>,
> {
    move |input: String| {
        let service = agent_service.clone();
        Box::pin(async move {
            let mut agent = service.lock().await;
            let response = agent.chat(&input).await?;
            Ok(response.text().to_string())
        })
    }
}

fn print_evaluation_report(
    report: &caliche::evaluation::dataset::EvaluationReport,
    output_format: &str,
) -> Result<()> {
    match output_format {
        "json" => {
            println!("{}", serde_json::to_string_pretty(report)?);
        }
        "yaml" => {
            println!("{}", serde_norway::to_string(report)?);
        }
        _ => print_evaluation_text_report(report),
    }
    Ok(())
}

fn print_evaluation_text_report(report: &caliche::evaluation::dataset::EvaluationReport) {
    println!("Evaluation Report");
    println!("=================");
    println!("Dataset: {}", report.dataset_name);
    println!("Evaluator: {}", report.evaluator_name);
    println!("Total Cases: {}", report.total_cases);
    println!("Passed: {}", report.passed_cases);
    println!("Failed: {}", report.failed_cases);
    println!("Average Score: {:.2}", report.average_score);
    println!("Duration: {}ms", report.duration_ms);
    println!("\nResults:");
    for (idx, result) in &report.results {
        let status = if result.passed { "PASS" } else { "FAIL" };
        println!(
            "  {} Test Case {}: Score {:.2}",
            status,
            idx + 1,
            result.score
        );
        if let Some(ref err) = result.error {
            println!("    Error: {err}");
        }
    }
}

/// Run the test command implementation.
pub async fn run_test_command(action: TestAction) -> Result<()> {
    use crate::test::*;

    match action {
        TestAction::Run {
            template,
            test_file,
            template_dir,
        } => {
            let template_dir = template_dir.map(PathBuf::from);
            let test_file = test_file.map(PathBuf::from);
            Box::pin(run_tests(template, test_file, template_dir)).await
        }
        TestAction::Benchmark {
            template,
            iterations,
            test_input,
            template_dir,
        } => {
            let template_dir = template_dir.map(PathBuf::from);
            Box::pin(benchmark_agent(
                Some(template),
                iterations,
                test_input,
                template_dir,
            ))
            .await
        }
        TestAction::Record {
            output,
            template,
            template_dir,
        } => {
            let output = PathBuf::from(output);
            let template_dir = template_dir.map(PathBuf::from);
            Box::pin(record_interaction(output, template, template_dir)).await
        }
        TestAction::Replay { input, compare } => {
            let input = PathBuf::from(input);
            Box::pin(replay_interaction(input, compare)).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_handler_metadata() {
        let handler = EvaluateHandler;
        assert_eq!(handler.name(), "evaluate");
        assert!(!handler.description().is_empty());
    }

    #[tokio::test]
    async fn llm_evaluator_without_provider_refuses_before_running_the_dataset() {
        let dir = tempfile::tempdir().expect("tempdir");
        let dataset = dir.path().join("cases.jsonl");
        std::fs::write(&dataset, "{\"input\":\"2+2\",\"expected_output\":\"4\"}\n")
            .expect("write dataset");
        let error = run_evaluate_command(
            dataset.display().to_string(),
            "llm".to_string(),
            None,
            Some("accuracy".to_string()),
            None,
            None,
            "text".to_string(),
        )
        .await
        .expect_err("no provider must refuse");
        assert_eq!(error.to_string(), LLM_EVALUATOR_NEEDS_PROVIDER_FLAG);
    }

    #[test]
    fn test_test_handler_metadata() {
        let handler = TestHandler;
        assert_eq!(handler.name(), "test");
        assert!(!handler.description().is_empty());
    }
}

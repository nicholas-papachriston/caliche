//! CLI handlers for reasoning patterns.

use caliche::agent::service::AgentService;
use caliche::reasoning::*;

/// Execute a reasoning pattern from CLI
pub async fn execute_reasoning_pattern(
    pattern: &str,
    message: &str,
    max_iterations: usize,
    tree_depth: Option<usize>,
    tree_branches: Option<usize>,
    service: AgentService,
) -> anyhow::Result<ReasoningResult> {
    match pattern.to_lowercase().as_str() {
        "react" => {
            let pattern = react::ReActPattern::new(service, max_iterations);
            pattern.execute(message).await
        }
        "plan-execute" | "plan_execute" => {
            let pattern = plan_execute::PlanExecutePattern::new(service, max_iterations);
            pattern.execute(message).await
        }
        "self-ask" | "self_ask" => {
            let pattern = self_ask::SelfAskPattern::new(service, max_iterations);
            pattern.execute(message).await
        }
        "reflexion" => {
            let pattern = reflexion::ReflexionPattern::new(service, max_iterations);
            pattern.execute(message).await
        }
        "tree-of-thoughts" | "tree_of_thoughts" | "tot" => {
            let branches = tree_branches.unwrap_or(3);
            let depth = tree_depth.unwrap_or(3);
            let pattern = tree_of_thoughts::TreeOfThoughtsPattern::new(service, depth, branches);
            pattern.execute(message).await
        }
        _ => anyhow::bail!("Unknown reasoning pattern: {}", pattern),
    }
}

/// Display reasoning result in a formatted way
pub fn display_reasoning_result(result: &ReasoningResult, verbose: bool) {
    if result.success {
        println!("\n=== Final Answer ===");
        println!("{}", result.final_answer);
    } else {
        println!("\n=== Failed ===");
        if let Some(error) = &result.error {
            println!("Error: {}", error);
        }
    }

    if verbose && !result.steps.is_empty() {
        println!("\n=== Reasoning Steps ===");
        for (idx, step) in result.steps.iter().enumerate() {
            println!("\nStep {}: {:?}", idx + 1, step.step_type);
            println!("{}", step.content);
            if let Some(metadata) = &step.metadata {
                println!("Metadata: {}", metadata);
            }
        }
    }

    if let Some(usage) = &result.token_usage {
        println!("\n=== Token Usage ===");
        if let Some(prompt) = usage.prompt_tokens {
            println!("Prompt tokens: {}", prompt);
        }
        if let Some(completion) = usage.completion_tokens {
            println!("Completion tokens: {}", completion);
        }
        if let Some(total) = usage.total_tokens {
            println!("Total tokens: {}", total);
        }
    }
}

//! Research assistant tool: delegates a query to a sub-agent that can use fs and related discovery tools.
//!
//! The main agent calls this tool with a research question; a research sub-agent (configured via
//! [`ResearchRunner`]) runs with tools such as fs and workspace
//! to gather information and returns a synthesized answer.
//!
//! [`ParallelResearchRunner`] performs a cursory scout to derive sub-questions, runs multiple
//! sub-agents in parallel, then compiles their findings into one answer.

/// Default max tool-call iterations per research sub-agent when the tool caller does not set `max_turns`.
/// Keeps sub-agents from burning the global default (50) and hitting the iteration limit.
const DEFAULT_RESEARCH_SUB_AGENT_ITERATIONS: usize = 12;

/// Scout and synthesis are single-purpose (one response); use a low cap to avoid runaway tool use.
const SCOUT_AND_SYNTHESIS_MAX_ITERATIONS: usize = 5;

use crate::agent::service::AgentService;
use crate::agent::tools::tool::{Tool, ToolMeta, ToolResult};
use crate::llm::ProviderType;
use crate::messages::TokenUsage;
use anyhow::Context;
use async_trait::async_trait;
use std::sync::Arc;

/// Type for optional progress sink (avoids `clippy::type_complexity`).
type ResearchProgressSink = Option<Box<dyn Fn(&str) + Send>>;

/// Runner that executes a research task (e.g. a sub-agent with fs) and returns a summary.
#[async_trait]
pub trait ResearchRunner: Send + Sync {
    /// Run a research task for the given query. The runner may use fs,
    /// fs op=list/read, and similar tools. Returns a synthesized answer and optional token usage
    /// (so the main agent can count sub-agent spend), or an error.
    async fn run(
        &self,
        query: &str,
        max_turns: Option<usize>,
    ) -> anyhow::Result<(String, Option<TokenUsage>)>;
}

/// Factory that creates a fresh research [`AgentService`] (e.g. for parallel sub-agents).
#[async_trait]
pub trait ResearchServiceFactory: Send + Sync {
    /// Create a new research service with the same config (provider, model, tools, system prompt).
    async fn create_research_service(&self) -> anyhow::Result<AgentService>;
}

/// Builds research services from provider, model, tools, and system prompt.
/// Use with [`ParallelResearchRunner`] so multiple sub-agents can run in parallel.
#[derive(Clone)]
pub struct ConfigurableResearchServiceFactory {
    provider: ProviderType,
    model: String,
    tools: Vec<String>,
    system_prompt: String,
}

impl std::fmt::Debug for ConfigurableResearchServiceFactory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConfigurableResearchServiceFactory")
            .field("provider", &self.provider)
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl ConfigurableResearchServiceFactory {
    /// Create a factory that will build research services with the given config.
    #[must_use]
    pub const fn new(
        provider: ProviderType,
        model: String,
        tools: Vec<String>,
        system_prompt: String,
    ) -> Self {
        Self {
            provider,
            model,
            tools,
            system_prompt,
        }
    }
}

#[async_trait]
impl ResearchServiceFactory for ConfigurableResearchServiceFactory {
    async fn create_research_service(&self) -> anyhow::Result<AgentService> {
        let mut svc = AgentService::from_config(
            crate::AgentServiceConfig::new()
                .with_provider(Some(self.provider), Some(self.model.clone()))
                .with_tool_names(self.tools.clone()),
        )
        .await
        .context("Failed to create research service")?;
        svc.agent_mut()
            .set_system_prompt(self.system_prompt.as_str());
        Ok(svc)
    }
}

/// Tool that delegates research to a sub-agent. Use when the main agent needs focused lookup
/// (e.g. "where is X configured", "how does Y work") without issuing many tool calls itself.
pub struct ResearchTool {
    runner: Arc<dyn ResearchRunner>,
}

impl std::fmt::Debug for ResearchTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResearchTool").finish_non_exhaustive()
    }
}

impl ResearchTool {
    pub fn new(runner: Arc<dyn ResearchRunner>) -> Self {
        Self { runner }
    }
}

impl ToolMeta for ResearchTool {
    fn description(&self) -> &'static str {
        "Delegate a research question to a research assistant. The runner scouts for 3-6 tightly scoped sub-questions, runs research sub-agents in parallel (each with a narrow responsibility for speed and fidelity), then compiles findings. Prefer topics that decompose into multiple focused sub-questions; parallel sub-agents yield better results than a single broad query."
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(
            "Delegate a research question to a research assistant. The runner scouts for 3-6 tightly scoped sub-questions, runs research sub-agents in parallel (each with a narrow responsibility for speed and fidelity), then compiles findings. Prefer topics that decompose into multiple focused sub-questions.",
        )
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed("research")
    }

    fn name(&self) -> &'static str {
        "research"
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The research topic. The runner scouts for 3-6 tightly scoped sub-questions, runs sub-agents in parallel (narrow responsibilities = faster, higher fidelity), and compiles one answer."
                },
                "max_turns": {
                    "type": "number",
                    "description": "Optional maximum turns per research sub-agent (default: 12). Use a higher value for complex questions; capped at 20."
                }
            },
            "required": ["query"]
        })
    }
}

#[async_trait]
impl Tool for ResearchTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let query = match arguments.get("query").and_then(|v| v.as_str()) {
            Some(q) => q.trim(),
            None => {
                return ToolResult::error("Missing required parameter: query");
            }
        };
        if query.is_empty() {
            return ToolResult::error("Parameter query must be non-empty");
        }
        let max_turns = arguments
            .get("max_turns")
            .and_then(serde_json::Value::as_u64)
            .map(|n| n as usize)
            .filter(|&n| n > 0 && n <= 20);

        match self.runner.run(query, max_turns).await {
            Ok((answer, None)) => ToolResult::success(answer),
            Ok((answer, Some(usage))) => ToolResult::success_with_usage(answer, usage),
            Err(e) => ToolResult::error(format!("Research failed: {e}")),
        }
    }
}

/// Research runner that runs a dedicated [`AgentService`] (configured with fs,
/// workspace) to answer the query. Use when building the main agent to enable
/// the research tool. The service is wrapped in a mutex so multiple research calls can be serialized.
pub struct AgentResearchRunner {
    service: std::sync::Arc<tokio::sync::Mutex<AgentService>>,
}

impl std::fmt::Debug for AgentResearchRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentResearchRunner")
            .finish_non_exhaustive()
    }
}

impl AgentResearchRunner {
    /// Create a runner from a shared, mutex-protected research service. The caller should build
    /// an `AgentService` with tools such as fs, workspace and
    /// wrap it in `Arc::new(tokio::sync::Mutex::new(service))`.
    pub const fn new(service: Arc<tokio::sync::Mutex<AgentService>>) -> Self {
        Self { service }
    }
}

#[async_trait]
impl ResearchRunner for AgentResearchRunner {
    async fn run(
        &self,
        query: &str,
        max_turns: Option<usize>,
    ) -> anyhow::Result<(String, Option<TokenUsage>)> {
        let mut guard = self.service.lock().await;
        if let Some(limit) = max_turns {
            guard.agent_mut().set_max_iterations(limit.clamp(1, 20));
        }
        let response = guard.chat(query).await?;
        let usage = response.usage().ensure_total_tokens();
        Ok((response.text().to_string(), Some(usage)))
    }
}

/// Runner that does a cursory scout to derive sub-questions, runs multiple research sub-agents
/// in parallel, then compiles their findings into one answer.
pub struct ParallelResearchRunner {
    factory: Arc<dyn ResearchServiceFactory>,
    /// Maximum number of sub-agents to run in parallel (capped when splitting the query).
    max_parallel: usize,
}

impl std::fmt::Debug for ParallelResearchRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelResearchRunner")
            .field("max_parallel", &self.max_parallel)
            .finish_non_exhaustive()
    }
}

impl ParallelResearchRunner {
    /// Create a parallel research runner that uses the factory to spawn sub-agents.
    /// `max_parallel` is the maximum number of sub-queries to run at once (e.g. 4).
    pub fn new(factory: Arc<dyn ResearchServiceFactory>, max_parallel: usize) -> Self {
        Self {
            factory,
            max_parallel: max_parallel.clamp(2, 6),
        }
    }
}

thread_local! {
    static RESEARCH_PROGRESS_SINK: std::cell::RefCell<ResearchProgressSink> =
        std::cell::RefCell::new(None);
}

/// Report progress for research phases (scout, parallel sub-agents, compile). If a sink was set
/// via [`set_research_progress_sink`], it is used; otherwise progress is printed to stderr so the
/// user sees that the request has not stalled.
#[expect(
    clippy::print_stderr,
    reason = "fallback operator progress when no research progress sink is installed"
)]
fn report_research_progress(msg: &str) {
    tracing::info!(target: "research_progress", "{}", msg);
    let used = RESEARCH_PROGRESS_SINK.try_with(|cell| {
        if let Some(ref sink) = *cell.borrow() {
            sink(msg);
            true
        } else {
            false
        }
    });
    if used != Ok(true) {
        eprintln!("\x1b[90m  [Research] {msg}\x1b[0m");
    }
}

/// Set a progress sink for research phases (scout, parallel sub-agents, compile).
/// Used by the CLI to show status in the REPL; call with `None` to clear.
pub fn set_research_progress_sink(sink: ResearchProgressSink) {
    RESEARCH_PROGRESS_SINK.with(|cell| *cell.borrow_mut() = sink);
}

/// Extract a JSON array of strings from scout response (may be wrapped in markdown/code block).
fn parse_sub_questions_from_scout(response: &str) -> Option<Vec<String>> {
    let slice = caliche_contracts::text::extract_json(response)?;
    let arr: Vec<String> = serde_json::from_str(slice).ok()?;
    let filtered: Vec<String> = arr
        .into_iter()
        .filter_map(|s| {
            let s = s.trim().to_string();
            if s.is_empty() { None } else { Some(s) }
        })
        .collect();
    if filtered.len() >= 2 {
        Some(filtered)
    } else {
        None
    }
}

#[async_trait]
impl ResearchRunner for ParallelResearchRunner {
    async fn run(
        &self,
        query: &str,
        max_turns: Option<usize>,
    ) -> anyhow::Result<(String, Option<TokenUsage>)> {
        let per_agent_limit = max_turns
            .unwrap_or(DEFAULT_RESEARCH_SUB_AGENT_ITERATIONS)
            .min(20);

        report_research_progress("Scouting sub-questions...");

        // Phase 1: Cursory scout to get 3-5 focused sub-questions (tighter scopes = faster parallel execution, better fidelity)
        let scout_prompt = format!(
            r#"You are a research planner. For the following topic, list 3-6 tightly scoped sub-questions that together would fully answer it. Each sub-question must be a single, focused research task with a narrow responsibility (e.g. "Where is X defined and what is its public API?", "What does module Y export and how is it used?"). Avoid broad questions; narrow scopes run in parallel faster and yield higher-fidelity results.
Reply with ONLY a JSON array of strings. No other text, no markdown, no explanation. Example: ["Where is X defined and how is it structured?", "What does interface Y export?", "What are the main call sites of Z?"]

Topic: {query}"#
        );

        let mut scout_svc = self
            .factory
            .create_research_service()
            .await
            .context("Failed to create scout research service")?;
        scout_svc
            .agent_mut()
            .set_max_iterations(SCOUT_AND_SYNTHESIS_MAX_ITERATIONS);
        let scout_response = scout_svc.chat(&scout_prompt).await;
        let (sub_questions, mut combined_usage) = match &scout_response {
            Ok(resp) => (
                parse_sub_questions_from_scout(resp.text()),
                Some(resp.usage().ensure_total_tokens()),
            ),
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    "research scout failed; falling back to single-query runner"
                );
                (None, None)
            }
        };

        // If we didn't get 2+ sub-questions, fall back to single full query
        let sub_questions: Vec<String> = if let Some(q) = sub_questions {
            let taken: Vec<String> = q.into_iter().take(self.max_parallel).collect();
            if taken.len() < 2 {
                report_research_progress("Running single full query...");
                let mut fallback_svc = self
                    .factory
                    .create_research_service()
                    .await
                    .context("Failed to create fallback research service")?;
                fallback_svc.agent_mut().set_max_iterations(per_agent_limit);
                let response = fallback_svc.chat(query).await?;
                let u = (combined_usage.unwrap_or_default() + (*response.usage()))
                    .ensure_total_tokens();
                return Ok((response.text().to_string(), Some(u)));
            }
            taken
        } else {
            report_research_progress("Running single full query...");
            let mut fallback_svc = self
                .factory
                .create_research_service()
                .await
                .context("Failed to create fallback research service")?;
            fallback_svc.agent_mut().set_max_iterations(per_agent_limit);
            let response = fallback_svc.chat(query).await?;
            let u =
                (combined_usage.unwrap_or_default() + (*response.usage())).ensure_total_tokens();
            return Ok((response.text().to_string(), Some(u)));
        };

        report_research_progress(&format!(
            "Running {} sub-agents in parallel...",
            sub_questions.len()
        ));

        // Run sub-agents in parallel. Each sub-agent has its own service.
        let factory = Arc::clone(&self.factory);
        let futures: Vec<_> = sub_questions
            .iter()
            .map(|q| {
                let query = q.clone();
                let fac = Arc::clone(&factory);
                let limit = per_agent_limit;
                async move {
                    let mut svc = fac.create_research_service().await?;
                    svc.agent_mut().set_max_iterations(limit);
                    let response = svc.chat(&query).await?;
                    let usage = response.usage().ensure_total_tokens();
                    anyhow::Result::<(String, TokenUsage)>::Ok((response.text().to_string(), usage))
                }
            })
            .collect();

        let results: Vec<Result<(String, TokenUsage), anyhow::Error>> =
            futures_util::future::join_all(futures).await;

        let (findings, parallel_usage): (Vec<String>, TokenUsage) = results
            .into_iter()
            .zip(sub_questions.iter())
            .filter_map(|(res, q)| {
                res.map_err(|e| tracing::warn!("Research sub-agent failed for '{}': {}", q, e))
                    .ok()
            })
            .fold(
                (Vec::new(), TokenUsage::default()),
                |(mut texts, acc_usage), (text, usage)| {
                    texts.push(text);
                    (texts, acc_usage + usage)
                },
            );
        combined_usage = Some(
            combined_usage.unwrap_or_default().ensure_total_tokens()
                + parallel_usage.ensure_total_tokens(),
        );

        if findings.is_empty() {
            anyhow::bail!("All research sub-agents failed");
        }

        report_research_progress("Compiling findings...");

        // Phase 3: Compile findings into one answer
        let synthesis_prompt = format!(
            r"Compile the following research findings into one coherent answer. Original topic: {}

Findings:
{}",
            query,
            findings
                .iter()
                .enumerate()
                .map(|(i, s)| format!("--- Finding {} ---\n{}", i + 1, s))
                .collect::<Vec<_>>()
                .join("\n\n")
        );

        let mut synth_svc = self
            .factory
            .create_research_service()
            .await
            .context("Failed to create synthesis research service")?;
        synth_svc
            .agent_mut()
            .set_max_iterations(SCOUT_AND_SYNTHESIS_MAX_ITERATIONS);
        let synth_response = synth_svc.chat(&synthesis_prompt).await?;
        let synth_usage = synth_response.usage().ensure_total_tokens();
        let total_usage = (combined_usage.unwrap_or_default() + synth_usage).ensure_total_tokens();
        Ok((synth_response.text().to_string(), Some(total_usage)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Agent;
    use crate::agent::tools::ToolRegistry;
    use crate::testing::MockProvider;

    /// Every research service shares one scripted provider.
    struct ScriptedFactory(Arc<MockProvider>);

    #[async_trait]
    impl ResearchServiceFactory for ScriptedFactory {
        async fn create_research_service(&self) -> anyhow::Result<AgentService> {
            let agent = Agent::new(
                Arc::clone(&self.0) as Arc<dyn crate::llm::Provider>,
                Arc::new(ToolRegistry::new()),
            );
            AgentService::new_with_agent(agent, ProviderType::OpenAI)
        }
    }

    /// Runs research with `replies` as the provider answers, oldest first.
    /// Returns the answer and the number of provider calls.
    async fn research(replies: &[&str]) -> (String, usize) {
        let provider = Arc::new(MockProvider::new(
            replies.iter().map(ToString::to_string).collect(),
        ));
        let runner =
            ParallelResearchRunner::new(Arc::new(ScriptedFactory(Arc::clone(&provider))), 4);
        let (answer, _) = runner
            .run("Which crate parses JSON?", None)
            .await
            .expect("research runs");
        (answer, provider.recorded_prompts().len())
    }

    #[tokio::test]
    async fn fenced_sub_questions_after_text_run_in_parallel() {
        let scout = "Sub-questions:\n```json\n[\"Where is A?\", \"Where is B?\"]\n```";
        assert_eq!(
            research(&[scout, "finding", "finding", "compiled"]).await,
            ("compiled".to_string(), 4)
        );
    }

    /// An object that holds the array is not read as sub-questions.
    #[tokio::test]
    async fn object_holding_the_array_falls_back_to_a_single_query() {
        let scout = r#"{"questions": ["Where is A?", "Where is B?"]}"#;
        assert_eq!(
            research(&[scout, "single answer"]).await,
            ("single answer".to_string(), 2)
        );
    }

    #[tokio::test]
    async fn object_before_the_array_falls_back_to_a_single_query() {
        let scout = r#"Plan {"n": 2}: ["Where is A?", "Where is B?"]"#;
        assert_eq!(
            research(&[scout, "single answer"]).await,
            ("single answer".to_string(), 2)
        );
    }
}

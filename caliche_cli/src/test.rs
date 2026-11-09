//! CLI commands for testing agents

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// A single recorded turn consisting of user input and the assistant's reply.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedTurn {
    pub user: String,
    pub assistant: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools_used: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
}

/// Recording header written as the first JSONL line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordingHeader {
    pub kind: String,
    pub version: u32,
    pub started_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template_id: Option<String>,
}

/// Magic string used in `RecordingHeader::kind` to identify recorded JSONL.
pub const RECORDING_KIND: &str = "caliche.recording";

/// Run test suite against agent
pub async fn run_tests(
    template_id: Option<String>,
    test_file: Option<PathBuf>,
    template_dir: Option<PathBuf>,
) -> Result<()> {
    if let Some(template_id) = template_id {
        // Test specific template
        println!("Testing template: {}", template_id);
        // Use template test function
        Box::pin(crate::templates::test_template(
            &template_id,
            None,
            template_dir,
        ))
        .await?;
    } else if let Some(test_file) = test_file {
        // Load test scenarios from file
        println!("Loading test scenarios from: {}", test_file.display());
        let content = tokio::fs::read_to_string(&test_file).await?;

        // Parse test file (supports both JSON and YAML)
        let test_suite = parse_test_file(&content, &test_file)?;

        println!("Found {} test scenario(s)", test_suite.scenarios.len());

        // Execute each test scenario
        let mut passed = 0;
        let mut failed = 0;

        for (idx, scenario) in test_suite.scenarios.iter().enumerate() {
            println!("\n--- Running test {}: {} ---", idx + 1, scenario.name);

            match Box::pin(execute_test_scenario(scenario, template_dir.as_ref())).await {
                Ok(()) => {
                    println!("Test passed: {}", scenario.name);
                    passed += 1;
                }
                Err(e) => {
                    println!("Test failed: {} - {}", scenario.name, e);
                    failed += 1;
                }
            }
        }

        println!("\n=== Test Results ===");
        println!("Passed: {}", passed);
        println!("Failed: {}", failed);
        println!("Total: {}", test_suite.scenarios.len());

        if failed > 0 {
            return Err(anyhow::anyhow!("{} test(s) failed", failed));
        }
    } else {
        return Err(anyhow::anyhow!(
            "Either --template or --test-file must be provided"
        ));
    }

    Ok(())
}

/// Benchmark agent performance
pub async fn benchmark_agent(
    template_id: Option<String>,
    iterations: usize,
    test_input: Option<String>,
    template_dir: Option<PathBuf>,
) -> Result<()> {
    if let Some(template_id) = template_id {
        Box::pin(crate::templates::benchmark_template(
            &template_id,
            iterations,
            test_input,
            template_dir,
        ))
        .await?;
    } else {
        return Err(anyhow::anyhow!("--template is required for benchmarking"));
    }

    Ok(())
}

/// Record an interactive session to a JSONL file.
///
/// The first JSONL line is a [`RecordingHeader`]; each subsequent line is a
/// [`RecordedTurn`] appended after the assistant responds. Recording stops on
/// EOF, an empty blank line, or when the user types `:quit`/`:exit`.
pub async fn record_interaction(
    output: PathBuf,
    template_id: Option<String>,
    template_dir: Option<PathBuf>,
) -> Result<()> {
    use std::io::Write;
    use tokio::io::{AsyncBufReadExt, BufReader};

    println!("Recording interaction to: {}", output.display());
    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
    {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("Failed to create directory {}", parent.display()))?;
    }

    let mut service = Box::pin(build_service_for(
        template_id.as_deref(),
        template_dir.as_ref(),
    ))
    .await?;

    let file = std::fs::File::create(&output)
        .with_context(|| format!("Failed to create recording file {}", output.display()))?;
    let mut file = std::io::BufWriter::new(file);

    let header = RecordingHeader {
        kind: RECORDING_KIND.to_string(),
        version: 1,
        started_at: Utc::now(),
        template_id,
    };
    serde_json::to_writer(&mut file, &header)?;
    writeln!(file)?;

    println!("Type a message and press enter. Blank line, EOF, or `:quit` to stop.");

    let mut reader = BufReader::new(tokio::io::stdin());
    let mut buffer = String::new();
    let mut turn_count = 0usize;

    loop {
        print!("> ");
        std::io::stdout().flush()?;

        buffer.clear();
        let bytes = reader.read_line(&mut buffer).await?;
        if bytes == 0 {
            break;
        }
        let trimmed = buffer.trim().to_string();
        if trimmed.is_empty() {
            break;
        }
        if matches!(trimmed.as_str(), ":quit" | ":exit") {
            break;
        }

        let response = service
            .chat(&trimmed)
            .await
            .with_context(|| format!("Agent error on turn {}", turn_count + 1))?;
        println!("{}", response.text());

        let turn = RecordedTurn {
            user: trimmed,
            assistant: response.text().to_string(),
            tools_used: response.tools_executed.clone(),
            total_tokens: response.usage().total_tokens.map(u64::from),
        };
        serde_json::to_writer(&mut file, &turn)?;
        writeln!(file)?;
        file.flush()?;
        turn_count += 1;
    }

    println!("Recorded {} turn(s) to {}", turn_count, output.display());
    Ok(())
}

/// Replay a previously recorded session.
///
/// Without `--compare`, each recorded user turn is replayed against the
/// configured agent and its new response is printed. With `--compare`, the
/// replayed output is diffed against the saved assistant text using
/// `similar::TextDiff` and a summary of match/mismatch turns is reported.
pub async fn replay_interaction(input: PathBuf, compare: bool) -> Result<()> {
    use similar::{ChangeTag, TextDiff};

    println!("Replaying interaction from: {}", input.display());

    let content = tokio::fs::read_to_string(&input)
        .await
        .with_context(|| format!("Failed to read {}", input.display()))?;

    let mut line_iter = content.lines();
    let header_line = line_iter
        .next()
        .ok_or_else(|| anyhow::anyhow!("Recording file is empty"))?;
    let header: RecordingHeader = serde_json::from_str::<RecordingHeader>(header_line)
        .context("Failed to parse recording header (expected JSONL)")?;
    if header.kind != RECORDING_KIND {
        anyhow::bail!("Unrecognised recording kind: {}", header.kind);
    }

    let turns: Vec<RecordedTurn> = line_iter
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            serde_json::from_str::<RecordedTurn>(line).context("Failed to parse recorded turn")
        })
        .collect::<Result<_>>()?;

    println!("Found {} recorded turn(s)", turns.len());
    if let Some(template_id) = &header.template_id {
        println!("Recorded template: {}", template_id);
    }

    let mut service = Box::pin(build_service_for(header.template_id.as_deref(), None)).await?;

    let mut matched = 0usize;
    let mut diverged = 0usize;
    for (idx, recorded) in turns.iter().enumerate() {
        println!("\n--- Turn {} ---", idx + 1);
        println!("user> {}", recorded.user);

        let response = service
            .chat(&recorded.user)
            .await
            .with_context(|| format!("Replay failed on turn {}", idx + 1))?;
        let actual = response.text();

        if compare {
            let diff = TextDiff::from_lines(recorded.assistant.as_str(), actual);
            let mut had_diff = false;
            let mut diff_out = String::new();
            for change in diff.iter_all_changes() {
                match change.tag() {
                    ChangeTag::Equal => {}
                    ChangeTag::Delete => {
                        had_diff = true;
                        diff_out.push_str(&format!("- {}", change));
                    }
                    ChangeTag::Insert => {
                        had_diff = true;
                        diff_out.push_str(&format!("+ {}", change));
                    }
                }
            }
            if had_diff {
                diverged += 1;
                println!("DIVERGENT:");
                println!("{}", diff_out);
            } else {
                matched += 1;
                println!("MATCH ({} chars)", actual.len());
            }
        } else {
            println!("actual> {}", actual);
        }
    }

    if compare {
        println!(
            "\n=== Replay Summary ===\nMatched: {}\nDivergent: {}\nTotal: {}",
            matched,
            diverged,
            turns.len()
        );
        if diverged > 0 {
            anyhow::bail!("{} turn(s) diverged from the recording", diverged);
        }
    }

    Ok(())
}

/// Build an `AgentService` from an optional template id, falling back to the
/// default builder. Shared by record/replay so their configurations stay
/// aligned.
async fn build_service_for(
    template_id: Option<&str>,
    template_dir: Option<&PathBuf>,
) -> Result<caliche::AgentService> {
    use caliche::agent::builder::AgentBuilder;

    match template_id {
        Some(id) => {
            let mut registry = caliche::templates::TemplateRegistry::new();
            let dir = template_dir
                .cloned()
                .unwrap_or_else(|| PathBuf::from("templates"));
            registry.add_template_dir(&dir);
            registry.load_templates().await?;
            let template = registry
                .get(id)
                .ok_or_else(|| anyhow::anyhow!("Template not found: {}", id))?;
            let builder = template.instantiate().await?;
            Box::pin(builder.build())
                .await
                .context("Failed to build agent from template")
        }
        None => Box::pin(AgentBuilder::new().build())
            .await
            .context("Failed to build default agent"),
    }
}

/// Test suite structure for file-based testing
#[derive(Debug, Deserialize, Serialize)]
struct TestSuite {
    /// Test scenarios to execute
    scenarios: Vec<TestScenario>,
}

/// Individual test scenario
#[derive(Debug, Deserialize, Serialize)]
struct TestScenario {
    /// Name of the test scenario
    name: String,
    /// Input message for the agent
    input: String,
    /// Expected output (exact match)
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_output: Option<String>,
    /// Expected output contains (substring match)
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_output_contains: Option<Vec<String>>,
    /// Expected output matches (regex pattern)
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_output_matches: Option<String>,
    /// Tools that should be allowed for this test
    #[serde(skip_serializing_if = "Option::is_none")]
    tools_allowed: Option<Vec<String>>,
    /// Maximum number of turns for this test
    #[serde(skip_serializing_if = "Option::is_none")]
    max_turns: Option<usize>,
    /// Expected tools to be used
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_tools_used: Option<Vec<String>>,
    /// Template ID to use for this test (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    template_id: Option<String>,
}

/// Parse test file (supports JSON and YAML)
fn parse_test_file(content: &str, file_path: &PathBuf) -> Result<TestSuite> {
    let extension = file_path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("json")
        .to_lowercase();

    match extension.as_str() {
        "json" => serde_json::from_str(content).context("Failed to parse JSON test file"),
        "yaml" | "yml" => serde_norway::from_str(content).context("Failed to parse YAML test file"),
        _ => Err(anyhow::anyhow!(
            "Unsupported file format: {}. Use .json or .yaml/.yml",
            extension
        )),
    }
}

/// Execute a single test scenario
async fn execute_test_scenario(
    scenario: &TestScenario,
    template_dir: Option<&PathBuf>,
) -> Result<()> {
    let mut service = build_scenario_service(scenario, template_dir).await?;
    let response = service
        .chat(&scenario.input)
        .await
        .context("Failed to execute chat")?;
    let response_text = response.text();
    validate_output_expectations(scenario, response_text)?;
    validate_tool_expectations(scenario, &response.tools_executed)
}

async fn build_scenario_service(
    scenario: &TestScenario,
    template_dir: Option<&PathBuf>,
) -> Result<caliche::AgentService> {
    use caliche::agent::builder::AgentBuilder;

    if let Some(template_id) = &scenario.template_id {
        let mut registry = caliche::templates::TemplateRegistry::new();
        let dir = template_dir
            .map(|p| p.clone())
            .unwrap_or_else(|| PathBuf::from("templates"));
        registry.add_template_dir(&dir);
        registry.load_templates().await?;

        let template = registry
            .get(template_id)
            .ok_or_else(|| anyhow::anyhow!("Template not found: {}", template_id))?;

        let builder = template.instantiate().await?;
        return Box::pin(builder.build())
            .await
            .context("Failed to build agent from template");
    }

    let mut builder = AgentBuilder::new();
    if let Some(ref tools) = scenario.tools_allowed {
        for tool in tools {
            builder = builder.enable_tool(tool);
        }
    }
    Box::pin(builder.build())
        .await
        .context("Failed to create agent")
}

fn validate_output_expectations(scenario: &TestScenario, response_text: &str) -> Result<()> {
    if let Some(ref expected) = scenario.expected_output
        && response_text.trim() != expected.trim()
    {
        return Err(anyhow::anyhow!(
            "Output mismatch.\nExpected: {}\nGot: {}",
            expected,
            response_text
        ));
    }

    if let Some(ref contains_list) = scenario.expected_output_contains {
        for expected in contains_list {
            if !response_text.contains(expected) {
                return Err(anyhow::anyhow!(
                    "Output does not contain expected text: '{}'",
                    expected
                ));
            }
        }
    }

    if let Some(ref pattern) = scenario.expected_output_matches {
        use regex::Regex;
        let re = Regex::new(pattern).context("Invalid regex pattern")?;
        if !re.is_match(response_text) {
            return Err(anyhow::anyhow!(
                "Output does not match pattern: {}",
                pattern
            ));
        }
    }
    Ok(())
}

fn validate_tool_expectations(scenario: &TestScenario, tools_executed: &[String]) -> Result<()> {
    let Some(ref expected_tools) = scenario.expected_tools_used else {
        return Ok(());
    };
    let actual_tools: std::collections::HashSet<String> = tools_executed.iter().cloned().collect();
    let expected_tools_set: std::collections::HashSet<String> =
        expected_tools.iter().cloned().collect();

    let missing_tools: Vec<String> = expected_tools_set
        .difference(&actual_tools)
        .cloned()
        .collect();
    if !missing_tools.is_empty() {
        return Err(anyhow::anyhow!(
            "Expected tools were not used: {:?}. Actual tools used: {:?}",
            missing_tools,
            tools_executed
        ));
    }

    let extra_tools: Vec<String> = actual_tools
        .difference(&expected_tools_set)
        .cloned()
        .collect();
    if !extra_tools.is_empty() {
        tracing::warn!(
            "Additional tools were used beyond expected: {:?}",
            extra_tools
        );
    }
    Ok(())
}

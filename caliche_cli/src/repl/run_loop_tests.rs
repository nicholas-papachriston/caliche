//! `ReplEngine::run` end to end with scripted input and a mock model.

use crate::core::slash_commands::SlashCommandResult;
use crate::core::traits::{InputSource, OutputRenderer};
use caliche::llm::ProviderType;
use caliche::testing::MockProvider;
use caliche::{Agent, AgentService, ToolRegistry};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// A line that makes `read_line` fail instead of returning text.
const PROMPT_FAILURE: &str = "<prompt failure>";

#[derive(Clone)]
struct ScriptedInput(Arc<Mutex<VecDeque<String>>>);

#[async_trait::async_trait]
impl InputSource for ScriptedInput {
    async fn read_line(&mut self, _prompt: &str) -> anyhow::Result<Option<String>> {
        match self.0.lock().expect("scripted input").pop_front() {
            Some(line) if line == PROMPT_FAILURE => Err(anyhow::anyhow!("scripted failure")),
            line => Ok(line),
        }
    }

    fn add_history(&mut self, _line: &str) {}

    fn save_history(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}

struct LineCapture(Arc<Mutex<Vec<String>>>);

impl LineCapture {
    fn push(&self, line: &str) {
        self.0.lock().expect("line capture").push(line.to_string());
    }
}

impl OutputRenderer for LineCapture {
    fn handle_command_result(&mut self, _result: SlashCommandResult) {}
    fn render_chunk(&mut self, _text: &str) {}
    fn render_tool_start(&mut self, _name: &str, _args: &serde_json::Value) {}
    fn render_tool_complete(&mut self, _name: &str, _result: &str) {}
    fn render_tool_error(&mut self, _name: &str, _error: &str) {}
    fn render_thinking(&mut self, _content: &str) {}
    fn render_token_usage(&mut self, _usage: &caliche::messages::TokenUsage) {}
    fn show_error(&mut self, message: &str) {
        self.push(message);
    }
    fn show_status(&mut self, message: &str) {
        self.push(message);
    }
    fn clear(&mut self) {}
}

/// What one scripted REPL run left behind.
struct Outcome {
    lines: Vec<String>,
    unread: Vec<String>,
    messages: Vec<(caliche::messages::MessageRole, String)>,
    agent_override: Option<String>,
}

async fn run_script(script: &[&str], agent_override: Option<&str>) -> Outcome {
    run_script_with_token(script, agent_override, None).await
}

#[expect(
    clippy::expect_used,
    reason = "test helper: a failure fails the calling test"
)]
async fn run_script_with_token(
    script: &[&str],
    agent_override: Option<&str>,
    token: Option<&crate::cancellation::CancellationToken>,
) -> Outcome {
    let workspace = tempfile::tempdir().expect("tempdir");
    let mut state = crate::core::state::InteractiveState::new(
        crate::session::CliSession::new("local".to_string(), workspace.path().to_path_buf()),
        "openai",
        "gpt-4o-mini",
        "You are Caliche.".to_string(),
    );
    state.current_agent_override = agent_override.map(str::to_string);
    let queue = Arc::new(Mutex::new(
        script.iter().map(|line| (*line).to_string()).collect(),
    ));
    let lines = Arc::new(Mutex::new(Vec::new()));
    let mut engine = super::ReplEngine::new(
        ScriptedInput(Arc::clone(&queue)),
        LineCapture(Arc::clone(&lines)),
        state,
    );

    let provider = Arc::new(MockProvider::new(vec!["ok".to_string(); 4]));
    let agent = Agent::new(provider, Arc::new(ToolRegistry::new()));
    let mut service =
        AgentService::new_with_agent(agent, ProviderType::OpenAI).expect("mock service");
    let conversation = caliche::persistence::SqliteConversationStore::from_path(
        &workspace.path().join("sessions.db"),
    )
    .await
    .expect("conversation store");
    let session_store = crate::storage::SessionStore::from_pool(conversation.pool())
        .await
        .expect("session store");
    Box::pin(engine.run(
        &mut service,
        session_store,
        Some((Vec::new(), None, None, false, false, false)),
        token,
        None,
    ))
    .await
    .expect("repl run");

    let lines = lines.lock().expect("line capture").clone();
    let unread = queue
        .lock()
        .expect("scripted input")
        .iter()
        .cloned()
        .collect();
    Outcome {
        lines,
        unread,
        messages: engine
            .state
            .messages
            .iter()
            .map(|message| (message.role.clone(), message.extract_text()))
            .collect(),
        agent_override: engine.state.current_agent_override.clone(),
    }
}

fn user_then_ok(text: &str) -> Vec<(caliche::messages::MessageRole, String)> {
    use caliche::messages::MessageRole;
    vec![
        (MessageRole::User, text.to_string()),
        (MessageRole::Assistant, "ok".to_string()),
    ]
}

#[tokio::test]
async fn run_answers_a_message_then_stops_at_end_of_input() {
    let outcome = run_script(&["hello"], None).await;
    assert_eq!(outcome.messages, user_then_ok("hello"));
    assert!(outcome.unread.is_empty());
}

#[tokio::test]
async fn run_stops_at_exit_and_reads_no_further() {
    let outcome = run_script(&["/exit", "hello"], None).await;
    assert!(outcome.messages.is_empty(), "{:?}", outcome.messages);
    assert_eq!(outcome.unread, vec!["hello".to_string()]);
}

#[tokio::test]
async fn run_reports_a_prompt_error_and_stops() {
    let outcome = run_script(&[PROMPT_FAILURE, "hello"], None).await;
    assert!(
        outcome
            .lines
            .contains(&"Prompt error: scripted failure".to_string()),
        "{:?}",
        outcome.lines
    );
    assert_eq!(outcome.unread, vec!["hello".to_string()]);
    assert!(outcome.messages.is_empty());
}

#[tokio::test]
async fn run_clears_a_missing_override_profile_without_a_turn() {
    let outcome = run_script(&["hi"], Some("no-such-profile-xyz")).await;
    assert!(
        outcome
            .lines
            .contains(&"Override profile 'no-such-profile-xyz' not found. Cleared.".to_string()),
        "{:?}",
        outcome.lines
    );
    assert_eq!(outcome.agent_override, None);
    assert!(outcome.messages.is_empty());
}

#[tokio::test]
async fn run_reports_an_unknown_mention_without_a_turn() {
    let outcome = run_script(&["@no-such-profile-xyz do it"], None).await;
    assert!(
        outcome.lines.contains(
            &"Unknown profile: 'no-such-profile-xyz'. Use 'caliche agent list' to see profiles."
                .to_string()
        ),
        "{:?}",
        outcome.lines
    );
    assert!(outcome.messages.is_empty());
}

#[tokio::test]
async fn run_sends_an_at_word_that_is_not_a_profile_name_as_a_message() {
    // `.` is not allowed in a profile name, so this is a context reference,
    // not a subagent mention: the turn runs with the main agent.
    let outcome = run_script(&["@foo.bar hi"], None).await;
    let roles: Vec<_> = outcome
        .messages
        .iter()
        .map(|(role, _)| role.clone())
        .collect();
    assert_eq!(
        roles,
        vec![
            caliche::messages::MessageRole::User,
            caliche::messages::MessageRole::Assistant
        ]
    );
    assert_eq!(outcome.messages[1].1, "ok");
    assert!(
        !outcome.lines.iter().any(|line| line.contains("profile")),
        "{:?}",
        outcome.lines
    );
}

#[tokio::test]
async fn run_keeps_reading_after_a_slash_command() {
    let outcome = run_script(&["/help", "hello"], None).await;
    assert_eq!(outcome.messages, user_then_ok("hello"));
    assert!(outcome.unread.is_empty());
}

#[tokio::test]
async fn run_shows_the_interrupt_hint_once_while_cancelled() {
    let token = crate::cancellation::CancellationToken::new();
    token.cancel();
    let outcome = run_script_with_token(&["/help", "/help"], None, Some(&token)).await;
    let hints = outcome
        .lines
        .iter()
        .filter(|line| line.as_str() == "Press Ctrl+C again to exit.")
        .count();
    assert_eq!(hints, 1, "{:?}", outcome.lines);
    assert!(outcome.unread.is_empty());
}

#[tokio::test]
async fn run_stops_after_a_second_interrupt_and_reads_nothing() {
    let token = crate::cancellation::CancellationToken::new();
    token.cancel();
    // A second Ctrl+C within 250 ms counts as the same keypress.
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    token.cancel();
    assert!(token.exit_requested());
    let outcome = run_script_with_token(&["hello"], None, Some(&token)).await;
    assert_eq!(
        outcome.lines,
        vec!["Interrupted twice. Exiting.".to_string()]
    );
    assert_eq!(outcome.unread, vec!["hello".to_string()]);
    assert!(outcome.messages.is_empty());
}

#[test]
fn subagent_mention_splits_a_profile_name_from_its_task() {
    let cases: &[(&str, Option<(&str, &str)>)] = &[
        ("@reviewer check this", Some(("reviewer", "check this"))),
        (
            "  @ code-review_2   look  ",
            Some(("code-review_2", "look")),
        ),
        ("@reviewer", Some(("reviewer", ""))),
        ("@foo.bar hi", None),
        ("@src/main.rs explain", None),
        ("@", None),
        ("@   ", None),
        ("hello @reviewer", None),
        ("reviewer", None),
    ];
    for (msg, expected) in cases {
        assert_eq!(super::subagent_mention(msg), *expected, "{msg:?}");
    }
}

//! TypeSafe System One (Jev) compaction client and decision types.
//!
//! Inspired by `fast-jev-compaction` (<https://github.com/tamaratran/fast-jev-compaction>).
//! Communicates with the TypeSafe System One API (`https://api.typesafe.ai/v1/systemone`)
//! to evaluate whether tool calls and tool results should be kept verbatim, truncated,
//! or dropped before feeding the conversation history into the summarizer.

use crate::messages::{
    CompactionChangeKind, CompactionReport, Message, MessageContent, MessageRole, ToolCallInfo,
};
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Default System One endpoint URL.
pub const SYSTEM_ONE_URL: &str = "https://api.typesafe.ai/v1/systemone";

/// Default Jev model name.
pub const DEFAULT_JEV_MODEL: &str = "jev-latest";

/// Default preserve recent messages.
pub const DEFAULT_PRESERVE_RECENT_MESSAGES: usize = 6;

/// Default keep probability threshold.
pub const DEFAULT_KEEP_THRESHOLD: f64 = 0.5;

/// Default maximum estimated tokens for state.
pub const DEFAULT_MAX_STATE_TOKENS: usize = 25_000;

/// Default maximum estimated tokens for state plus questions.
pub const DEFAULT_MAX_REQUEST_TOKENS: usize = 30_000;

/// Default head characters to keep when truncating tool output.
pub const DEFAULT_TRUNCATE_HEAD_CHARS: usize = 300;

/// Approximate overhead tokens for Jev request wrapper.
pub const REQUEST_OVERHEAD_TOKENS: usize = 20;

/// Successive caps on serialized tool input in state fitting stages.
pub const INPUT_CHARS_STAGES: [usize; 3] = [1000, 200, 60];
pub const TEXT_HEAD_CHARS: usize = 400;
pub const TEXT_TAIL_CHARS: usize = 150;

/// System instructions embedded in Jev state.
pub const STATE_CONTEXT: &str = "A coding assistant conversation is being compacted to free context. `history` is the whole conversation so far, oldest first; tool outputs are replaced by a short `result` note and long texts may be abridged. Each question asks whether one tool call, or the full output of that call, still needs to stay in the history verbatim. Whatever is not kept is deleted permanently, but the assistant can always re-run a tool or re-read a file.";

/// Options controlling Jev-assisted compaction.
#[derive(Debug, Clone)]
pub struct JevCompactOptions {
    /// Optional goal override (defaults to recent user queries).
    pub goal: Option<String>,
    /// Minimum keep probability for a call or result to stay. Default: 0.5.
    pub keep_threshold: f64,
    /// Number of recent messages to preserve from pruning/truncation. Default: 6.
    pub preserve_recent_messages: usize,
    /// Estimated token ceiling for state payload. Default: 25,000.
    pub max_state_tokens: usize,
    /// Estimated token ceiling for state + questions payload. Default: 30,000.
    pub max_request_tokens: usize,
    /// Characters of a dropped tool result retained before note. Default: 300.
    pub truncate_head_chars: usize,
    /// Minimum reduction ratio required; otherwise considered insufficient. Default: 0.0 (no min reduction requirement).
    pub min_reduction_ratio: f64,
}

impl Default for JevCompactOptions {
    fn default() -> Self {
        Self {
            goal: None,
            keep_threshold: DEFAULT_KEEP_THRESHOLD,
            preserve_recent_messages: DEFAULT_PRESERVE_RECENT_MESSAGES,
            max_state_tokens: DEFAULT_MAX_STATE_TOKENS,
            max_request_tokens: DEFAULT_MAX_REQUEST_TOKENS,
            truncate_head_chars: DEFAULT_TRUNCATE_HEAD_CHARS,
            min_reduction_ratio: 0.0,
        }
    }
}

/// Configuration for Jev-assisted compaction.
#[derive(Debug, Clone)]
pub struct JevCompactionConfig {
    /// Jev asker implementation (e.g. `JevHttpClient`).
    pub asker: Arc<dyn JevAsker>,
    /// Jev compaction options.
    pub options: JevCompactOptions,
}

impl JevCompactionConfig {
    /// Create a new config with the given asker and options.
    #[must_use]
    pub fn new(asker: Arc<dyn JevAsker>, options: JevCompactOptions) -> Self {
        Self { asker, options }
    }

    /// Attempt to construct from environment variables (`TYPESAFE_API_KEY` or `CALICHE_JEV_API_KEY`).
    /// Also checks if Jev compaction is explicitly enabled or key is present.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        let enabled = std::env::var("CALICHE_JEV_COMPACTION")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(true);
        if !enabled {
            return None;
        }

        let client = JevHttpClient::from_env()?;
        let mut opts = JevCompactOptions::default();

        if let Ok(val) = std::env::var("CALICHE_JEV_KEEP_THRESHOLD") {
            if let Ok(parsed) = val.parse::<f64>() {
                opts.keep_threshold = parsed;
            }
        }
        if let Ok(val) = std::env::var("CALICHE_JEV_PRESERVE_RECENT") {
            if let Ok(parsed) = val.parse::<usize>() {
                opts.preserve_recent_messages = parsed;
            }
        }
        if let Ok(val) = std::env::var("CALICHE_JEV_MAX_STATE_TOKENS") {
            if let Ok(parsed) = val.parse::<usize>() {
                opts.max_state_tokens = parsed;
            }
        }
        if let Ok(val) = std::env::var("CALICHE_JEV_MAX_REQUEST_TOKENS") {
            if let Ok(parsed) = val.parse::<usize>() {
                opts.max_request_tokens = parsed;
            }
        }
        if let Ok(val) = std::env::var("CALICHE_JEV_TRUNCATE_HEAD_CHARS") {
            if let Ok(parsed) = val.parse::<usize>() {
                opts.truncate_head_chars = parsed;
            }
        }
        if let Ok(val) = std::env::var("CALICHE_JEV_MIN_REDUCTION_RATIO") {
            if let Ok(parsed) = val.parse::<f64>() {
                opts.min_reduction_ratio = parsed;
            }
        }

        Some(Self {
            asker: Arc::new(client),
            options: opts,
        })
    }
}

/// Noul question type for Jev.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoulQuestion {
    #[serde(rename = "type")]
    pub question_type: String,
    pub instructions: String,
}

/// A Jev question. Currently noul questions are used for binary keep decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum JevQuestion {
    Noul(NoulQuestion),
}

impl JevQuestion {
    /// Create a noul question with the given instruction prompt.
    #[must_use]
    pub fn noul(instructions: impl Into<String>) -> Self {
        Self::Noul(NoulQuestion {
            question_type: "noul".to_string(),
            instructions: instructions.into(),
        })
    }
}

/// Raw request body to TypeSafe System One API.
#[derive(Debug, Clone, Serialize)]
pub struct JevRequestBody<'a> {
    pub model: &'a str,
    pub state: &'a serde_json::Value,
    pub questions: &'a HashMap<String, JevQuestion>,
}

/// Answer from Jev for a single question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JevAnswer {
    #[serde(default)]
    pub noul: Option<f64>,
}

/// Response payload from TypeSafe System One API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JevResponse {
    #[serde(default)]
    pub model: Option<String>,
    pub answers: HashMap<String, JevAnswer>,
    #[serde(default)]
    pub usage: Option<JevUsage>,
}

/// Token usage metadata returned by Jev.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct JevUsage {
    pub input_tokens: Option<usize>,
    pub output_tokens: Option<usize>,
}

/// Trait for asking Jev questions over HTTP or mock transport.
#[async_trait]
pub trait JevAsker: Send + Sync + std::fmt::Debug {
    /// Send state and question set to Jev, receiving typed answers.
    async fn ask(
        &self,
        state: &serde_json::Value,
        questions: &HashMap<String, JevQuestion>,
    ) -> Result<JevResponse>;
}

/// Production HTTP client for TypeSafe System One.
#[derive(Debug, Clone)]
pub struct JevHttpClient {
    api_key: String,
    model: String,
    base_url: String,
    #[cfg(feature = "reqwest")]
    client: reqwest::Client,
}

impl JevHttpClient {
    /// Create a new Jev HTTP client with the given API key, optional model, and optional base URL.
    pub fn new(
        api_key: impl Into<String>,
        model: Option<String>,
        base_url: Option<String>,
    ) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.unwrap_or_else(|| DEFAULT_JEV_MODEL.to_string()),
            base_url: base_url.unwrap_or_else(|| SYSTEM_ONE_URL.to_string()),
            #[cfg(feature = "reqwest")]
            client: reqwest::Client::new(),
        }
    }

    /// Attempt to build from environment variables `TYPESAFE_API_KEY` (or `CALICHE_JEV_API_KEY`).
    /// Also checks if `CALICHE_JEV_COMPACTION=1` is enabled.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        let key = std::env::var("TYPESAFE_API_KEY")
            .or_else(|_| std::env::var("CALICHE_JEV_API_KEY"))
            .ok()?;
        if key.trim().is_empty() {
            return None;
        }
        let model = std::env::var("CALICHE_JEV_MODEL").ok();
        let base_url = std::env::var("CALICHE_JEV_BASE_URL").ok();
        Some(Self::new(key, model, base_url))
    }
}

#[async_trait]
impl JevAsker for JevHttpClient {
    async fn ask(
        &self,
        state: &serde_json::Value,
        questions: &HashMap<String, JevQuestion>,
    ) -> Result<JevResponse> {
        if self.api_key.trim().is_empty() {
            return Err(anyhow!("TYPESAFE_API_KEY is not configured"));
        }

        #[cfg(feature = "reqwest")]
        {
            let body = JevRequestBody {
                model: &self.model,
                state,
                questions,
            };

            let response = self
                .client
                .post(&self.base_url)
                .header(
                    reqwest::header::AUTHORIZATION,
                    format!("Bearer {}", self.api_key.trim()),
                )
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .json(&body)
                .send()
                .await
                .context("failed to send request to TypeSafe System One")?;

            let status = response.status();
            let text = response
                .text()
                .await
                .context("failed to read response body from TypeSafe System One")?;

            if !status.is_success() {
                let preview = if text.len() > 200 {
                    caliche_contracts::text::truncate_utf8(&text, 200)
                } else {
                    &text
                };
                return Err(anyhow!(
                    "Jev request failed with status {status}: {preview}"
                ));
            }

            let parsed: JevResponse =
                serde_json::from_str(&text).context("failed to parse Jev response as JSON")?;

            Ok(parsed)
        }

        #[cfg(not(feature = "reqwest"))]
        {
            let _ = (state, questions);
            Err(anyhow!(
                "reqwest feature is required to make live Jev HTTP calls"
            ))
        }
    }
}

/// Internal paired representation of a tool call and its result.
#[derive(Debug, Clone)]
pub struct JevToolCall {
    /// Short identifier (`t1`, `t2`, ...) used in Jev questions and state.
    pub id: String,
    /// Wire tool call identifier.
    pub tool_use_id: String,
    /// Tool name.
    pub tool_name: String,
    /// Arguments value.
    pub arguments: serde_json::Value,
    /// Message index where the call was initiated.
    pub call_index: usize,
    /// Message index where the result is located.
    pub result_index: usize,
    /// Character count of the result.
    pub result_chars: usize,
    /// Whether the result is an error.
    pub is_error: bool,
    /// Whether this call is pinned (e.g. within preserved recent window or first turn).
    pub pinned: bool,
}

/// Action decision for a tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JevCallAction {
    /// Keep both call and result verbatim.
    KeepBoth,
    /// Keep call, truncate result to head chars + note.
    TruncateResult,
    /// Drop both call and result from summarizer raw input.
    DropBoth,
}

/// Decision outcome for a tool call.
#[derive(Debug, Clone)]
pub struct JevCallDecision {
    pub id: String,
    pub tool_use_id: String,
    pub tool_name: String,
    pub action: JevCallAction,
    pub keep_call: f64,
    pub keep_result: f64,
}

/// Map Jev tool-call decisions onto a machine-readable compaction report.
#[must_use]
pub fn report_from_jev_decisions(decisions: &[JevCallDecision]) -> CompactionReport {
    let mut report = CompactionReport::new();
    for decision in decisions {
        match decision.action {
            JevCallAction::KeepBoth => {}
            JevCallAction::TruncateResult => {
                report.record(
                    CompactionChangeKind::Truncated,
                    "jev",
                    format!(
                        "truncate tool result {} ({})",
                        decision.tool_use_id, decision.tool_name
                    ),
                );
            }
            JevCallAction::DropBoth => {
                report.record(
                    CompactionChangeKind::Omitted,
                    "jev",
                    format!(
                        "drop tool call {} ({})",
                        decision.tool_use_id, decision.tool_name
                    ),
                );
            }
        }
    }
    report
}

/// Check if a message index is pinned.
#[must_use]
pub fn is_pinned(index: usize, total: usize, preserve_recent: usize) -> bool {
    index == 0 || index >= total.saturating_sub(preserve_recent)
}

/// Scan conversation messages and pair tool calls with their corresponding tool results.
pub fn collect_tool_calls(messages: &[Message], preserve_recent: usize) -> Vec<JevToolCall> {
    let mut results: HashMap<String, (usize, usize, bool)> = HashMap::new();

    for (index, msg) in messages.iter().enumerate() {
        if matches!(msg.role, MessageRole::Tool) {
            let tool_id = match &msg.content {
                MessageContent::Structured { tool_call_id, .. } => tool_call_id
                    .as_deref()
                    .map(std::string::ToString::to_string),
                _ => None,
            };
            if let Some(id) = tool_id {
                let text = msg.extract_text();
                let is_error =
                    text.starts_with("Error:") || text.starts_with("Tool call validation failed");
                results.insert(id, (index, text.len(), is_error));
            }
        }
    }

    let mut calls = Vec::new();
    let total = messages.len();

    for (call_index, msg) in messages.iter().enumerate() {
        if let Some(tool_calls) = msg.content.as_tool_calls() {
            for tc in tool_calls {
                let tool_id = tc.id.to_string();
                if let Some(&(result_index, result_chars, is_error)) = results.get(&tool_id) {
                    let id = format!("t{}", calls.len().saturating_add(1));
                    let pinned = is_pinned(call_index, total, preserve_recent)
                        || is_pinned(result_index, total, preserve_recent);
                    calls.push(JevToolCall {
                        id,
                        tool_use_id: tool_id,
                        tool_name: tc.name.to_string(),
                        arguments: tc.arguments.clone(),
                        call_index,
                        result_index,
                        result_chars,
                        is_error,
                        pinned,
                    });
                }
            }
        }
    }

    calls
}

/// Construct the two noul questions for a single tool call.
#[must_use]
pub fn questions_for(call: &JevToolCall) -> HashMap<String, JevQuestion> {
    let mut map = HashMap::new();
    map.insert(
        format!("call_{}", call.id),
        JevQuestion::noul(format!(
            "Tool call {} ({}) should stay in the history: knowing this call was made, with its input, still matters for what the assistant does next",
            call.id, call.tool_name
        )),
    );
    map.insert(
        format!("result_{}", call.id),
        JevQuestion::noul(format!(
            "The full output of tool call {} ({}, {} chars) should stay in the history verbatim: the assistant still needs its contents and re-running the tool would not do",
            call.id, call.tool_name, call.result_chars
        )),
    );
    map
}

/// Rough token estimation matching fast-jev-compaction heuristics:
/// - Words (letters): 1 token per 6 letters
/// - Digits: 0.5 token per digit
/// - Other symbols: 0.9 token per char
#[must_use]
pub fn estimate_tokens(text: &str) -> usize {
    let mut tokens = 0.0f64;
    let mut char_iter = text.chars().peekable();

    while let Some(&ch) = char_iter.peek() {
        if ch.is_ascii_alphabetic() {
            let mut count = 0usize;
            while let Some(&c) = char_iter.peek() {
                if c.is_ascii_alphabetic() {
                    count = count.saturating_add(1);
                    char_iter.next();
                } else {
                    break;
                }
            }
            let word_tokens = 1.0 + ((count.saturating_sub(1)) / 6) as f64;
            tokens += word_tokens;
        } else if ch.is_ascii_digit() {
            let mut count = 0usize;
            while let Some(&c) = char_iter.peek() {
                if c.is_ascii_digit() {
                    count = count.saturating_add(1);
                    char_iter.next();
                } else {
                    break;
                }
            }
            tokens += (count as f64) * 0.5;
        } else if ch.is_whitespace() {
            char_iter.next();
        } else {
            tokens += 0.9;
            char_iter.next();
        }
    }

    tokens.ceil() as usize
}

fn truncate_str(text: &str, limit: usize) -> String {
    if text.len() <= limit {
        text.to_string()
    } else {
        let keep = limit.saturating_sub(1);
        let mut s = String::with_capacity(limit + 3);
        s.push_str(caliche_contracts::text::truncate_utf8(text, keep));
        s.push('…');
        s
    }
}

fn abridge_str(text: &str, head: usize, tail: usize) -> String {
    if text.len() <= head + tail + 40 {
        return text.to_string();
    }
    let omitted = text.len().saturating_sub(head + tail);
    format!(
        "{}\n[… {} chars omitted …]\n{}",
        caliche_contracts::text::truncate_utf8(text, head),
        omitted,
        &text[text.ceil_char_boundary(text.len().saturating_sub(tail))..]
    )
}

/// Extract default goal from recent user messages.
#[must_use]
pub fn goal_from_messages(messages: &[Message]) -> String {
    messages
        .iter()
        .filter(|m| matches!(m.role, MessageRole::User) && !m.extract_text().trim().is_empty())
        .rev()
        .take(3)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(|m| truncate_str(&m.extract_text(), 500))
        .collect::<Vec<_>>()
        .join("\n")
}

/// History entry for Jev state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JevHistoryEntry {
    pub i: usize,
    pub role: String,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
}

fn build_history_entries(
    messages: &[Message],
    calls: &[JevToolCall],
    input_limit: usize,
) -> Vec<JevHistoryEntry> {
    let mut by_msg: HashMap<usize, Vec<&JevToolCall>> = HashMap::new();
    for call in calls {
        by_msg.entry(call.call_index).or_default().push(call);
    }

    let mut entries = Vec::with_capacity(messages.len());
    for (i, msg) in messages.iter().enumerate() {
        let role = match msg.role {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
        };
        let text = msg.extract_text();
        let tool_calls = by_msg.get(&i).map(|list| {
            let list_val: Vec<serde_json::Value> = list
                .iter()
                .map(|c| {
                    let mut input_str = serde_json::to_string(&c.arguments).unwrap_or_default();
                    if input_str.len() > input_limit {
                        input_str = truncate_str(&input_str, input_limit);
                    }
                    serde_json::json!({
                        "id": c.id,
                        "tool": c.tool_name,
                        "input": input_str,
                        "result": format!("{}, {} chars (omitted)", if c.is_error { "error" } else { "ok" }, c.result_chars),
                    })
                })
                .collect();
            serde_json::Value::Array(list_val)
        });

        if text.trim().is_empty() && tool_calls.is_none() {
            continue;
        }

        entries.push(JevHistoryEntry {
            i,
            role: role.to_string(),
            text,
            tool_calls,
        });
    }

    entries
}

/// Fit state into token budget across stages.
pub fn fit_state(
    messages: &[Message],
    calls: &[JevToolCall],
    options: &JevCompactOptions,
) -> Result<(serde_json::Value, usize)> {
    let goal = options
        .goal
        .clone()
        .unwrap_or_else(|| goal_from_messages(messages));

    for &limit in &INPUT_CHARS_STAGES {
        let history = build_history_entries(messages, calls, limit);
        let state = serde_json::json!({
            "context": STATE_CONTEXT,
            "goal": goal,
            "history": history,
        });
        let serialized = serde_json::to_string(&state)?;
        let tokens = estimate_tokens(&serialized);
        if tokens <= options.max_state_tokens {
            return Ok((state, tokens));
        }
    }

    // Abridge long texts
    let mut history = build_history_entries(messages, calls, INPUT_CHARS_STAGES[2]);
    for entry in &mut history {
        if !is_pinned(entry.i, messages.len(), options.preserve_recent_messages)
            && entry.text.len() > TEXT_HEAD_CHARS + TEXT_TAIL_CHARS + 40
        {
            entry.text = abridge_str(&entry.text, TEXT_HEAD_CHARS, TEXT_TAIL_CHARS);
        }
    }
    let state = serde_json::json!({
        "context": STATE_CONTEXT,
        "goal": goal,
        "history": history,
    });
    let serialized = serde_json::to_string(&state)?;
    let tokens = estimate_tokens(&serialized);
    if tokens <= options.max_state_tokens {
        return Ok((state, tokens));
    }

    // Collapse non-pinned message texts
    for entry in &mut history {
        if !is_pinned(entry.i, messages.len(), options.preserve_recent_messages)
            && !entry.text.is_empty()
        {
            let orig_len = messages[entry.i].extract_text().len();
            entry.text = format!("[… {orig_len} chars omitted …]");
        }
    }
    let state = serde_json::json!({
        "context": STATE_CONTEXT,
        "goal": goal,
        "history": history,
    });
    let serialized = serde_json::to_string(&state)?;
    let tokens = estimate_tokens(&serialized);
    if tokens <= options.max_state_tokens {
        return Ok((state, tokens));
    }

    Err(anyhow!(
        "history too large for Jev (~{tokens} tokens after fitting stages, limit {})",
        options.max_state_tokens
    ))
}

/// Partition tool calls into batches where state + batch questions <= max_request_tokens.
pub fn batch_calls(
    calls: &[JevToolCall],
    state_tokens: usize,
    max_request_tokens: usize,
) -> Result<Vec<Vec<JevToolCall>>> {
    let budget = max_request_tokens
        .saturating_sub(state_tokens)
        .saturating_sub(REQUEST_OVERHEAD_TOKENS);

    let mut batches: Vec<Vec<JevToolCall>> = Vec::new();
    let mut current: Vec<JevToolCall> = Vec::new();
    let mut current_tokens = 0usize;

    for call in calls {
        let q = questions_for(call);
        let q_str = serde_json::to_string(&q)?;
        let q_tokens = estimate_tokens(&q_str);

        if !current.is_empty() && current_tokens.saturating_add(q_tokens) > budget {
            batches.push(std::mem::take(&mut current));
            current_tokens = 0;
        }

        if current.is_empty() && q_tokens > budget {
            return Err(anyhow!(
                "state leaves no room for questions (~{state_tokens} tokens of {max_request_tokens})"
            ));
        }

        current.push(call.clone());
        current_tokens = current_tokens.saturating_add(q_tokens);
    }

    if !current.is_empty() {
        batches.push(current);
    }

    Ok(batches)
}

/// Evaluate answer to decide whether to keep both, truncate result, or drop both.
#[must_use]
pub fn decide_call(call: &JevToolCall, answer: (f64, f64), threshold: f64) -> JevCallDecision {
    if call.pinned {
        return JevCallDecision {
            id: call.id.clone(),
            tool_use_id: call.tool_use_id.clone(),
            tool_name: call.tool_name.clone(),
            action: JevCallAction::KeepBoth,
            keep_call: 1.0,
            keep_result: 1.0,
        };
    }

    let (keep_call, keep_result) = answer;
    let action = if keep_result >= threshold {
        JevCallAction::KeepBoth
    } else if keep_call >= threshold {
        JevCallAction::TruncateResult
    } else {
        JevCallAction::DropBoth
    };

    JevCallDecision {
        id: call.id.clone(),
        tool_use_id: call.tool_use_id.clone(),
        tool_name: call.tool_name.clone(),
        action,
        keep_call,
        keep_result,
    }
}

/// Ask Jev questions across batches and collect decisions.
pub async fn evaluate_tool_decisions(
    asker: &dyn JevAsker,
    messages: &[Message],
    calls: &[JevToolCall],
    options: &JevCompactOptions,
) -> Result<Vec<JevCallDecision>> {
    let candidates: Vec<&JevToolCall> = calls.iter().filter(|c| !c.pinned).collect();
    if candidates.is_empty() {
        return Ok(calls
            .iter()
            .map(|c| decide_call(c, (1.0, 1.0), options.keep_threshold))
            .collect());
    }

    let (state, state_tokens) = fit_state(messages, calls, options)?;
    let unpinned_calls: Vec<JevToolCall> = candidates.into_iter().cloned().collect();
    let batches = batch_calls(&unpinned_calls, state_tokens, options.max_request_tokens)?;

    let mut answers_map: HashMap<String, (f64, f64)> = HashMap::new();

    for batch in batches {
        let mut questions = HashMap::new();
        for call in &batch {
            questions.extend(questions_for(call));
        }

        let resp = asker.ask(&state, &questions).await?;
        for call in &batch {
            let call_key = format!("call_{}", call.id);
            let res_key = format!("result_{}", call.id);

            let keep_call = resp
                .answers
                .get(&call_key)
                .and_then(|a| a.noul)
                .ok_or_else(|| anyhow!("missing or invalid answer for {call_key}"))?;
            let keep_result = resp
                .answers
                .get(&res_key)
                .and_then(|a| a.noul)
                .ok_or_else(|| anyhow!("missing or invalid answer for {res_key}"))?;

            answers_map.insert(call.id.clone(), (keep_call, keep_result));
        }
    }

    let decisions = calls
        .iter()
        .map(|call| {
            let ans = answers_map.get(&call.id).copied().unwrap_or((1.0, 1.0));
            decide_call(call, ans, options.keep_threshold)
        })
        .collect();

    Ok(decisions)
}

/// Truncates tool result text with an explanatory note.
#[must_use]
pub fn truncate_tool_result_text(text: &str, is_error: bool, head_chars: usize) -> String {
    if text.len() <= head_chars.saturating_add(120) {
        return text.to_string();
    }
    let head = if head_chars > 0 {
        format!(
            "{}\n",
            caliche_contracts::text::truncate_utf8(text, head_chars)
        )
    } else {
        String::new()
    };
    let omitted = text.len().saturating_sub(head_chars);
    format!(
        "{head}[fast-jev-compaction truncated {omitted} chars of this tool result{}; re-run the tool if needed]",
        if is_error { " (error)" } else { "" }
    )
}

/// Build filtered messages for the summarizer based on Jev decisions.
///
/// In the hybrid pipeline:
/// - Pinned tool calls/results stay verbatim.
/// - Kept tool calls/results stay verbatim.
/// - Truncated results have their content truncated to `head_chars` + note, keeping call arguments.
/// - Dropped tool calls and results are removed from the summarizer's view so the summarizer is not polluted by stale voluminous outputs.
/// - User and assistant texts stay verbatim and in order.
pub fn filter_messages_for_summarizer(
    messages: &[Message],
    decisions: &[JevCallDecision],
    head_chars: usize,
) -> Vec<Message> {
    let mut actions: HashMap<&str, JevCallAction> = HashMap::new();
    for d in decisions {
        actions.insert(d.tool_use_id.as_str(), d.action);
    }

    let mut filtered = Vec::with_capacity(messages.len());

    for msg in messages {
        match msg.role {
            MessageRole::Tool => {
                let tool_id = match &msg.content {
                    MessageContent::Structured { tool_call_id, .. } => tool_call_id.as_deref(),
                    _ => None,
                };

                if let Some(id) = tool_id {
                    let action = actions.get(id).copied().unwrap_or(JevCallAction::KeepBoth);
                    match action {
                        JevCallAction::DropBoth => {
                            // Omit this tool result entirely from summarizer input
                            continue;
                        }
                        JevCallAction::TruncateResult => {
                            let orig_text = msg.extract_text();
                            let is_error = orig_text.starts_with("Error:");
                            let truncated =
                                truncate_tool_result_text(&orig_text, is_error, head_chars);
                            filtered.push(Message {
                                role: MessageRole::Tool,
                                content: MessageContent::Structured {
                                    text: Some(Arc::new(truncated)),
                                    tool_calls: None,
                                    tool_call_id: Some(beef::lean::Cow::owned(id.to_string())),
                                },
                                metadata: msg.metadata.clone(),
                            });
                            continue;
                        }
                        JevCallAction::KeepBoth => {
                            filtered.push(msg.clone());
                            continue;
                        }
                    }
                }

                filtered.push(msg.clone());
            }
            MessageRole::Assistant => {
                if let Some(tool_calls) = msg.content.as_tool_calls() {
                    let mut kept_calls: Vec<ToolCallInfo> = Vec::new();
                    for tc in tool_calls {
                        let id = tc.id.as_ref();
                        let action = actions.get(id).copied().unwrap_or(JevCallAction::KeepBoth);
                        if action != JevCallAction::DropBoth {
                            kept_calls.push(tc.clone());
                        }
                    }

                    let text = msg.content.as_text().map(|s| Arc::new(s.to_string()));

                    if text.is_none() && kept_calls.is_empty() {
                        // Empty assistant message after dropping tool calls: omit
                        continue;
                    }

                    filtered.push(Message {
                        role: MessageRole::Assistant,
                        content: MessageContent::Structured {
                            text,
                            tool_calls: if kept_calls.is_empty() {
                                None
                            } else {
                                Some(kept_calls)
                            },
                            tool_call_id: None,
                        },
                        metadata: msg.metadata.clone(),
                    });
                } else {
                    filtered.push(msg.clone());
                }
            }
            _ => {
                filtered.push(msg.clone());
            }
        }
    }

    filtered
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        let tokens = estimate_tokens("hello world 12345 !?");
        assert!(tokens > 0);
    }

    #[test]
    fn test_truncate_tool_result_text() {
        let text = "a".repeat(500);
        let truncated = truncate_tool_result_text(&text, false, 50);
        assert!(truncated.contains("[fast-jev-compaction truncated"));
        assert!(truncated.len() < 300);
    }

    #[test]
    fn test_decide_call() {
        let call = JevToolCall {
            id: "t1".to_string(),
            tool_use_id: "call_1".to_string(),
            tool_name: "test".to_string(),
            arguments: serde_json::json!({}),
            call_index: 1,
            result_index: 2,
            result_chars: 500,
            is_error: false,
            pinned: false,
        };

        let d1 = decide_call(&call, (0.8, 0.9), 0.5);
        assert_eq!(d1.action, JevCallAction::KeepBoth);

        let d2 = decide_call(&call, (0.8, 0.2), 0.5);
        assert_eq!(d2.action, JevCallAction::TruncateResult);

        let d3 = decide_call(&call, (0.1, 0.1), 0.5);
        assert_eq!(d3.action, JevCallAction::DropBoth);
    }

    #[tokio::test]
    async fn test_jev_mock_asker_and_fallback() {
        struct MockAsker {
            should_fail: bool,
        }

        impl std::fmt::Debug for MockAsker {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "MockAsker")
            }
        }

        #[async_trait::async_trait]
        impl JevAsker for MockAsker {
            async fn ask(
                &self,
                _state: &serde_json::Value,
                questions: &HashMap<String, JevQuestion>,
            ) -> anyhow::Result<JevResponse> {
                if self.should_fail {
                    return Err(anyhow!("simulated network failure"));
                }
                let mut answers = HashMap::new();
                for k in questions.keys() {
                    answers.insert(k.clone(), JevAnswer { noul: Some(0.8) });
                }
                Ok(JevResponse {
                    model: Some("jev-latest".to_string()),
                    answers,
                    usage: None,
                })
            }
        }

        let msgs = vec![
            Message::system("sys"),
            Message::user("u"),
            Message {
                role: MessageRole::Assistant,
                content: MessageContent::Structured {
                    text: None,
                    tool_calls: Some(vec![ToolCallInfo::from_json("t1", "test", "{}")]),
                    tool_call_id: None,
                },
                metadata: None,
            },
            Message {
                role: MessageRole::Tool,
                content: MessageContent::Structured {
                    text: Some(Arc::new("res".to_string())),
                    tool_calls: None,
                    tool_call_id: Some(beef::lean::Cow::owned("t1".to_string())),
                },
                metadata: None,
            },
        ];

        let calls = collect_tool_calls(&msgs, 0);
        let opts = JevCompactOptions::default();

        let ok_asker = MockAsker { should_fail: false };
        let decisions = evaluate_tool_decisions(&ok_asker, &msgs, &calls, &opts)
            .await
            .expect("evaluate decisions");
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].action, JevCallAction::KeepBoth);

        let err_asker = MockAsker { should_fail: true };
        assert!(
            evaluate_tool_decisions(&err_asker, &msgs, &calls, &opts)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn test_jev_insufficient_reduction_ratio_fallback() {
        use crate::agent::core::compaction::CompactionStrategy;
        use crate::agent::memory::TokenCounter;
        use crate::agent::memory::token_counter::ApproximateTokenCounter;
        use crate::testing::mock_provider::MockProvider;

        #[derive(Debug)]
        struct KeepAllAsker;

        #[async_trait::async_trait]
        impl JevAsker for KeepAllAsker {
            async fn ask(
                &self,
                _state: &serde_json::Value,
                questions: &HashMap<String, JevQuestion>,
            ) -> anyhow::Result<JevResponse> {
                let mut answers = HashMap::new();
                for k in questions.keys() {
                    answers.insert(k.clone(), JevAnswer { noul: Some(0.99) });
                }
                Ok(JevResponse {
                    model: Some("jev-latest".to_string()),
                    answers,
                    usage: None,
                })
            }
        }

        let counter = ApproximateTokenCounter::default();
        let messages = vec![
            Message::system("sys"),
            Message::user("do something"),
            Message {
                role: MessageRole::Assistant,
                content: MessageContent::Structured {
                    text: None,
                    tool_calls: Some(vec![ToolCallInfo::from_json("t1", "test", "{}")]),
                    tool_call_id: None,
                },
                metadata: None,
            },
            Message {
                role: MessageRole::Tool,
                content: MessageContent::Structured {
                    text: Some(Arc::new("res".to_string())),
                    tool_calls: None,
                    tool_call_id: Some(beef::lean::Cow::owned("t1".to_string())),
                },
                metadata: None,
            },
            Message::user("recent turn"),
            Message::assistant("recent answer"),
        ];

        let before = counter.estimate_messages_tokens(&messages);
        let provider = Arc::new(MockProvider::new(vec!["Fallback summary".to_string()]));

        let strategy = crate::agent::core::compaction::LlmSummarizationCompaction::builder()
            .provider(provider)
            .preserve_recent(2)
            .max_summary_tokens(100)
            .jev_asker(Arc::new(KeepAllAsker))
            .jev_options(JevCompactOptions {
                min_reduction_ratio: 0.5, // Requires at least 50% candidate reduction; KeepAll gives 0%
                ..Default::default()
            })
            .build()
            .expect("build strategy");

        let result = strategy
            .compact(&messages, before / 2, &counter)
            .await
            .expect("compact with fallback");

        assert_eq!(
            result.compaction_type,
            crate::agent::core::compaction::CompactionType::LlmSummarization
        );
        assert!(result.summary.is_some());
        // Since KeepAll gave 0% reduction < 50%, summarizer received all messages verbatim and produced summary
        assert_eq!(result.messages.len(), 3);
    }

    #[test]
    #[serial_test::serial(env)]
    fn test_jev_client_from_env() {
        let mut env = crate::test_env::EnvGuard::acquire();
        env.remove("TYPESAFE_API_KEY");
        env.remove("CALICHE_JEV_API_KEY");
        assert!(JevHttpClient::from_env().is_none());

        env.set("TYPESAFE_API_KEY", "test-key-123");
        let client = JevHttpClient::from_env();
        assert!(client.is_some());
        if let Some(c) = client {
            assert_eq!(c.api_key, "test-key-123");
        }
    }
}

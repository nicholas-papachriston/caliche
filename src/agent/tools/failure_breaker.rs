//! Stop a turn from re-running a command that already failed the same way.
//!
//! The first failure is executed and its stderr stays in the model context.
//! A later attempt with the same bash command (and working directory), or the
//! same arguments for any other tool, is not executed once
//! [`TRIP_AFTER_FAILURES`] identical failures have been recorded. The tool
//! result is [`RETRY_BLOCKED_CODE`] and does not repeat the stderr. A
//! different command string runs.
//!
//! The block clears when a successful mutating bash command runs in that
//! failure's working directory, when a file tool writes a path the failure
//! named, or when the mtime or content hash of a named path changes.
//! An omitted `working_directory` is the process working directory, so two
//! bash calls that both leave it unset share a directory. Bare relative
//! names such as `main.py` count as those paths, including for the mtime
//! check. A Python `open` for write counts as a mutating command.
//! Transient network failures (timeouts, HTTP 408/429/5xx on fetch tools)
//! are not recorded. Set [`BREAKER_ENV`] to `0` to disable the breaker.
//! Outcome lines stay on unless `RUST_LOG` drops `caliche::tool_outcome`.
//!
//! Outcome lines are one JSON object each on the `caliche::tool_outcome`
//! tracing target, which the REPL writes to `repl-*.log`. Logged stderr is
//! redacted with [`TranscriptRedactor::with_default_patterns`].

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

use crate::agent::session::TranscriptRedactor;
use crate::agent::tools::names::{
    TOOL_APPLY_PATCH, TOOL_BASH, TOOL_FILESYSTEM, TOOL_HTTP, TOOL_WEB_FETCH, TOOL_WEB_SEARCH,
    is_fs_write_op,
};
use crate::agent::tools::tool::ToolResult;

/// Model-facing code for a failure that was not executed again.
pub const RETRY_BLOCKED_CODE: &str = "retry_blocked_until_code_change";

/// Env var that disables the breaker when set to `0`, `false`, `off`, `no`, or `disabled`.
pub const BREAKER_ENV: &str = "CALICHE_IDENTICAL_FAILURE_BREAKER";

/// Tracing target for JSON tool outcome lines.
pub const OUTCOME_TARGET: &str = "caliche::tool_outcome";

const MAX_RECORDS: usize = 64;
const STDERR_HASH_BYTES: usize = 6;
const STDERR_LOG_CHARS: usize = 160;
const COMMAND_LABEL_CHARS: usize = 120;
/// Block the next identical attempt after this many real failures.
const TRIP_AFTER_FAILURES: u32 = 1;
const STAMP_LIMIT: usize = 16;
const STAMP_HASH_MAX_BYTES: u64 = 256 * 1024;
const WRITE_FILE: &str = "write_file";
const EDIT_FILE: &str = "edit_file";

tokio::task_local! {
    static TURN_SCOPE: TurnScope;
}

#[cfg(test)]
tokio::task_local! {
    static BREAKER_ENV_INJECT: Option<&'static str>;
}

/// Run `fut` with an injected breaker env value so tests do not read the process.
#[cfg(test)]
async fn with_injected_breaker_env<F>(value: Option<&'static str>, fut: F) -> F::Output
where
    F: std::future::Future,
{
    BREAKER_ENV_INJECT.scope(value, fut).await
}

struct TurnScope {
    breaker: Arc<Mutex<IdenticalFailureBreaker>>,
    action_ids: Arc<Mutex<HashMap<String, String>>>,
}

/// Run `fut` with this turn's failure breaker and action-id map installed.
pub async fn with_turn_scope<F>(
    breaker: Arc<Mutex<IdenticalFailureBreaker>>,
    action_ids: Arc<Mutex<HashMap<String, String>>>,
    fut: F,
) -> F::Output
where
    F: std::future::Future,
{
    TURN_SCOPE
        .scope(
            TurnScope {
                breaker,
                action_ids,
            },
            fut,
        )
        .await
}

/// One user turn's identical-failure memory.
#[derive(Debug)]
pub struct IdenticalFailureBreaker {
    enabled: bool,
    records: HashMap<String, FailureRecord>,
    order: VecDeque<String>,
}

#[derive(Debug, Clone)]
struct FailureRecord {
    seen: u32,
    stderr_hash: String,
    exit_code: Option<i64>,
    label: String,
    cwd: Option<String>,
    relevant_paths: Vec<String>,
    stamps: Vec<PathStamp>,
}

#[derive(Debug, Clone)]
struct PathStamp {
    path: String,
    mtime_nanos: Option<u128>,
    content_hash: Option<String>,
}

impl Default for IdenticalFailureBreaker {
    fn default() -> Self {
        Self::new()
    }
}

impl IdenticalFailureBreaker {
    /// Empty breaker for a new user turn. Disabled when [`BREAKER_ENV`] is off.
    #[must_use]
    pub fn new() -> Self {
        Self::from_env_value(live_breaker_env().as_deref())
    }

    /// Breaker from an injected env value. `None` means the variable is unset.
    fn from_env_value(value: Option<&str>) -> Self {
        Self::with_enabled(value.is_none_or(env_enables_breaker))
    }

    /// Empty breaker that ignores the process environment.
    #[must_use]
    pub fn with_enabled(enabled: bool) -> Self {
        Self {
            enabled,
            records: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    /// Cached failure when this call would repeat a recorded failure.
    #[must_use]
    pub fn blocked(&mut self, tool: &str, args: &Value) -> Option<ToolResult> {
        if !self.enabled {
            return None;
        }
        let key = fingerprint(tool, args);
        let changed = self
            .records
            .get(&key)
            .is_some_and(|record| stamps_changed(&record.stamps));
        if changed {
            self.forget(&key);
            return None;
        }
        let record = self.records.get(&key)?;
        if record.seen < TRIP_AFTER_FAILURES {
            return None;
        }
        Some(cached_failure(tool, record))
    }

    /// Remember a real failure, or clear records a successful edit touches.
    pub fn observe(&mut self, tool: &str, args: &Value, result: &ToolResult) {
        if !self.enabled {
            return;
        }
        if result.success {
            self.note_success(tool, args);
            return;
        }
        if is_retry_blocked(result) || is_transient(tool, result) {
            return;
        }
        self.note_failure(tool, args, result);
    }

    fn note_failure(&mut self, tool: &str, args: &Value, result: &ToolResult) {
        let key = fingerprint(tool, args);
        let body = failure_body(result);
        let cwd = logical_cwd(args);
        let relevant_paths = relevant_paths(args, &body);
        let stamps = stamp_paths(cwd.as_deref(), &relevant_paths);
        if let Some(existing) = self.records.get_mut(&key) {
            existing.seen = existing.seen.saturating_add(1);
            existing.stderr_hash = stderr_hash(&body);
            existing.exit_code = exit_code_of(result);
            existing.cwd = cwd;
            existing.relevant_paths = relevant_paths;
            existing.stamps = stamps;
            return;
        }
        let record = FailureRecord {
            seen: 1,
            stderr_hash: stderr_hash(&body),
            exit_code: exit_code_of(result),
            label: failure_label(tool, args),
            cwd,
            relevant_paths,
            stamps,
        };
        self.records.insert(key.clone(), record);
        self.order.push_back(key);
        while self.order.len() > MAX_RECORDS {
            if let Some(oldest) = self.order.pop_front() {
                self.records.remove(&oldest);
            }
        }
    }

    fn note_success(&mut self, tool: &str, args: &Value) {
        if tool == TOOL_BASH {
            self.note_bash_success(args);
            return;
        }
        let paths = mutation_paths(tool, args);
        if paths.is_empty() {
            if tool == TOOL_APPLY_PATCH {
                self.records.clear();
                self.order.clear();
            }
            return;
        }
        self.records
            .retain(|_, record| !paths.iter().any(|path| record.touched_by(path)));
        self.drop_forgotten_order();
    }

    fn note_bash_success(&mut self, args: &Value) {
        let Some(command) = args.get("command").and_then(Value::as_str) else {
            return;
        };
        if !bash_command_mutates(command) {
            return;
        }
        let cwd = logical_cwd(args);
        let paths = extract_paths(command);
        self.records.retain(|_, record| {
            if cwd_same(record.cwd.as_deref(), cwd.as_deref()) {
                return false;
            }
            !paths.iter().any(|path| record.touched_by(path))
        });
        self.drop_forgotten_order();
    }

    fn forget(&mut self, key: &str) {
        self.records.remove(key);
        self.drop_forgotten_order();
    }

    fn drop_forgotten_order(&mut self) {
        self.order.retain(|key| self.records.contains_key(key));
    }
}

impl FailureRecord {
    fn touched_by(&self, mutated: &str) -> bool {
        let mutated = normalize_path(mutated);
        if mutated.is_empty() {
            return false;
        }
        if self.relevant_paths.is_empty() {
            return true;
        }
        self.relevant_paths
            .iter()
            .any(|path| paths_overlap(path, &mutated))
    }
}

/// Return a cached block, logging it, when the turn already saw this failure.
#[must_use]
pub fn blocked_result(tool: &str, call_id: &str, args: &Value) -> Option<ToolResult> {
    let blocked = TURN_SCOPE
        .try_with(|scope| {
            scope
                .breaker
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .blocked(tool, args)
        })
        .ok()
        .flatten()?;
    log_tool_end(tool, call_id, args, &blocked);
    Some(blocked)
}

/// Record the outcome against the current turn's breaker, if one is installed.
pub fn observe(tool: &str, args: &Value, result: &ToolResult) {
    let _ = TURN_SCOPE.try_with(|scope| {
        scope
            .breaker
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .observe(tool, args, result);
    });
}

/// JSON line for `tool.start`.
pub fn log_tool_start(tool: &str, call_id: &str, args: &Value) {
    emit(&OutcomeLine {
        event: "tool.start",
        action_id: current_action_id(call_id),
        tool: tool.to_string(),
        ok: None,
        exit_code: None,
        cwd: cwd_of(args),
        stderr_hash: None,
        stderr: None,
        awaiting_confirmation: false,
        retry_blocked: false,
    });
}

/// JSON line for `tool.end`, including failed and blocked results.
pub fn log_tool_end(tool: &str, call_id: &str, args: &Value, result: &ToolResult) {
    let failed = !result.success;
    let retry_blocked = is_retry_blocked(result);
    let body = if failed {
        failure_body(result)
    } else {
        Cow::Borrowed("")
    };
    let logged_stderr = if failed {
        let redacted = redact_for_log(&body);
        let text = truncate_text(&redacted, STDERR_LOG_CHARS);
        (!text.is_empty()).then_some(text)
    } else {
        None
    };
    let logged_hash = if retry_blocked {
        payload_string(result, "stderr_hash")
    } else if failed {
        Some(stderr_hash(&body))
    } else {
        None
    };
    emit(&OutcomeLine {
        event: "tool.end",
        action_id: current_action_id(call_id),
        tool: tool.to_string(),
        ok: Some(result.success),
        exit_code: exit_code_of(result),
        cwd: cwd_of(args),
        stderr_hash: logged_hash,
        stderr: logged_stderr,
        awaiting_confirmation: false,
        retry_blocked,
    });
}

/// JSON line emitted when a tool pauses for human approval.
pub fn log_awaiting_confirmation(tool: &str, call_id: &str, args: &Value) {
    emit(&OutcomeLine {
        event: "tool.awaiting_confirmation",
        action_id: current_action_id(call_id),
        tool: tool.to_string(),
        ok: None,
        exit_code: None,
        cwd: cwd_of(args),
        stderr_hash: None,
        stderr: None,
        awaiting_confirmation: true,
        retry_blocked: false,
    });
}

#[derive(Serialize)]
struct OutcomeLine {
    event: &'static str,
    action_id: String,
    tool: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    ok: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    exit_code: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cwd: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stderr_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stderr: Option<String>,
    awaiting_confirmation: bool,
    retry_blocked: bool,
}

fn emit(line: &OutcomeLine) {
    match serde_json::to_string(line) {
        Ok(json) => tracing::info!(target: OUTCOME_TARGET, "{json}"),
        Err(err) => {
            tracing::warn!(
                target: OUTCOME_TARGET,
                "failed to encode tool outcome: {err}"
            );
        }
    }
}

fn current_action_id(call_id: &str) -> String {
    TURN_SCOPE
        .try_with(|scope| {
            scope
                .action_ids
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get(call_id)
                .cloned()
                .unwrap_or_default()
        })
        .unwrap_or_default()
}

fn cached_failure(tool: &str, record: &FailureRecord) -> ToolResult {
    let message = blocked_message(tool, record);
    let mut result = ToolResult::error(message.clone());
    result.content = message;
    result.typed_payload = Some(serde_json::json!({
        RETRY_BLOCKED_CODE: true,
        "stderr_hash": record.stderr_hash,
        "exit_code": record.exit_code,
    }));
    result
}

fn blocked_message(tool: &str, record: &FailureRecord) -> String {
    let exit = record
        .exit_code
        .map(|code| format!(", exit_code={code}"))
        .unwrap_or_default();
    let subject = if tool == TOOL_BASH {
        format!("`{}`", truncate_text(&record.label, COMMAND_LABEL_CHARS))
    } else {
        format!("`{tool}` with the same arguments")
    };
    format!(
        "{RETRY_BLOCKED_CODE}: {subject} already failed this turn (stderr_hash={}{exit}). \
         Do not run it again until a write or edit changes the relevant files. \
         The previous failure is already in context.",
        record.stderr_hash
    )
}

fn is_retry_blocked(result: &ToolResult) -> bool {
    result
        .typed_payload
        .as_ref()
        .and_then(|payload| payload.get(RETRY_BLOCKED_CODE))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn fingerprint(tool: &str, args: &Value) -> String {
    if tool == TOOL_BASH {
        let command = args
            .get("command")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        let cwd = args
            .get("working_directory")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        return format!("bash\n{cwd}\n{command}");
    }
    format!("{tool}\n{}", canonical_args(args))
}

fn failure_label(tool: &str, args: &Value) -> String {
    if tool == TOOL_BASH {
        return args
            .get("command")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
    }
    tool.to_string()
}

fn canonical_args(value: &Value) -> String {
    canonical_value(value).to_string()
}

fn canonical_value(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();
            let mut out = serde_json::Map::new();
            for key in keys {
                if let Some(child) = map.get(key) {
                    out.insert(key.clone(), canonical_value(child));
                }
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(canonical_value).collect()),
        other => other.clone(),
    }
}

fn failure_body(result: &ToolResult) -> Cow<'_, str> {
    if let Some(error) = result.error.as_deref().filter(|text| !text.is_empty()) {
        return Cow::Borrowed(error);
    }
    if result.content.is_empty() {
        Cow::Borrowed("")
    } else {
        Cow::Borrowed(result.content.as_str())
    }
}

fn payload_string(result: &ToolResult, key: &str) -> Option<String> {
    result
        .typed_payload
        .as_ref()
        .and_then(|payload| payload.get(key))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn exit_code_of(result: &ToolResult) -> Option<i64> {
    if let Some(code) = result
        .typed_payload
        .as_ref()
        .and_then(|payload| payload.get("exit_code"))
        .and_then(Value::as_i64)
    {
        return Some(code);
    }
    parse_exit_code(result.error.as_deref().unwrap_or(""))
}

fn parse_exit_code(text: &str) -> Option<i64> {
    let marker = "exit code ";
    let start = text.find(marker)?;
    let rest = &text[start + marker.len()..];
    let digits: String = rest
        .chars()
        .take_while(|ch| ch.is_ascii_digit() || *ch == '-')
        .collect();
    digits.parse().ok()
}

fn cwd_of(args: &Value) -> Option<String> {
    args.get("working_directory")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|cwd| !cwd.is_empty())
        .map(str::to_string)
}

/// Explicit `working_directory`, or the process cwd when the field is omitted.
fn logical_cwd(args: &Value) -> Option<String> {
    if let Some(cwd) = cwd_of(args) {
        let normalized = normalize_path(&cwd);
        return (!normalized.is_empty()).then_some(normalized);
    }
    process_cwd()
}

fn process_cwd() -> Option<String> {
    std::env::current_dir().ok().and_then(|path| {
        let normalized = normalize_path(&path.to_string_lossy());
        (!normalized.is_empty()).then_some(normalized)
    })
}

fn cwd_same(recorded: Option<&str>, success: Option<&str>) -> bool {
    match (recorded, success) {
        (Some(left), Some(right)) => left == right,
        (None, None) => true,
        _ => false,
    }
}

fn relevant_paths(args: &Value, body: &str) -> Vec<String> {
    let mut paths = Vec::new();
    if let Some(command) = args.get("command").and_then(Value::as_str) {
        paths.extend(extract_paths(command));
    }
    if let Some(path) = string_field(args, "path") {
        paths.push(normalize_path(&path));
    }
    paths.extend(extract_paths(body));
    paths
}

fn extract_paths(text: &str) -> Vec<String> {
    let mut paths = Vec::new();
    for token in text.split(|ch: char| {
        ch.is_whitespace()
            || matches!(
                ch,
                '"' | '\'' | ',' | '(' | ')' | ':' | ';' | '[' | ']' | '{' | '}'
            )
    }) {
        let token = token.trim_matches(|ch| matches!(ch, '.' | '`' | ','));
        if token.is_empty() || !token_is_path(token) {
            continue;
        }
        let normalized = normalize_path(token);
        if !normalized.is_empty() && !normalized.starts_with("http") {
            paths.push(normalized);
        }
    }
    paths
}

fn token_is_path(token: &str) -> bool {
    token.contains('/') || token.contains('\\') || bare_filename(token)
}

fn bare_filename(token: &str) -> bool {
    if token.starts_with('-') || token.len() > 180 {
        return false;
    }
    let Some((stem, ext)) = token.rsplit_once('.') else {
        return false;
    };
    if stem.is_empty() || ext.is_empty() || ext.len() > 8 {
        return false;
    }
    let name_ok = token
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | '@' | '+'));
    name_ok && ext.chars().all(|ch| ch.is_ascii_alphabetic()) && stem_has_letter(stem)
}

fn stem_has_letter(stem: &str) -> bool {
    stem.chars().any(|ch| ch.is_ascii_alphabetic())
}

fn mutation_paths(tool: &str, args: &Value) -> Vec<String> {
    match tool {
        TOOL_FILESYSTEM => fs_mutation_path(args).into_iter().collect(),
        WRITE_FILE | EDIT_FILE => string_field(args, "path").into_iter().collect(),
        TOOL_APPLY_PATCH => args
            .get("patch")
            .and_then(Value::as_str)
            .map(patch_file_paths)
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

fn fs_mutation_path(args: &Value) -> Option<String> {
    let op = args.get("op").and_then(Value::as_str)?;
    if !is_fs_write_op(op) {
        return None;
    }
    string_field(args, "path")
}

fn string_field(args: &Value, key: &str) -> Option<String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .map(str::to_string)
}

fn patch_file_paths(patch: &str) -> Vec<String> {
    const MARKERS: [&str; 3] = ["*** Update File:", "*** Add File:", "*** Delete File:"];
    patch
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            let marker = MARKERS.iter().find(|marker| line.starts_with(*marker))?;
            let file = line[marker.len()..].trim();
            if file.is_empty() {
                None
            } else {
                Some(file.to_string())
            }
        })
        .collect()
}

fn normalize_path(path: &str) -> String {
    let mut normalized = path.trim().replace('\\', "/");
    loop {
        let Some(next) = normalized.strip_prefix("./").map(str::to_string) else {
            break;
        };
        normalized = next;
    }
    while normalized.len() > 1 && normalized.ends_with('/') {
        normalized.pop();
    }
    normalized
}

fn paths_overlap(left: &str, right: &str) -> bool {
    let left = normalize_path(left);
    let right = normalize_path(right);
    if left.is_empty() || right.is_empty() {
        return false;
    }
    left == right
        || left.ends_with(&format!("/{right}"))
        || right.ends_with(&format!("/{left}"))
        || left.starts_with(&format!("{right}/"))
        || right.starts_with(&format!("{left}/"))
}

fn stderr_hash(text: &str) -> String {
    short_hash(text.as_bytes())
}

fn short_hash(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(STDERR_HASH_BYTES * 2);
    for byte in digest.iter().take(STDERR_HASH_BYTES) {
        push_hex(&mut out, *byte);
    }
    out
}

fn push_hex(out: &mut String, byte: u8) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    out.push(char::from(HEX[usize::from(byte >> 4)]));
    out.push(char::from(HEX[usize::from(byte & 0x0f)]));
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    let collapsed = collapse_whitespace(text);
    let count = collapsed.chars().count();
    if count <= max_chars {
        return collapsed;
    }
    let end = collapsed
        .char_indices()
        .nth(max_chars)
        .map_or(collapsed.len(), |(idx, _)| idx);
    format!("{}...", &collapsed[..end])
}

fn live_breaker_env() -> Option<String> {
    match injected_breaker_env() {
        Some(value) => value,
        None => std::env::var(BREAKER_ENV).ok(),
    }
}

#[cfg(test)]
fn injected_breaker_env() -> Option<Option<String>> {
    BREAKER_ENV_INJECT
        .try_with(|value| (*value).map(str::to_owned))
        .ok()
}

#[cfg(not(test))]
fn injected_breaker_env() -> Option<Option<String>> {
    None
}

/// `true` unless `value` is an explicit off switch.
#[must_use]
pub fn env_enables_breaker(value: &str) -> bool {
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "0" | "false" | "off" | "no" | "disabled"
    )
}

fn is_transient(tool: &str, result: &ToolResult) -> bool {
    let body = failure_body(result);
    let lower = body.to_ascii_lowercase();
    if lower.contains("timed out")
        || lower.contains("timeout")
        || lower.contains("etimedout")
        || lower.contains("econnreset")
        || lower.contains("connection reset")
        || lower.contains("connection refused")
        || lower.contains("temporarily unavailable")
    {
        return true;
    }
    matches!(tool, TOOL_WEB_FETCH | TOOL_HTTP | TOOL_WEB_SEARCH) && http_status_is_transient(&lower)
}

fn http_status_is_transient(text: &str) -> bool {
    ["408", "429", "500", "502", "503", "504"]
        .iter()
        .any(|code| contains_status_code(text, code))
}

fn contains_status_code(text: &str, code: &str) -> bool {
    text.match_indices(code).any(|(idx, _)| {
        let before_ok = idx == 0 || !text.as_bytes()[idx - 1].is_ascii_digit();
        let after = idx + code.len();
        let after_ok = after >= text.len() || !text.as_bytes()[after].is_ascii_digit();
        before_ok && after_ok
    })
}

fn bash_command_mutates(command: &str) -> bool {
    if command_has_file_redirect(command) || python_open_writes(command) {
        return true;
    }
    let lower = command.to_ascii_lowercase();
    if lower.contains("sed") && (lower.contains("-i") || lower.contains("--in-place")) {
        return true;
    }
    ["tee ", "patch ", " rm ", " mv ", " cp ", "mkdir ", "touch "]
        .iter()
        .any(|marker| lower.contains(marker))
        || lower.starts_with("rm ")
        || lower.starts_with("mv ")
        || lower.starts_with("cp ")
}

fn python_open_writes(command: &str) -> bool {
    let lower = command.to_ascii_lowercase();
    lower.match_indices("open(").any(|(idx, _)| {
        let rest = &lower[idx..];
        let end = rest.find(')').unwrap_or(rest.len());
        open_call_writes(&rest[..end])
    })
}

fn open_call_writes(call: &str) -> bool {
    call.split(',').skip(1).any(|arg| {
        let mode = arg.trim().trim_matches(|ch: char| ch == '\'' || ch == '"');
        is_write_mode(mode)
    })
}

fn is_write_mode(mode: &str) -> bool {
    matches!(
        mode,
        "w" | "a" | "x" | "wb" | "ab" | "xb" | "wt" | "at" | "w+" | "a+" | "r+" | "rb+" | "wb+"
    )
}

fn command_has_file_redirect(command: &str) -> bool {
    let bytes = command.as_bytes();
    for (idx, byte) in bytes.iter().enumerate() {
        if *byte != b'>' {
            continue;
        }
        let prev = idx.checked_sub(1).map(|pos| bytes[pos]);
        let next = bytes.get(idx + 1).copied();
        if matches!(prev, Some(b'=' | b'-')) {
            continue;
        }
        if next == Some(b'&') || prev == Some(b'&') {
            continue;
        }
        return true;
    }
    false
}

fn stamp_paths(cwd: Option<&str>, paths: &[String]) -> Vec<PathStamp> {
    let mut stamps = Vec::new();
    for path in paths {
        if stamps.len() >= STAMP_LIMIT {
            break;
        }
        let resolved = resolve_against_cwd(cwd, path);
        if resolved.is_empty()
            || stamps
                .iter()
                .any(|stamp: &PathStamp| stamp.path == resolved)
        {
            continue;
        }
        stamps.push(stamp_one(&resolved));
    }
    stamps
}

fn resolve_against_cwd(cwd: Option<&str>, path: &str) -> String {
    let normalized = normalize_path(path);
    if normalized.is_empty() {
        return String::new();
    }
    if normalized.starts_with('/') {
        return normalized;
    }
    match cwd {
        Some(cwd) if !cwd.is_empty() => format!("{cwd}/{normalized}"),
        _ => normalized,
    }
}

fn stamp_one(path: &str) -> PathStamp {
    let canonical = std::path::Path::new(path).canonicalize().ok();
    let meta = canonical
        .as_deref()
        .and_then(|file| crate::agent::tools::common::path::resolved_metadata(file).ok());
    let mtime_nanos = meta.as_ref().and_then(|meta| {
        meta.modified()
            .ok()
            .and_then(|modified| system_time_nanos(modified))
    });
    let content_hash = meta.as_ref().and_then(|meta| {
        if !meta.is_file() || meta.len() > STAMP_HASH_MAX_BYTES {
            return None;
        }
        let file = canonical.as_deref()?;
        crate::agent::tools::common::path::read_resolved_bytes(file)
            .ok()
            .map(|bytes| short_hash(&bytes))
    });
    PathStamp {
        path: path.to_string(),
        mtime_nanos,
        content_hash,
    }
}

fn system_time_nanos(time: SystemTime) -> Option<u128> {
    time.duration_since(SystemTime::UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_nanos())
}

fn stamps_changed(stamps: &[PathStamp]) -> bool {
    stamps.iter().any(|stamp| {
        let current = stamp_one(&stamp.path);
        current.mtime_nanos != stamp.mtime_nanos || current.content_hash != stamp.content_hash
    })
}

fn redact_for_log(text: &str) -> String {
    outcome_redactor().map_or_else(
        || "[redaction-unavailable]".to_string(),
        |redactor| redactor.redact_text(text).text,
    )
}

fn outcome_redactor() -> Option<&'static TranscriptRedactor> {
    static REDACTOR: OnceLock<Option<TranscriptRedactor>> = OnceLock::new();
    REDACTOR
        .get_or_init(|| TranscriptRedactor::with_default_patterns().ok())
        .as_ref()
}

fn collapse_whitespace(text: &str) -> String {
    let mut out = String::with_capacity(text.len().min(STDERR_LOG_CHARS));
    let mut pending_space = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            pending_space = !out.is_empty();
            continue;
        }
        if pending_space {
            out.push(' ');
            pending_space = false;
        }
        out.push(ch);
    }
    out
}

#[cfg(test)]
#[path = "failure_breaker_unit_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "failure_breaker_tests.rs"]
mod failure_breaker_tests;

#[cfg(test)]
#[path = "failure_breaker_loop_tests.rs"]
mod failure_breaker_loop_tests;

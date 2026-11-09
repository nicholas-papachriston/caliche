//! One pending approval id, a human answer, and a binding re-check.
//!
//! Doors render [`ActionCard`]. This module does not prompt, draw a UI, or
//! expose an HTTP route. The first human answer wins. An unattended
//! principal, a plugin, a peer, or a subagent answer does not settle the
//! approval. Expiry denies. A stale binding cancels the call and is not
//! reused.

use crate::ids::{ActionBinding, PendingId};
use crate::principal::Principal;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Default time a pending approval stays open.
pub const APPROVAL_TTL: Duration = Duration::from_secs(120);

/// Facts the broker stores for one ask.
pub struct ApprovalRequest<'a> {
    /// Tool name (`bash`, `fs`, `http`, `server__tool`).
    pub tool: &'a str,
    /// Raw tool arguments. The card shows these. The binding hashes them.
    pub args: &'a Value,
    /// Working directory included in the binding.
    pub cwd: &'a str,
    /// Shell or code dialect, when the tool has one.
    pub dialect: Option<&'a str>,
    /// Policy snapshot version included in the binding.
    pub policy_version: u64,
    /// Why the hook asked. This text is shown on the card.
    pub reason: &'a str,
}

/// Text a door prints or sends. The summary always names the arguments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActionCard {
    tool: String,
    reason: String,
    summary: String,
}

impl ActionCard {
    /// Build a card from a tool call.
    ///
    /// The summary never uses the placeholder `<no command>`. A missing shell
    /// command is `(empty command)`.
    #[must_use]
    pub fn from_call(tool: &str, args: &Value, reason: &str) -> Self {
        Self {
            tool: tool.to_owned(),
            reason: reason.to_owned(),
            summary: summarize(tool, args),
        }
    }

    /// Tool name.
    #[must_use]
    pub fn tool(&self) -> &str {
        &self.tool
    }

    /// Argument summary for this class.
    #[must_use]
    pub fn summary(&self) -> &str {
        &self.summary
    }

    /// Lines a CLI door can print.
    #[must_use]
    pub fn render(&self) -> String {
        format!(
            "[PERMISSION] {}: {}\n  {}",
            one_line(&self.tool),
            one_line(&self.reason),
            one_line(&self.summary)
        )
    }
}

/// Human answer. `Once` allows this binding. `Deny` blocks it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApprovalAnswer {
    /// Run this binding one time.
    Once,
    /// Do not run this call.
    Deny,
}

/// Result of the first valid human answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Settlement {
    /// The binding may run after [`ApprovalBroker::release`] says it is fresh.
    Granted {
        /// Pending id.
        id: PendingId,
        /// Binding captured when the approval opened.
        binding: ActionBinding,
    },
    /// The human denied the call.
    Denied {
        /// Pending id.
        id: PendingId,
    },
}

/// Why an answer or a release did not produce a grant.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum BrokerError {
    /// The principal is not [`Principal::Human`].
    #[error("only a human principal may answer an approval")]
    NotHuman,
    /// No slot exists for this id.
    #[error("pending approval '{0}' was not found")]
    Unknown(String),
    /// `now` is at or after the expiry instant.
    #[error("pending approval '{0}' expired")]
    Expired(String),
    /// A human already answered this id.
    #[error("pending approval '{0}' already has an answer")]
    AlreadyAnswered(String),
    /// The human answer was deny.
    #[error("pending approval '{0}' was denied")]
    Denied(String),
    /// Release ran before a human answered.
    #[error("pending approval '{0}' has no answer yet")]
    NotAnswered(String),
    /// Arguments or policy version changed. The grant was dropped.
    #[error("approval binding is stale and will not be reused")]
    StaleBinding,
}

#[derive(Clone, Debug)]
enum Slot {
    Pending {
        expires_at: Instant,
        binding: ActionBinding,
    },
    Granted {
        binding: ActionBinding,
    },
    Denied,
}

/// In-memory approvals for one process.
///
/// Each `open` mints one `apr_` id. Doors share that id. The first human
/// answer settles it.
#[derive(Debug)]
pub struct ApprovalBroker {
    ttl: Duration,
    slots: Mutex<HashMap<String, Slot>>,
}

impl ApprovalBroker {
    /// Broker with [`APPROVAL_TTL`].
    #[must_use]
    pub fn new() -> Self {
        Self::with_ttl(APPROVAL_TTL)
    }

    /// Broker with a caller-chosen lifetime. Tests use a short value.
    #[must_use]
    pub fn with_ttl(ttl: Duration) -> Self {
        Self {
            ttl,
            slots: Mutex::new(HashMap::new()),
        }
    }

    /// Register one pending approval and return its card.
    #[must_use]
    pub fn open(&self, request: ApprovalRequest<'_>, now: Instant) -> OpenedApproval {
        let id = PendingId::new();
        let binding = ActionBinding::compute(
            request.tool,
            request.args,
            request.cwd,
            request.dialect,
            request.policy_version,
        );
        let card = ActionCard::from_call(request.tool, request.args, request.reason);
        let expires_at = now + self.ttl;
        let mut slots = lock_slots(&self.slots);
        slots.insert(
            id.as_str().to_owned(),
            Slot::Pending {
                expires_at,
                binding,
            },
        );
        OpenedApproval {
            id,
            card,
            binding,
            expires_at,
        }
    }

    /// Apply one answer.
    ///
    /// A non-human principal leaves the slot pending.
    ///
    /// # Errors
    ///
    /// See [`BrokerError`].
    pub fn answer(
        &self,
        id: &PendingId,
        principal: &Principal,
        answer: ApprovalAnswer,
        now: Instant,
    ) -> Result<Settlement, BrokerError> {
        if !principal.is_human() {
            return Err(BrokerError::NotHuman);
        }
        let mut slots = lock_slots(&self.slots);
        let Some(slot) = slots.get(id.as_str()) else {
            return Err(BrokerError::Unknown(id.as_str().to_owned()));
        };
        match slot {
            Slot::Pending {
                expires_at,
                binding,
            } => {
                if now >= *expires_at {
                    slots.remove(id.as_str());
                    return Err(BrokerError::Expired(id.as_str().to_owned()));
                }
                let binding = *binding;
                let settled = settle_pending(answer, id, binding);
                let next = match &settled {
                    Settlement::Granted { binding, .. } => Slot::Granted { binding: *binding },
                    Settlement::Denied { .. } => Slot::Denied,
                };
                slots.insert(id.as_str().to_owned(), next);
                Ok(settled)
            }
            Slot::Granted { .. } | Slot::Denied => {
                Err(BrokerError::AlreadyAnswered(id.as_str().to_owned()))
            }
        }
    }

    /// Recompute check before the tool runs.
    ///
    /// A fresh grant is consumed. A stale grant is dropped and is not reused.
    ///
    /// # Errors
    ///
    /// See [`BrokerError`].
    pub fn release(&self, id: &PendingId, actual: ActionBinding) -> Result<(), BrokerError> {
        let mut slots = lock_slots(&self.slots);
        let Some(slot) = slots.remove(id.as_str()) else {
            return Err(BrokerError::Unknown(id.as_str().to_owned()));
        };
        match slot {
            Slot::Granted { binding } => release_grant(binding, actual),
            Slot::Denied => Err(BrokerError::Denied(id.as_str().to_owned())),
            Slot::Pending {
                expires_at,
                binding,
            } => {
                slots.insert(
                    id.as_str().to_owned(),
                    Slot::Pending {
                        expires_at,
                        binding,
                    },
                );
                Err(BrokerError::NotAnswered(id.as_str().to_owned()))
            }
        }
    }

    /// Drop a slot after the door fails before a human answers.
    pub fn dismiss(&self, id: &PendingId) {
        lock_slots(&self.slots).remove(id.as_str());
    }
}

impl Default for ApprovalBroker {
    fn default() -> Self {
        Self::new()
    }
}

/// What [`ApprovalBroker::open`] returns to a door.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenedApproval {
    /// `apr_` id shared by every door.
    pub id: PendingId,
    /// Full argument card.
    pub card: ActionCard,
    /// Binding captured at open.
    pub binding: ActionBinding,
    /// Instant at which [`ApprovalBroker::answer`] expires the slot.
    pub expires_at: Instant,
}

fn settle_pending(answer: ApprovalAnswer, id: &PendingId, binding: ActionBinding) -> Settlement {
    match answer {
        ApprovalAnswer::Once => Settlement::Granted {
            id: id.clone(),
            binding,
        },
        ApprovalAnswer::Deny => Settlement::Denied { id: id.clone() },
    }
}

fn release_grant(expected: ActionBinding, actual: ActionBinding) -> Result<(), BrokerError> {
    match expected.recheck(actual) {
        crate::ids::BindingRecheck::Fresh => Ok(()),
        crate::ids::BindingRecheck::Stale { .. } => Err(BrokerError::StaleBinding),
    }
}

fn lock_slots(
    slots: &Mutex<HashMap<String, Slot>>,
) -> std::sync::MutexGuard<'_, HashMap<String, Slot>> {
    slots.lock().unwrap_or_else(|err| err.into_inner())
}

fn summarize(tool: &str, args: &Value) -> String {
    match card_kind(tool) {
        CardKind::Shell => shell_summary(args),
        CardKind::Fs => fs_summary(args),
        CardKind::Http => http_summary(args),
        CardKind::Python => python_summary(args),
        CardKind::Mcp => mcp_summary(tool, args),
        CardKind::Other => other_summary(args),
    }
}

enum CardKind {
    Shell,
    Fs,
    Http,
    Python,
    Mcp,
    Other,
}

fn card_kind(tool: &str) -> CardKind {
    match tool {
        "bash" | "shell" => CardKind::Shell,
        "fs" | "apply_patch" | "write_file" | "edit_file" => CardKind::Fs,
        "http" | "web_fetch" => CardKind::Http,
        "execute_python" => CardKind::Python,
        _ if tool.contains("__") => CardKind::Mcp,
        _ => CardKind::Other,
    }
}

fn shell_summary(args: &Value) -> String {
    let command = string_field(args, &["command"]).unwrap_or("(empty command)");
    let mut line = format!("Shell command: {command}");
    if let Some(dir) = string_field(args, &["working_directory"]) {
        line.push_str(" cwd ");
        line.push_str(dir);
    }
    line
}

fn fs_summary(args: &Value) -> String {
    let op = string_field(args, &["op", "operation"]).unwrap_or("write");
    let path = string_field(args, &["path", "file_path", "target"]).unwrap_or("(no path)");
    format!("Fs {op}: {path}")
}

fn http_summary(args: &Value) -> String {
    let method = string_field(args, &["method"]).unwrap_or("GET");
    let mut line = format!(
        "Http {method} {} body {} bytes",
        http_target(args),
        body_len(args)
    );
    if let Some(query) = args.get("query_params").filter(|value| !value.is_null()) {
        line.push_str(" query ");
        line.push_str(&truncate_chars(&query.to_string(), 200));
    }
    if string_field(args, &["credential_type"]).is_some() {
        line.push_str(" credential attached");
    }
    line
}

fn http_target(args: &Value) -> String {
    if let Some(url) = string_field(args, &["url"]) {
        return url.to_owned();
    }
    if let Some(urls) = args.get("urls").filter(|value| !value.is_null()) {
        return truncate_chars(&urls.to_string(), 300);
    }
    let Some(endpoint) = string_field(args, &["endpoint"]) else {
        return "(no url)".to_owned();
    };
    if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
        return endpoint.to_owned();
    }
    let base = string_field(args, &["base_url"]).unwrap_or("(no base_url)");
    format!("{base}{endpoint}")
}

fn python_summary(args: &Value) -> String {
    let code = string_field(args, &["code"]).unwrap_or("");
    let preview = first_line(code);
    let lines = if code.is_empty() {
        0
    } else {
        code.lines().count()
    };
    let hash = blake3::hash(code.as_bytes());
    let hex = hash.to_hex();
    format!("Python: {preview} lines {lines} hash {}", &hex[..8])
}

fn mcp_summary(tool: &str, args: &Value) -> String {
    let (server, name) = tool.split_once("__").unwrap_or(("mcp", tool));
    let rendered = truncate_chars(&args.to_string(), 400);
    format!("MCP {server} {name}: {rendered}")
}

fn other_summary(args: &Value) -> String {
    format!("Args: {}", truncate_chars(&args.to_string(), 400))
}

fn string_field<'a>(args: &'a Value, keys: &[&str]) -> Option<&'a str> {
    let object = args.as_object()?;
    for key in keys {
        if let Some(text) = object.get(*key).and_then(Value::as_str) {
            if !text.is_empty() {
                return Some(text);
            }
        }
    }
    None
}

fn body_len(args: &Value) -> usize {
    match args.get("body") {
        Some(Value::String(text)) => text.len(),
        Some(Value::Null) | None => 0,
        Some(other) => other.to_string().len(),
    }
}

fn first_line(code: &str) -> String {
    let line = code.lines().next().unwrap_or("(empty code)");
    truncate_chars(line, 120)
}

fn one_line(text: &str) -> String {
    text.replace(['\n', '\r'], " ")
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    let mut end = text.len();
    for (count, (index, _)) in text.char_indices().enumerate() {
        if count == max_chars {
            end = index;
            break;
        }
    }
    if end == text.len() {
        return text.to_owned();
    }
    let mut out = text[..end].to_owned();
    out.push_str("...");
    out
}

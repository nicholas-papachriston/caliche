//! Session-scoped idempotent inbox for runtime inputs.
//!
//! Unreal mapping:
//! - `Input` / `InputKind` → [`InboxInput`] / [`InputKind`]
//! - control modes → [`ControlMessage`]
//! - volatile session dedupe → [`SessionInbox`]
//!
//! Caliche-native extras: inbox state is persisted on
//! [`crate::harness::suspend::SuspendedSession`] so redelivery after
//! suspend/resume does not double-apply.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Caller-supplied (or generated) stable input id.
pub type InputId = String;

/// Classification of an inbox item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputKind {
    /// User / gateway / API prompt.
    External,
    /// Hard stop, when-idle, heartbeat, or settings.
    Control,
    /// Recovery input after a crash or reconnect.
    Crash,
}

/// Control-plane modes that share the inbox id space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlMode {
    /// Stop immediately; further turns are rejected.
    Hard,
    /// Stop after the in-flight turn (or immediately when idle).
    WhenIdle,
    /// Liveness ping; must include a reason.
    Heartbeat,
    /// Patch session settings (currently reasoning effort).
    Settings,
}

/// Settings payload accepted by [`ControlMode::Settings`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InboxSettings {
    /// Optional reasoning-effort hint (`none`, `low`, `medium`, `high`, …).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

/// Decoded control message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlMessage {
    /// Control mode.
    pub mode: ControlMode,
    /// Human-readable reason (required for heartbeat).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub reason: String,
    /// Settings patch when `mode == Settings`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub settings: Option<InboxSettings>,
}

/// One session inbox item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InboxInput {
    /// Stable id; generated if the caller omitted one.
    pub id: InputId,
    /// Kind of input.
    pub kind: InputKind,
    /// External / crash prompt text, or unused for control.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub payload: String,
    /// Control body when `kind == Control`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control: Option<ControlMessage>,
}

impl InboxInput {
    /// External prompt. Generates an id when `id` is `None` or empty.
    #[must_use]
    pub fn external(id: Option<impl Into<String>>, payload: impl Into<String>) -> Self {
        Self {
            id: normalize_id(id),
            kind: InputKind::External,
            payload: payload.into(),
            control: None,
        }
    }

    /// Crash / reconnect recovery prompt.
    #[must_use]
    pub fn crash(id: Option<impl Into<String>>, payload: impl Into<String>) -> Self {
        Self {
            id: normalize_id(id),
            kind: InputKind::Crash,
            payload: payload.into(),
            control: None,
        }
    }

    /// Control-plane message.
    #[must_use]
    pub fn control(id: Option<impl Into<String>>, control: ControlMessage) -> Self {
        Self {
            id: normalize_id(id),
            kind: InputKind::Control,
            payload: String::new(),
            control: Some(control),
        }
    }

    /// Validate kind-specific invariants (id is never empty after construction).
    pub fn validate(&self) -> Result<()> {
        if self.id.is_empty() {
            bail!("inbox input id is empty");
        }
        match self.kind {
            InputKind::External | InputKind::Crash => Ok(()),
            InputKind::Control => {
                let Some(control) = self.control.as_ref() else {
                    bail!("control input {} is missing a control message", self.id);
                };
                match control.mode {
                    ControlMode::Heartbeat if control.reason.is_empty() => {
                        bail!("heartbeat input {} is missing a reason", self.id);
                    }
                    ControlMode::Settings if control.settings.is_none() => {
                        bail!("settings input {} is missing parameters", self.id);
                    }
                    _ => Ok(()),
                }
            }
        }
    }
}

fn normalize_id(id: Option<impl Into<String>>) -> String {
    match id {
        Some(value) => {
            let owned = value.into();
            if owned.is_empty() {
                uuid::Uuid::now_v7().to_string()
            } else {
                owned
            }
        }
        None => uuid::Uuid::now_v7().to_string(),
    }
}

/// Whether submit accepted new work or saw a redelivery.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InboxAck {
    /// Stable input id (caller-supplied or generated).
    pub input_id: InputId,
    /// True when this id was already accepted.
    pub duplicate: bool,
    /// Kind of the stored input.
    pub kind: InputKind,
    /// True after the input has been applied (turn or control).
    pub applied: bool,
}

/// Persisted record for one input id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InboxRecord {
    /// Original input.
    pub input: InboxInput,
    /// Whether apply() has run.
    pub applied: bool,
    /// Short result summary (turn text or control outcome).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

/// Session-scoped dedupe store.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionInbox {
    records: HashMap<InputId, InboxRecord>,
}

impl SessionInbox {
    /// Empty inbox.
    #[must_use]
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
        }
    }

    /// Accept `input` or report a duplicate. Does not apply side effects.
    pub fn submit(&mut self, input: InboxInput) -> Result<InboxAck> {
        input.validate()?;
        if let Some(existing) = self.records.get(&input.id) {
            return Ok(InboxAck {
                input_id: input.id,
                duplicate: true,
                kind: existing.input.kind,
                applied: existing.applied,
            });
        }
        let ack = InboxAck {
            input_id: input.id.clone(),
            duplicate: false,
            kind: input.kind,
            applied: false,
        };
        self.records.insert(
            input.id.clone(),
            InboxRecord {
                input,
                applied: false,
                summary: None,
            },
        );
        Ok(ack)
    }

    /// Mark `input_id` applied and store an optional summary.
    pub fn mark_applied(&mut self, input_id: &str, summary: Option<String>) -> Result<()> {
        let record = self
            .records
            .get_mut(input_id)
            .ok_or_else(|| anyhow::anyhow!("unknown inbox input '{input_id}'"))?;
        record.applied = true;
        record.summary = summary;
        Ok(())
    }

    /// Look up a stored input.
    #[must_use]
    pub fn get(&self, input_id: &str) -> Option<&InboxRecord> {
        self.records.get(input_id)
    }

    /// True when `input_id` was already submitted.
    #[must_use]
    pub fn contains(&self, input_id: &str) -> bool {
        self.records.contains_key(input_id)
    }

    /// Number of distinct input ids.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// True when no inputs have been submitted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

/// Result of submitting and applying one inbox item.
#[derive(Debug, Clone)]
pub struct InboxTurnOutcome {
    /// Submit acknowledgment (includes generated id and duplicate flag).
    pub ack: InboxAck,
    /// Turn result when the input was an external or crash prompt.
    pub turn: Option<crate::harness::HarnessTurnResult>,
    /// Control result when the input was a control message.
    pub control: Option<ControlOutcome>,
}

/// Outcome of applying a control message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlOutcome {
    /// Session is hard-stopped.
    HardStopped {
        /// Reason supplied by the caller.
        reason: String,
    },
    /// Stop was scheduled for idle (or applied immediately because idle).
    WhenIdle {
        /// True when the session was already idle.
        applied_immediately: bool,
        /// Reason supplied by the caller.
        reason: String,
    },
    /// Heartbeat recorded.
    Heartbeat {
        /// Heartbeat reason.
        reason: String,
    },
    /// Settings patched.
    SettingsApplied {
        /// New settings snapshot.
        settings: InboxSettings,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redelivery_of_same_id_is_duplicate() {
        let mut inbox = SessionInbox::new();
        let first = inbox
            .submit(InboxInput::external(Some("in-1"), "hello"))
            .expect("submit");
        assert!(!first.duplicate);
        let second = inbox
            .submit(InboxInput::external(Some("in-1"), "hello again"))
            .expect("redeliver");
        assert!(second.duplicate);
        assert_eq!(inbox.len(), 1);
    }

    #[test]
    fn generated_ids_are_unique() {
        let mut inbox = SessionInbox::new();
        let a = inbox
            .submit(InboxInput::external(None::<String>, "a"))
            .expect("a");
        let b = inbox
            .submit(InboxInput::external(None::<String>, "b"))
            .expect("b");
        assert_ne!(a.input_id, b.input_id);
        assert_eq!(inbox.len(), 2);
    }

    #[test]
    fn heartbeat_requires_reason() {
        let err = InboxInput::control(
            Some("hb"),
            ControlMessage {
                mode: ControlMode::Heartbeat,
                reason: String::new(),
                settings: None,
            },
        )
        .validate()
        .expect_err("empty reason");
        assert!(err.to_string().contains("reason"));
    }

    #[test]
    fn inbox_json_round_trip() {
        let mut inbox = SessionInbox::new();
        inbox
            .submit(InboxInput::external(Some("x"), "hi"))
            .expect("submit");
        inbox.mark_applied("x", Some("ack".into())).expect("apply");
        let json = serde_json::to_string(&inbox).expect("ser");
        let loaded: SessionInbox = serde_json::from_str(&json).expect("de");
        assert!(loaded.contains("x"));
        assert!(loaded.get("x").is_some_and(|r| r.applied));
    }
}

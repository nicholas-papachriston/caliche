use crate::data::persistence::session_registry::{SessionQuery, SessionStatus, TrackedSession};
use anyhow::{Result, anyhow};
use uuid::Uuid;

/// Parse a persisted status string into [`SessionStatus`] strictly. Returns
/// an error on unknown values so that corrupted rows surface loudly instead of
/// silently defaulting to `Active`.
pub fn status_from_db(value: &str) -> Result<SessionStatus> {
    match value {
        "active" => Ok(SessionStatus::Active),
        "archived" => Ok(SessionStatus::Archived),
        "expired" => Ok(SessionStatus::Expired),
        other => Err(anyhow!(
            "Unknown session status '{other}' in database row; refusing to coerce to active"
        )),
    }
}

pub const fn status_to_db(status: SessionStatus) -> &'static str {
    status.as_str()
}

pub fn new_tracked_session(id: Uuid, agent_id: &str, user_id: Option<&str>) -> TrackedSession {
    TrackedSession::new(id, agent_id, user_id.map(str::to_string))
}

pub enum SessionFilter<'a> {
    AgentId(&'a str),
    UserId(&'a str),
    Status(SessionStatus),
}

pub fn collect_filters(query: &SessionQuery) -> Vec<SessionFilter<'_>> {
    let mut out = Vec::new();
    if let Some(agent_id) = query.agent_id.as_deref() {
        out.push(SessionFilter::AgentId(agent_id));
    }
    if let Some(user_id) = query.user_id.as_deref() {
        out.push(SessionFilter::UserId(user_id));
    }
    if let Some(status) = query.status {
        out.push(SessionFilter::Status(status));
    }
    out
}

//! History tab — server sessions from `GET /api/agent/sessions`.

use chrono::{DateTime, NaiveDate, Utc};
use dioxus::prelude::*;

use crate::api::LiveSessionInfo;
use crate::components::chrome::{
    EMPTY_SENTENCE, EmptyHint, RETRY_ACTION, START_CHAT_ACTION, load_hint_body,
    load_kind_from_error, server_message_from_error, session_display_title,
};

/// A history row the operator chose to open in chat.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenedSession {
    pub session_id: String,
    pub title: Option<String>,
}

struct HistoryGroup {
    label: String,
    sessions: Vec<LiveSessionInfo>,
}

/// Props for the HistoryView component.
#[derive(Props, Clone, PartialEq)]
pub struct HistoryViewProps {
    /// Live sessions from `GET /api/agent/sessions`, newest activity first.
    pub sessions: Signal<Vec<LiveSessionInfo>>,
    /// Last list error, when the request failed.
    pub error: Signal<Option<String>>,
    /// True while the first list request is in flight.
    pub loading: Signal<bool>,
    /// API address shown on failure.
    pub address: String,
    /// Open this server session in chat.
    pub on_open: EventHandler<OpenedSession>,
    /// Retry the list request.
    pub on_retry: EventHandler<()>,
    /// Leave history and start a chat when the list is empty.
    pub on_start_chat: EventHandler<()>,
}

/// True when `query` is blank or matches the title or session id.
#[must_use]
pub fn session_matches_query(title: Option<&str>, session_id: &str, query: &str) -> bool {
    let query = query.trim();
    if query.is_empty() {
        return true;
    }
    let query = query.to_ascii_lowercase();
    session_id.to_ascii_lowercase().contains(&query)
        || title.is_some_and(|title| title.to_ascii_lowercase().contains(&query))
}

/// Bucket label for a session's `last_activity` relative to `today`.
#[must_use]
pub fn history_group_label(last_activity: &str, today: NaiveDate) -> &'static str {
    let Some(saved) = activity_date(last_activity) else {
        return "older";
    };
    let days = (today - saved).num_days();
    if days <= 0 {
        "today"
    } else if days == 1 {
        "yesterday"
    } else if days < 7 {
        "last 7 days"
    } else if days < 30 {
        "last 30 days"
    } else {
        "older"
    }
}

fn activity_date(last_activity: &str) -> Option<NaiveDate> {
    DateTime::parse_from_rfc3339(last_activity)
        .ok()
        .map(|value| value.with_timezone(&Utc).date_naive())
}

fn activity_key(session: &LiveSessionInfo) -> i64 {
    DateTime::parse_from_rfc3339(&session.last_activity)
        .map(|value| value.timestamp())
        .unwrap_or(i64::MIN)
}

fn sorted_sessions(mut sessions: Vec<LiveSessionInfo>) -> Vec<LiveSessionInfo> {
    sessions.sort_by(|left, right| activity_key(right).cmp(&activity_key(left)));
    sessions
}

fn group_sessions(sessions: Vec<LiveSessionInfo>, today: NaiveDate) -> Vec<HistoryGroup> {
    let mut groups: Vec<HistoryGroup> = Vec::new();
    for session in sessions {
        let label = history_group_label(&session.last_activity, today).to_string();
        if let Some(group) = groups.iter_mut().find(|group| group.label == label) {
            group.sessions.push(session);
        } else {
            groups.push(HistoryGroup {
                label,
                sessions: vec![session],
            });
        }
    }
    groups
}

fn relative_activity(last_activity: &str) -> String {
    let Some(saved) = DateTime::parse_from_rfc3339(last_activity)
        .ok()
        .map(|value| value.with_timezone(&Utc))
    else {
        return "unknown time".to_string();
    };
    let secs = (Utc::now() - saved).num_seconds().max(0);
    if secs < 60 {
        return "just now".to_string();
    }
    if secs < 3_600 {
        return format!("{}m ago", secs / 60);
    }
    if secs < 86_400 {
        return format!("{}h ago", secs / 3_600);
    }
    if secs < 7 * 86_400 {
        return format!("{}d ago", secs / 86_400);
    }
    saved.format("%b %d").to_string()
}

/// Server session list. Opening a row resumes that session in chat.
#[component]
pub fn HistoryView(props: HistoryViewProps) -> Element {
    let mut search_query = use_signal(String::new);
    let address = props.address.clone();
    let sessions = sorted_sessions(props.sessions.read().clone());
    let error = props.error.read().clone();
    let loading = *props.loading.read();
    let query = search_query.read().trim().to_string();
    let today = Utc::now().date_naive();
    let filtered = sessions
        .iter()
        .filter(|session| {
            session_matches_query(session.title.as_deref(), &session.session_id, &query)
        })
        .cloned()
        .collect::<Vec<_>>();
    let groups = group_sessions(filtered, today);
    let visible_count = groups
        .iter()
        .map(|group| group.sessions.len())
        .sum::<usize>();
    let failure_body = error.as_deref().map(|err| {
        load_hint_body(
            load_kind_from_error(err),
            &address,
            server_message_from_error(err).as_deref(),
        )
    });

    rsx! {
        div { class: "history-view",
            style: "flex: 1; min-height: 0; overflow-y: auto; padding: 1ch; display: flex; flex-direction: column; gap: 1ch;",

            div { style: "display: flex; align-items: center; gap: 1ch; padding: 0 0 0.5em 0; border-bottom: 1px solid var(--border-color);",
                input {
                    r#type: "search",
                    value: "{query}",
                    placeholder: "search sessions",
                    aria_label: "Search server sessions",
                    style: "flex: 1; min-width: 0; background: var(--input-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.45em 1ch;",
                    oninput: move |evt| search_query.set(evt.value()),
                }
                span { style: "color: var(--muted-color); font-size: var(--font-size-xs); white-space: nowrap;",
                    if query.is_empty() {
                        "{sessions.len()} sessions"
                    } else {
                        "{visible_count} / {sessions.len()}"
                    }
                }
            }

            if let Some(body) = failure_body.clone() {
                EmptyHint {
                    body: body,
                    action: RETRY_ACTION.to_string(),
                    on_action: move |_| props.on_retry.call(()),
                }
            }

            if loading && sessions.is_empty() && failure_body.is_none() {
                div { style: "color: var(--muted-color); font-size: var(--font-size-xs); padding: 2ch;",
                    "loading..."
                }
            } else if failure_body.is_none() && sessions.is_empty() && !loading {
                EmptyHint {
                    body: EMPTY_SENTENCE.to_string(),
                    action: START_CHAT_ACTION.to_string(),
                    on_action: move |_| props.on_start_chat.call(()),
                }
            } else if failure_body.is_none() && !sessions.is_empty() && groups.is_empty() {
                div { style: "color: var(--muted-color); font-size: var(--font-size-xs); padding: 2ch; text-align: center;",
                    "no matching sessions"
                }
            } else if !groups.is_empty() {
                div { style: "display: flex; flex-direction: column;",
                    for group in groups.iter() {
                        div { key: "{group.label}",
                            div { style: "color: var(--muted-color); font-size: var(--font-size-xs); padding: 0.75em 1ch 0.25em; text-transform: uppercase; letter-spacing: 0.08em;",
                                "{group.label}"
                            }
                            for session in group.sessions.iter() {
                                {
                                    let session_id = session.session_id.clone();
                                    let title = session_display_title(session.title.as_deref());
                                    let opened = OpenedSession {
                                        session_id: session.session_id.clone(),
                                        title: session.title.clone(),
                                    };
                                    let turns = session.turn_count;
                                    let turns_label = if turns == 1 { "turn" } else { "turns" };
                                    let when = relative_activity(&session.last_activity);
                                    let on_open = props.on_open.clone();
                                    rsx! {
                                        button {
                                            key: "{session_id}",
                                            class: "history-row-btn",
                                            style: "width: 100%; text-align: left; background: transparent; border: none; border-bottom: 1px solid var(--border-color); padding: 0.75em 1ch; cursor: pointer; display: flex; flex-direction: column; gap: 0.25em; color: var(--text-color);",
                                            onclick: move |_| on_open.call(opened.clone()),
                                            span { style: "overflow: hidden; text-overflow: ellipsis; white-space: nowrap;",
                                                "{title}"
                                            }
                                            span { style: "color: var(--muted-color); font-size: var(--font-size-xs);",
                                                "{turns} {turns_label} · {when} · {session_id}"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_matches_title_or_id_and_ignores_blank() {
        assert!(session_matches_query(Some("Desk"), "abc", ""));
        assert!(session_matches_query(Some("Desk"), "abc", "  "));
        assert!(session_matches_query(Some("Desk"), "abc", "des"));
        assert!(session_matches_query(None, "Session-9", "session-9"));
        assert!(!session_matches_query(Some("Desk"), "abc", "other"));
    }

    #[test]
    fn group_label_uses_last_activity_day() {
        let today = NaiveDate::from_ymd_opt(2026, 10, 3).unwrap_or_default();
        assert_eq!(history_group_label("2026-10-03T15:00:00Z", today), "today");
        assert_eq!(
            history_group_label("2026-10-02T15:00:00Z", today),
            "yesterday"
        );
        assert_eq!(
            history_group_label("2026-09-28T15:00:00Z", today),
            "last 7 days"
        );
        assert_eq!(history_group_label("not-a-date", today), "older");
    }
}

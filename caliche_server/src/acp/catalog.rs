//! `session/list` paging over the live session cache.
//!
//! The cache is the same in-memory map `GET /api/agent/sessions` reads.
//! Cursors are opaque tokens. A missing `nextCursor` means the page is the last.

use crate::acp::driver::AcpListedSession;
use agent_client_protocol_schema::v1::{ListSessionsResponse, SessionInfo};
use std::path::Path;

pub(crate) const SESSION_LIST_PAGE_SIZE: usize = 50;

const CURSOR_PREFIX: &str = "ao1.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SessionListError {
    RelativeCwd,
    InvalidCursor,
}

pub(crate) fn cwd_is_absolute(cwd: &str) -> bool {
    Path::new(cwd).is_absolute()
}

pub(crate) fn same_cwd(left: &str, right: &str) -> bool {
    Path::new(left)
        .components()
        .eq(Path::new(right).components())
}

pub(crate) fn session_list_page(
    mut rows: Vec<AcpListedSession>,
    cwd_filter: Option<&str>,
    cursor: Option<&str>,
    page_size: usize,
) -> Result<ListSessionsResponse, SessionListError> {
    if let Some(cwd) = cwd_filter {
        if !cwd_is_absolute(cwd) {
            return Err(SessionListError::RelativeCwd);
        }
        rows.retain(|row| same_cwd(&row.cwd, cwd));
    }
    rows.sort_by(|left, right| left.session_id.cmp(&right.session_id));
    let start = cursor_start(&rows, cursor)?;
    let page_size = page_size.max(1);
    let end = start.saturating_add(page_size).min(rows.len());
    let next_cursor = if end < rows.len() {
        rows.get(end.saturating_sub(1))
            .map(|row| encode_cursor(&row.session_id))
    } else {
        None
    };
    let sessions = rows
        .into_iter()
        .skip(start)
        .take(end.saturating_sub(start))
        .map(into_entry)
        .collect();
    Ok(ListSessionsResponse::new(sessions).next_cursor(next_cursor))
}

fn cursor_start(
    rows: &[AcpListedSession],
    cursor: Option<&str>,
) -> Result<usize, SessionListError> {
    let Some(token) = cursor else {
        return Ok(0);
    };
    let session_id = decode_cursor(token).ok_or(SessionListError::InvalidCursor)?;
    let index = rows
        .iter()
        .position(|row| row.session_id == session_id)
        .ok_or(SessionListError::InvalidCursor)?;
    Ok(index.saturating_add(1))
}

fn into_entry(row: AcpListedSession) -> SessionInfo {
    SessionInfo::new(row.session_id, row.cwd)
        .title(row.title)
        .updated_at(
            row.updated_at
                .to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        )
}

fn encode_cursor(session_id: &str) -> String {
    let mut cursor = String::with_capacity(CURSOR_PREFIX.len() + session_id.len());
    cursor.push_str(CURSOR_PREFIX);
    cursor.push_str(session_id);
    cursor
}

fn decode_cursor(cursor: &str) -> Option<&str> {
    let session_id = cursor.strip_prefix(CURSOR_PREFIX)?;
    if session_id.is_empty() {
        None
    } else {
        Some(session_id)
    }
}

#[cfg(test)]
mod tests {
    use super::{SessionListError, same_cwd, session_list_page};
    use crate::acp::driver::AcpListedSession;
    use chrono::{TimeZone, Utc};

    fn row(id: &str, cwd: &str) -> AcpListedSession {
        AcpListedSession {
            session_id: id.to_string(),
            cwd: cwd.to_string(),
            title: None,
            updated_at: Utc
                .with_ymd_and_hms(2026, 1, 2, 3, 4, 5)
                .single()
                .expect("timestamp"),
        }
    }

    #[test]
    fn page_filters_cwd_and_advances_the_cursor() {
        let rows = vec![
            row("sess-b", "/tmp/other"),
            row("sess-a", "/tmp/proj"),
            row("sess-c", "/tmp/proj/"),
        ];
        let first = session_list_page(rows.clone(), Some("/tmp/proj"), None, 1).expect("page");
        assert_eq!(first.sessions.len(), 1);
        assert_eq!(first.sessions[0].session_id.0.as_ref(), "sess-a");
        assert_eq!(
            first.sessions[0].updated_at.as_deref(),
            Some("2026-01-02T03:04:05Z")
        );
        assert!(first.sessions[0].title.is_none());
        let cursor = first.next_cursor.expect("next page");

        let second =
            session_list_page(rows, Some("/tmp/proj"), Some(&cursor), 1).expect("second page");
        assert_eq!(second.sessions.len(), 1);
        assert_eq!(second.sessions[0].session_id.0.as_ref(), "sess-c");
        assert!(second.next_cursor.is_none());
    }

    #[test]
    fn relative_cwd_and_bad_cursor_are_errors() {
        let rows = vec![row("sess-a", "/tmp/proj")];
        assert_eq!(
            session_list_page(rows.clone(), Some("relative"), None, 50).expect_err("relative"),
            SessionListError::RelativeCwd
        );
        assert_eq!(
            session_list_page(rows, None, Some("not-a-cursor"), 50).expect_err("cursor"),
            SessionListError::InvalidCursor
        );
    }

    #[test]
    fn trailing_slash_matches() {
        assert!(same_cwd("/tmp/proj", "/tmp/proj/"));
        assert!(!same_cwd("/tmp/proj", "/tmp/other"));
    }
}

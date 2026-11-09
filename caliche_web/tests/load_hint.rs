//! Shared empty and failure sentences.

use caliche_web::components::chrome::{
    EMPTY_SENTENCE, FLEET_SENTENCE, LoadKind, REQUEST_FAILED_SENTENCE, TOKEN_REJECTED_SENTENCE,
    load_hint_body, load_kind_from_error, server_message_from_error, session_display_title,
};
use caliche_web::components::history_view::{history_group_label, session_matches_query};
use chrono::NaiveDate;

#[test]
fn empty_sentence_ignores_address_and_message() {
    let body = load_hint_body(LoadKind::Empty, "http://127.0.0.1:3060", Some("nope"));
    assert_eq!(body, EMPTY_SENTENCE);
}

#[test]
fn request_failed_includes_address_and_server_message() {
    let error = "RPC transport failed for endpoints.list: HTTP 500 {\"message\":\"backend down\"}";
    assert_eq!(load_kind_from_error(error), LoadKind::RequestFailed);
    assert_eq!(
        server_message_from_error(error).as_deref(),
        Some("backend down")
    );
    assert_eq!(
        load_hint_body(
            LoadKind::RequestFailed,
            "http://127.0.0.1:3060",
            server_message_from_error(error).as_deref()
        ),
        format!("{REQUEST_FAILED_SENTENCE}. http://127.0.0.1:3060. backend down")
    );
}

#[test]
fn http_401_is_token_rejected_and_keeps_plain_server_text() {
    let error = "HTTP 401 invalid token";
    assert_eq!(load_kind_from_error(error), LoadKind::TokenRejected);
    assert_eq!(
        server_message_from_error(error).as_deref(),
        Some("invalid token")
    );
    assert_eq!(
        load_hint_body(
            LoadKind::TokenRejected,
            "http://127.0.0.1:3060",
            Some("invalid token")
        ),
        format!("{TOKEN_REJECTED_SENTENCE}. http://127.0.0.1:3060. invalid token")
    );
}

#[test]
fn http_401_without_body_omits_the_server_message() {
    let error = "HTTP 401";
    assert_eq!(load_kind_from_error(error), LoadKind::TokenRejected);
    assert_eq!(server_message_from_error(error), None);
    assert_eq!(
        load_hint_body(LoadKind::TokenRejected, "http://127.0.0.1:3060", None),
        format!("{TOKEN_REJECTED_SENTENCE}. http://127.0.0.1:3060")
    );
}

#[test]
fn access_denied_is_not_a_rejected_token() {
    let error = "HTTP 403 session access denied";
    assert_eq!(load_kind_from_error(error), LoadKind::RequestFailed);
    assert_eq!(
        server_message_from_error(error).as_deref(),
        Some("session access denied")
    );
}

#[test]
fn nested_json_error_message_is_the_server_text() {
    let error = "HTTP 502 {\"error\":{\"message\":\"upstream timeout\"}}";
    assert_eq!(
        server_message_from_error(error).as_deref(),
        Some("upstream timeout")
    );
}

#[test]
fn session_title_falls_back_to_untitled() {
    assert_eq!(session_display_title(Some("  desk  ")), "desk");
    assert_eq!(session_display_title(Some("   ")), "untitled");
    assert_eq!(session_display_title(None), "untitled");
}

#[test]
fn fleet_sentence_names_workers_in_this_process() {
    assert_eq!(FLEET_SENTENCE, "every worker in this process");
}

#[test]
fn history_query_and_group_use_server_fields() {
    assert!(session_matches_query(Some("Desk"), "abc", "desk"));
    assert!(!session_matches_query(Some("Desk"), "abc", "other"));
    let today = NaiveDate::from_ymd_opt(2026, 10, 3).unwrap_or_default();
    assert_eq!(history_group_label("2026-10-03T12:00:00Z", today), "today");
    assert_eq!(history_group_label("not-a-date", today), "older");
}

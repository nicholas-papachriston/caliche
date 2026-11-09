//! Multipart upload size limit tests.

use axum::body::Body;
use axum::extract::{FromRequest, multipart::Multipart};
use axum::http::{Request, StatusCode};
use caliche_server::handlers::multipart_limits::{
    MAX_ATTACHMENT_BYTES, MAX_ATTACHMENTS, MAX_MULTIPART_FIELD_BYTES, field_too_large,
    read_text_field, too_many_attachments,
};

#[test]
fn multipart_limit_constants() {
    assert_eq!(MAX_MULTIPART_FIELD_BYTES, 64 * 1024);
    assert_eq!(MAX_ATTACHMENT_BYTES, 10 * 1024 * 1024);
    assert_eq!(MAX_ATTACHMENTS, 16);
}

#[test]
fn field_too_large_returns_payload_too_large() {
    let (status, message) = field_too_large("message");
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert!(message.contains("message"));
    assert!(message.contains("size limit"));
}

#[test]
fn too_many_attachments_returns_payload_too_large() {
    let (status, message) = too_many_attachments();
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert!(message.contains(&MAX_ATTACHMENTS.to_string()));
}

#[tokio::test]
async fn read_text_field_rejects_oversized_body() {
    let boundary = "test-boundary";
    let oversized = "x".repeat(MAX_MULTIPART_FIELD_BYTES + 1);
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"message\"\r\n\r\n{oversized}\r\n--{boundary}--\r\n"
    );
    let request = Request::builder()
        .method("POST")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .expect("request");

    let mut multipart = Multipart::from_request(request, &())
        .await
        .expect("multipart parser should accept body");
    let field = multipart
        .next_field()
        .await
        .expect("next field")
        .expect("field should exist");

    let err = read_text_field(field, "message")
        .await
        .expect_err("oversized field should be rejected");
    assert_eq!(err.0, StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn read_text_field_accepts_within_limit() {
    let boundary = "test-boundary";
    let payload = "hello";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"message\"\r\n\r\n{payload}\r\n--{boundary}--\r\n"
    );
    let request = Request::builder()
        .method("POST")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .expect("request");

    let mut multipart = Multipart::from_request(request, &())
        .await
        .expect("multipart parser should accept body");
    let field = multipart
        .next_field()
        .await
        .expect("next field")
        .expect("field should exist");

    let value = read_text_field(field, "message")
        .await
        .expect("field within limit should pass");
    assert_eq!(value, payload);
}

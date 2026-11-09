//! Shared multipart parsing with size caps for chat upload endpoints.

use axum::extract::multipart::Field;
use axum::http::StatusCode;

/// Maximum bytes per non-attachment text field.
pub const MAX_MULTIPART_FIELD_BYTES: usize = 64 * 1024;
/// Maximum bytes per file attachment.
pub const MAX_ATTACHMENT_BYTES: usize = 10 * 1024 * 1024;
/// Maximum number of attachments per request.
pub const MAX_ATTACHMENTS: usize = 16;

pub fn field_too_large(field: &str) -> (StatusCode, String) {
    (
        StatusCode::PAYLOAD_TOO_LARGE,
        format!("multipart field '{field}' exceeds size limit"),
    )
}

pub fn too_many_attachments() -> (StatusCode, String) {
    (
        StatusCode::PAYLOAD_TOO_LARGE,
        format!("at most {MAX_ATTACHMENTS} attachments allowed"),
    )
}

/// Read a text field with a byte cap.
pub async fn read_text_field(field: Field<'_>, name: &str) -> Result<String, (StatusCode, String)> {
    let bytes = field.bytes().await.map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("failed to read form field '{name}': {e}"),
        )
    })?;
    if bytes.len() > MAX_MULTIPART_FIELD_BYTES {
        return Err(field_too_large(name));
    }
    String::from_utf8(bytes.to_vec()).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("form field '{name}' is not valid UTF-8: {e}"),
        )
    })
}

/// Read an attachment field with a byte cap.
pub async fn read_attachment_bytes(
    field: Field<'_>,
) -> Result<(Option<String>, Vec<u8>, String), (StatusCode, String)> {
    let filename = field.file_name().map(str::to_string);
    let bytes = field.bytes().await.map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("failed to read attachment: {e}"),
        )
    })?;
    if bytes.len() > MAX_ATTACHMENT_BYTES {
        return Err(field_too_large("attachments"));
    }
    let media_type = filename
        .as_deref()
        .map(|name| caliche::messages::detect_media_type(std::path::Path::new(name)))
        .unwrap_or_else(|| "application/octet-stream".to_string());
    Ok((filename, bytes.to_vec(), media_type))
}

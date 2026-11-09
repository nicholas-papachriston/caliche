use crate::commands::credentials_service::CredentialService;
use anyhow::Result;
use axum::http::StatusCode;
use std::path::PathBuf;

pub(crate) fn resolve_dir(path_str: &str) -> Result<PathBuf> {
    CredentialService::resolve_dir(path_str)
}

pub(crate) fn internal_error(error: impl std::fmt::Display) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, error.to_string())
}

pub(crate) fn bad_request(error: impl std::fmt::Display) -> (StatusCode, String) {
    (StatusCode::BAD_REQUEST, error.to_string())
}

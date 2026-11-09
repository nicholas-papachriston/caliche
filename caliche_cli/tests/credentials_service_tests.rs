//! CredentialService masking and vault import confirmation tests.

use caliche_cli::commands::credentials_service::{
    CredentialService, SECRET_CONFIRM_FLAG, VAULT_IMPORT_CONFIRM_MESSAGE,
};

#[test]
fn mask_for_display_never_reveals_partial_key_material() {
    assert_eq!(
        CredentialService::mask_for_display("sk-live-secret"),
        "****"
    );
    assert_eq!(CredentialService::mask_for_display(""), "****");
}

#[test]
fn format_credential_value_masks_by_default() {
    assert_eq!(
        CredentialService::format_credential_value("sk-secret", false),
        "****"
    );
}

#[test]
fn format_credential_value_unmasks_when_requested() {
    assert_eq!(
        CredentialService::format_credential_value("sk-secret", true),
        "sk-secret"
    );
}

#[test]
fn require_secret_confirmation_blocks_without_flag() {
    let err = CredentialService::require_secret_confirmation(false, "config export")
        .expect_err("should require explicit confirmation");
    assert!(err.to_string().contains(SECRET_CONFIRM_FLAG));
}

#[test]
fn require_secret_confirmation_allows_when_confirmed() {
    CredentialService::require_secret_confirmation(true, "config export").expect("confirmed");
}

#[test]
fn validate_vault_import_confirm_requires_true() {
    let err = CredentialService::validate_vault_import_confirm(None).expect_err("missing confirm");
    assert_eq!(err.to_string(), VAULT_IMPORT_CONFIRM_MESSAGE);

    let err =
        CredentialService::validate_vault_import_confirm(Some(false)).expect_err("false confirm");
    assert_eq!(err.to_string(), VAULT_IMPORT_CONFIRM_MESSAGE);
}

#[test]
fn validate_vault_import_confirm_accepts_true() {
    CredentialService::validate_vault_import_confirm(Some(true)).expect("confirm true should pass");
}

#[test]
fn vault_import_request_json_confirm_field() {
    let request: serde_json::Value = serde_json::json!({
        "dir": "/tmp/project",
        "confirm": false
    });
    let confirm = request.get("confirm").and_then(|value| value.as_bool());
    let err =
        CredentialService::validate_vault_import_confirm(confirm).expect_err("false should fail");
    assert_eq!(err.to_string(), VAULT_IMPORT_CONFIRM_MESSAGE);
}

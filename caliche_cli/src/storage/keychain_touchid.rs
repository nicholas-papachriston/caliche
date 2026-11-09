//! macOS keychain access with Touch ID (or device passcode) using the data protection keychain.
//!
//! When `CALICHE_KEYCHAIN_USE_TOUCH_ID=true`, the CLI uses this module instead of the
//! default keyring for the unified credentials entry. The system will prompt for Touch ID
//! (or passcode if no biometrics) when reading or writing credentials.

use anyhow::{Context, Result};
use security_framework::passwords::{
    AccessControlOptions, PasswordOptions, generic_password, set_generic_password_options,
};

/// macOS Security framework: item not found in keychain
const ERR_SEC_ITEM_NOT_FOUND: i32 = -25300;

/// Read a password from the data-protection keychain (Touch ID / passcode protected).
/// Returns `Ok(None)` if the item does not exist.
pub fn get(service: &str, account: &str) -> Result<Option<String>> {
    let mut options = PasswordOptions::new_generic_password(service, account);
    options.set_access_control_options(AccessControlOptions::USER_PRESENCE);
    options.use_protected_keychain();

    match generic_password(options) {
        Ok(bytes) => {
            let s = String::from_utf8(bytes).context("Keychain value was not valid UTF-8")?;
            Ok(Some(s))
        }
        Err(e) => {
            if e.code() == ERR_SEC_ITEM_NOT_FOUND {
                return Ok(None);
            }
            Err(e).context("Failed to read from keychain")
        }
    }
}

/// Write a password to the data-protection keychain (Touch ID / passcode protected).
pub fn set(service: &str, account: &str, value: &str) -> Result<()> {
    let mut options = PasswordOptions::new_generic_password(service, account);
    options.set_access_control_options(AccessControlOptions::USER_PRESENCE);
    options.use_protected_keychain();

    set_generic_password_options(value.as_bytes(), options)
        .context("Failed to write to keychain")?;
    Ok(())
}

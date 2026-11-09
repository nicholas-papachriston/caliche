//! In-process credential values, checked before the process environment.
//!
//! The CLI loads provider and gateway credentials from the keychain, and the
//! user can enter more during a session. Those values go into this store, not
//! into the process environment: writing the environment while the async
//! runtime and other threads read it is a data race.
//!
//! Every credential read goes through [`var`], which returns the stored value
//! for a name, or else the process environment value. A caller that must not
//! replace a value from the shell checks [`var`] before it calls [`set`].
//!
//! Child processes do not see this store. A spawn path that must pass a stored
//! credential to a child reads it here and sets it on that child's command.

use std::collections::HashMap;
use std::env::VarError;
use std::sync::{LazyLock, PoisonError, RwLock};

static STORE: LazyLock<RwLock<HashMap<String, String>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// The value of the credential `name`: the stored value, else the process
/// environment value.
///
/// # Errors
///
/// Returns the [`std::env::var`] error when neither source has the name.
pub fn var(name: &str) -> Result<String, VarError> {
    if let Some(value) = stored(name) {
        return Ok(value);
    }
    std::env::var(name)
}

/// Whether [`var`] finds a value for `name`.
#[must_use]
pub fn is_set(name: &str) -> bool {
    var(name).is_ok()
}

/// Store `value` for `name` in this process. It replaces an earlier stored value.
pub fn set(name: impl Into<String>, value: impl Into<String>) {
    STORE
        .write()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(name.into(), value.into());
}

/// Remove the stored value for `name`, so [`var`] reads the process
/// environment again.
pub fn remove(name: &str) {
    STORE
        .write()
        .unwrap_or_else(PoisonError::into_inner)
        .remove(name);
}

/// Every stored name and value, sorted by name.
#[must_use]
pub fn snapshot() -> Vec<(String, String)> {
    let mut entries: Vec<(String, String)> = STORE
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    entries.sort();
    entries
}

fn stored(name: &str) -> Option<String> {
    STORE
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .get(name)
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_value_is_returned_and_the_environment_stays_unset() {
        let name = "CALICHE_CREDENTIAL_ENV_TEST_STORED";
        assert_eq!(var(name), Err(VarError::NotPresent));

        set(name, "stored-value");

        assert_eq!(var(name).as_deref(), Ok("stored-value"));
        assert!(is_set(name));
        assert_eq!(std::env::var(name), Err(VarError::NotPresent));
        assert!(snapshot().contains(&(name.to_string(), "stored-value".to_string())));
    }

    #[test]
    fn a_removed_value_is_no_longer_returned() {
        let name = "CALICHE_CREDENTIAL_ENV_TEST_REMOVED";
        set(name, "stored-value");

        remove(name);

        assert_eq!(var(name), Err(VarError::NotPresent));
        assert!(!snapshot().iter().any(|(stored, _)| stored == name));
    }

    #[test]
    fn environment_value_is_used_when_nothing_is_stored() {
        // PATH is set for every test process and is never stored.
        assert_eq!(var("PATH"), std::env::var("PATH"));
    }
}

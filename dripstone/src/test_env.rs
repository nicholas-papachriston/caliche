//! Process-wide environment lock for dripstone unit tests.
//!
//! The guard records each variable it sets or removes and writes the previous
//! value back on drop. Cache-dir tests must point at a [`tempfile::TempDir`].

use std::ffi::{OsStr, OsString};
use std::sync::{Mutex, MutexGuard};

/// Shared lock for process environment mutation in this crate's unit tests.
pub static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Saves and restores every environment variable it changes.
pub struct EnvGuard {
    _lock: MutexGuard<'static, ()>,
    saved: Vec<(&'static str, Option<OsString>)>,
}

impl EnvGuard {
    /// Lock [`ENV_LOCK`] and start a restore scope.
    pub fn acquire() -> Self {
        let lock = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Self {
            _lock: lock,
            saved: Vec::new(),
        }
    }

    /// Set `key`, remembering the previous value the first time.
    pub fn set(&mut self, key: &'static str, value: impl AsRef<OsStr>) {
        self.remember(key);
        #[expect(
            unsafe_code,
            clippy::disallowed_methods,
            reason = "unit tests mutate process env under ENV_LOCK and restore it on drop"
        )]
        unsafe {
            std::env::set_var(key, value);
        }
    }

    /// Remove `key`, remembering the previous value the first time.
    pub fn remove(&mut self, key: &'static str) {
        self.remember(key);
        #[expect(
            unsafe_code,
            clippy::disallowed_methods,
            reason = "unit tests clear process env under ENV_LOCK and restore it on drop"
        )]
        unsafe {
            std::env::remove_var(key);
        }
    }

    fn remember(&mut self, key: &'static str) {
        if self.saved.iter().any(|(saved, _)| *saved == key) {
            return;
        }
        self.saved.push((key, std::env::var_os(key)));
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, previous) in self.saved.drain(..).rev() {
            #[expect(
                unsafe_code,
                clippy::disallowed_methods,
                reason = "unit tests restore process env when the guard drops"
            )]
            unsafe {
                match previous {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
    }
}

//! The one guard for changing the process environment in tests.
//!
//! `std::env::set_var` and `remove_var` are `unsafe` because any thread,
//! including C code such as `getaddrinfo`, time zone lookups or OpenSSL, may
//! read the environment at the same moment, and nothing orders those reads.
//! [`EnvGuard`] is the only place in the workspace that calls them. The lint
//! check in `tests/no_raw_env_mutation.rs` and `disallowed-methods` in
//! `clippy.toml` keep it that way.
//!
//! A guard takes a process-wide lock, records the first previous value of
//! each variable it changes, and writes every one back when it drops, also
//! while a panic unwinds.

use std::cell::Cell;
use std::ffi::{OsStr, OsString};
use std::sync::{Mutex, MutexGuard, PoisonError};

/// Orders guard users in one test binary. It does not order anything else.
static ENV_LOCK: Mutex<()> = Mutex::new(());

thread_local! {
    /// Whether this thread holds [`ENV_LOCK`] through a live [`EnvGuard`].
    static HOLDS_ENV_LOCK: Cell<bool> = const { Cell::new(false) };
}

/// Saves and restores every environment variable it changes.
///
/// # Examples
///
/// ```
/// use caliche_test_env::EnvGuard;
///
/// // SAFETY: this is the only doctest in the crate, and it starts no threads.
/// let mut env = unsafe { EnvGuard::acquire() };
/// env.set("CALICHE_TEST_ENV_DOC", "on");
/// assert_eq!(std::env::var("CALICHE_TEST_ENV_DOC").as_deref(), Ok("on"));
/// drop(env);
/// assert!(std::env::var("CALICHE_TEST_ENV_DOC").is_err());
/// ```
#[must_use = "the environment is restored when the guard drops"]
pub struct EnvGuard {
    _lock: MutexGuard<'static, ()>,
    saved: Vec<(OsString, Option<OsString>)>,
}

impl EnvGuard {
    /// Lock the process environment for this test and start a restore scope.
    ///
    /// A guard left poisoned by a panicking test is taken over: that guard
    /// already restored its variables while it unwound.
    ///
    /// # Safety
    ///
    /// While the returned guard lives, no other thread, including C code,
    /// may read or write the process environment.
    ///
    /// The lock only orders guard users. It does not stop a test without a
    /// guard, a Tokio worker, or a C library from reading the environment
    /// at the same moment. So every test that holds a guard must also be
    /// `#[serial]` (or be the only test in its binary), must not spawn
    /// threads that read the environment while the guard lives, and must not
    /// run beside a test that reads the environment without a guard.
    #[expect(
        unsafe_code,
        reason = "the caller must keep every other env reader and writer out"
    )]
    pub unsafe fn acquire() -> Self {
        let lock = ENV_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        HOLDS_ENV_LOCK.set(true);
        Self {
            _lock: lock,
            saved: Vec::new(),
        }
    }

    /// Set `key`, remembering its previous value the first time.
    pub fn set(&mut self, key: impl AsRef<OsStr>, value: impl AsRef<OsStr>) {
        let key = key.as_ref();
        self.remember(key);
        #[expect(
            unsafe_code,
            clippy::disallowed_methods,
            reason = "the one env writer; the acquire contract excludes every other reader and writer"
        )]
        // SAFETY: `acquire` made the caller promise that no other thread
        // reads or writes the environment while this guard lives.
        unsafe {
            std::env::set_var(key, value);
        }
    }

    /// Remove `key`, remembering its previous value the first time.
    pub fn remove(&mut self, key: impl AsRef<OsStr>) {
        let key = key.as_ref();
        self.remember(key);
        #[expect(
            unsafe_code,
            clippy::disallowed_methods,
            reason = "the one env writer; the acquire contract excludes every other reader and writer"
        )]
        // SAFETY: as in `set`.
        unsafe {
            std::env::remove_var(key);
        }
    }

    fn remember(&mut self, key: &OsStr) {
        if self.saved.iter().any(|(saved, _)| saved == key) {
            return;
        }
        self.saved.push((key.to_os_string(), std::env::var_os(key)));
    }
}

/// Set `key` for the rest of the process. Nothing restores it.
///
/// For a fixture that every test in a binary shares, such as a temporary
/// config directory that must outlive each test.
///
/// # Safety
///
/// The [`EnvGuard::acquire`] contract, for the duration of this call: no
/// other thread, including C code, reads or writes the process environment
/// meanwhile. The call takes the guard lock, so it waits for any guard on
/// another thread; a thread that already holds a guard sets the variable
/// under that guard's lock. The lock only orders guard users, so in practice
/// every test in the calling binary must be `#[serial]`.
#[expect(
    unsafe_code,
    reason = "the caller must keep every other env reader and writer out"
)]
pub unsafe fn set_for_process(key: impl AsRef<OsStr>, value: impl AsRef<OsStr>) {
    let _lock = if HOLDS_ENV_LOCK.get() {
        None
    } else {
        Some(ENV_LOCK.lock().unwrap_or_else(PoisonError::into_inner))
    };
    #[expect(
        unsafe_code,
        clippy::disallowed_methods,
        reason = "the one env writer; the caller upholds the acquire contract"
    )]
    // SAFETY: this thread holds the guard lock, and the caller promised that
    // no other thread reads or writes the environment during this call.
    unsafe {
        std::env::set_var(key, value);
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, previous) in self.saved.drain(..).rev() {
            #[expect(
                unsafe_code,
                clippy::disallowed_methods,
                reason = "the one env writer restores what it changed"
            )]
            // SAFETY: the guard is still alive, so the `acquire` contract
            // still holds.
            unsafe {
                match previous {
                    Some(value) => std::env::set_var(&key, value),
                    None => std::env::remove_var(&key),
                }
            }
        }
        // The lock field is released right after this; also runs on unwind.
        HOLDS_ENV_LOCK.set(false);
    }
}

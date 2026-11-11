//! `EnvGuard` restores the environment, also when a test panics.
//!
//! Every test in this binary touches the environment only while it holds a
//! guard, so the guard lock alone orders them, and none starts a thread.
//! That is the `acquire` contract for each SAFETY comment below.

#![expect(
    unsafe_code,
    reason = "every test takes the env guard; each SAFETY comment says why the contract holds"
)]

use caliche_test_env::EnvGuard;
use std::env::VarError;

/// Read `key` while holding a guard, so no other test here writes it meanwhile.
fn read(key: &str) -> Result<String, VarError> {
    // SAFETY: every env access in this binary holds a guard (module docs).
    let _env = unsafe { EnvGuard::acquire() };
    std::env::var(key)
}

#[test]
fn a_panic_while_the_guard_lives_removes_what_it_set() {
    let outcome = std::panic::catch_unwind(|| {
        // SAFETY: every env access in this binary holds a guard (module docs).
        let mut env = unsafe { EnvGuard::acquire() };
        env.set("CALICHE_TEST_ENV_PANIC", "set before the panic");
        panic!("test body failed");
    });
    assert!(outcome.is_err());
    assert_eq!(read("CALICHE_TEST_ENV_PANIC"), Err(VarError::NotPresent));
}

#[test]
fn a_panic_puts_back_a_value_that_was_there_before() {
    let before = read("PATH");
    assert!(before.is_ok(), "the test process has PATH");
    let outcome = std::panic::catch_unwind(|| {
        // SAFETY: every env access in this binary holds a guard (module docs).
        let mut env = unsafe { EnvGuard::acquire() };
        env.set("PATH", "/changed");
        env.remove("PATH");
        panic!("test body failed");
    });
    assert!(outcome.is_err());
    assert_eq!(read("PATH"), before);
}

#[test]
fn drop_restores_the_value_from_before_the_first_change() {
    {
        // SAFETY: every env access in this binary holds a guard (module docs).
        let mut env = unsafe { EnvGuard::acquire() };
        env.set("CALICHE_TEST_ENV_FIRST", "one");
        env.set("CALICHE_TEST_ENV_FIRST", "two");
        assert_eq!(
            std::env::var("CALICHE_TEST_ENV_FIRST").as_deref(),
            Ok("two")
        );
    }
    assert_eq!(read("CALICHE_TEST_ENV_FIRST"), Err(VarError::NotPresent));
}

#[test]
fn a_guard_after_a_panicking_guard_still_locks() {
    let outcome = std::panic::catch_unwind(|| {
        // SAFETY: every env access in this binary holds a guard (module docs).
        let _env = unsafe { EnvGuard::acquire() };
        panic!("poisons the lock");
    });
    assert!(outcome.is_err());
    // SAFETY: every env access in this binary holds a guard (module docs).
    let mut env = unsafe { EnvGuard::acquire() };
    env.set("CALICHE_TEST_ENV_AFTER_POISON", "ok");
    assert_eq!(
        std::env::var("CALICHE_TEST_ENV_AFTER_POISON").as_deref(),
        Ok("ok")
    );
}

#[test]
fn set_for_process_does_not_wait_for_a_guard_held_by_its_own_thread() {
    let (done, finished) = std::sync::mpsc::channel();
    // Detached: if the call waits on its own lock, the test fails, not hangs.
    std::thread::spawn(move || {
        // SAFETY: every env access in this binary holds a guard (module docs).
        let _env = unsafe { EnvGuard::acquire() };
        // SAFETY: this thread holds the guard, so no other test touches the
        // environment.
        unsafe { caliche_test_env::set_for_process("CALICHE_TEST_ENV_OWN", "kept") };
        done.send(()).ok();
    });
    let outcome = finished.recv_timeout(std::time::Duration::from_secs(5));
    assert!(
        outcome.is_ok(),
        "set_for_process waited on its own thread's guard"
    );
    assert_eq!(read("CALICHE_TEST_ENV_OWN").as_deref(), Ok("kept"));
}

#[test]
fn set_for_process_waits_for_a_guard_on_another_thread() {
    // SAFETY: every env access in this binary holds a guard (module docs);
    // the setter below runs only after this guard drops.
    let env = unsafe { EnvGuard::acquire() };
    let (done, finished) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        // SAFETY: the setter takes the guard lock, so it runs only when no
        // guard in this binary is alive.
        unsafe { caliche_test_env::set_for_process("CALICHE_TEST_ENV_OTHER", "after") };
        done.send(()).ok();
    });
    let early = finished.recv_timeout(std::time::Duration::from_millis(300));
    assert!(
        early.is_err(),
        "set_for_process ran while another thread held the guard"
    );
    drop(env);
    let late = finished.recv_timeout(std::time::Duration::from_secs(5));
    assert!(
        late.is_ok(),
        "set_for_process did not run after the guard dropped"
    );
    assert_eq!(read("CALICHE_TEST_ENV_OTHER").as_deref(), Ok("after"));
}

#[test]
fn a_guard_dropped_in_a_panic_no_longer_counts_as_held() {
    let (ready, is_ready) = std::sync::mpsc::channel();
    let (go, wait_go) = std::sync::mpsc::channel::<()>();
    let (done, finished) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let outcome = std::panic::catch_unwind(|| {
            // SAFETY: every env access in this binary holds a guard (module docs).
            let _env = unsafe { EnvGuard::acquire() };
            panic!("test body failed");
        });
        assert!(outcome.is_err());
        ready.send(()).ok();
        wait_go.recv().ok();
        // SAFETY: the setter takes the guard lock, so it runs only when no
        // guard in this binary is alive.
        unsafe { caliche_test_env::set_for_process("CALICHE_TEST_ENV_UNWOUND", "after") };
        done.send(()).ok();
    });
    assert!(
        is_ready
            .recv_timeout(std::time::Duration::from_secs(5))
            .is_ok()
    );
    // SAFETY: every env access in this binary holds a guard (module docs);
    // the setter runs only after this guard drops.
    let env = unsafe { EnvGuard::acquire() };
    go.send(()).ok();
    let early = finished.recv_timeout(std::time::Duration::from_millis(300));
    assert!(
        early.is_err(),
        "the unwound thread still counted as holding the guard"
    );
    drop(env);
    assert!(
        finished
            .recv_timeout(std::time::Duration::from_secs(5))
            .is_ok()
    );
}

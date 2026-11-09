//! Test-only faults for one Landlock exec.
//!
//! `run_exec` calls into this module under `cfg(test)`. The flags stay out of
//! the production function so a release build has no fault branch.

use std::os::fd::RawFd;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};

pub(super) static FAULT_RULESET_FD: AtomicBool = AtomicBool::new(false);
pub(super) static CLEAR_RULESET_CLOEXEC: AtomicBool = AtomicBool::new(false);

pub(super) fn child_ruleset_fd(fd: RawFd) -> RawFd {
    if CLEAR_RULESET_CLOEXEC.load(Ordering::Relaxed) {
        // SAFETY: this test flag clears close-on-exec so a child-side check
        // can see the descriptor when the pre-exec hook stops marking it.
        #[expect(unsafe_code, reason = "test-only clear of ruleset CLOEXEC")]
        unsafe {
            libc::fcntl(fd, libc::F_SETFD, 0);
        }
    }
    if FAULT_RULESET_FD.load(Ordering::Relaxed) {
        return -1;
    }
    fd
}

pub(super) fn note_ruleset_fd(cmd: &mut Command, ruleset_fd: RawFd) {
    if CLEAR_RULESET_CLOEXEC.load(Ordering::Relaxed) {
        cmd.env("CLS_RULESET_FD", ruleset_fd.to_string());
    }
}

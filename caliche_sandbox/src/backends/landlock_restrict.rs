//! Async-signal-safe Landlock restrict for the forked child.
//!
//! The parent builds the ruleset and opens every path fd. This module is the
//! only code `Command::pre_exec` may run. A unit test allowlists both `::`
//! paths and bare call names in the marked region, so a later edit that
//! formats, logs, locks, or calls a helper defined outside the region fails
//! the build's tests.

use std::os::fd::RawFd;

// BEGIN SIGNAL_SAFE
const CLOEXEC_FIRST: RawFd = 3;
const EINTR_TRIES: i32 = 16;
/// CLOSE_RANGE_CLOEXEC is 1 shifted left by 2.
const CLOSE_RANGE_CLOEXEC: u32 = 4;

/// Apply a new session, cgroup membership, the `.git` mount guard when one is
/// set, a prepared Landlock ruleset, and close-on-exec on every descriptor
/// from 3 upward.
///
/// `close_range` with the cloexec flag is present on every kernel that has
/// Landlock. Landlock starts at 5.13 and the flag at 5.11. If that syscall
/// returns ENOSYS or EINVAL, fcntl walks the soft NOFILE limit instead. An
/// infinite limit fails closed with ERANGE. The error carries the raw errno
/// and does not allocate.
pub(super) fn pre_exec_restrict(
    cgroup_procs: Option<RawFd>,
    git_mount: Option<&super::git_guard::MountExec>,
    ruleset_fd: RawFd,
) -> std::io::Result<()> {
    if setsid_detach() < 0 {
        return Err(std::io::Error::from_raw_os_error(super::cgroup::raw_errno()));
    }
    if let Some(fd) = cgroup_procs {
        super::cgroup::join(fd)?;
    }
    if let Some(mount) = git_mount {
        super::git_guard::enter(mount)?;
    }
    if prctl_no_new_privs() != 0 {
        return Err(std::io::Error::from_raw_os_error(super::cgroup::raw_errno()));
    }
    if landlock_restrict(ruleset_fd) != 0 {
        return Err(std::io::Error::from_raw_os_error(super::cgroup::raw_errno()));
    }
    mark_inherited_cloexec()
}

fn setsid_detach() -> i32 {
    // SAFETY: `setsid` is async-signal-safe. It makes this process the leader
    // of a new session, so it has no controlling terminal. A failure is EPERM
    // when the caller is already a process-group leader.
    #[expect(unsafe_code, reason = "async-signal-safe setsid")]
    unsafe {
        libc::setsid()
    }
}

fn prctl_no_new_privs() -> i32 {
    // SAFETY: these arguments only set the no-new-privs flag. `prctl` is
    // async-signal-safe. Landlock refuses the restrict syscall without it
    // unless the caller has CAP_SYS_ADMIN.
    #[expect(unsafe_code, reason = "async-signal-safe no_new_privs")]
    unsafe {
        libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
    }
}

fn landlock_restrict(ruleset_fd: RawFd) -> i64 {
    // SAFETY: `ruleset_fd` is a Landlock ruleset created in the parent. The
    // kernel requires a zero flags argument. `syscall` is async-signal-safe.
    #[expect(unsafe_code, reason = "async-signal-safe landlock_restrict_self")]
    unsafe {
        libc::syscall(libc::SYS_landlock_restrict_self, ruleset_fd, 0) as i64
    }
}

fn mark_inherited_cloexec() -> std::io::Result<()> {
    let mut tries = 0;
    while tries < EINTR_TRIES {
        if close_range_cloexec() == 0 {
            return Ok(());
        }
        let err = super::cgroup::raw_errno();
        if err != libc::EINTR {
            if err == libc::ENOSYS || err == libc::EINVAL {
                return cloexec_fallback();
            }
            return Err(std::io::Error::from_raw_os_error(err));
        }
        tries += 1;
    }
    cloexec_fallback()
}

fn close_range_cloexec() -> i64 {
    // SAFETY: marks every descriptor from 3 upward close-on-exec. It does not
    // close them, so a still-running parent thread is unaffected. `syscall`
    // is async-signal-safe.
    #[expect(unsafe_code, reason = "async-signal-safe close_range")]
    unsafe {
        libc::syscall(
            libc::SYS_close_range,
            CLOEXEC_FIRST,
            !0u32,
            CLOSE_RANGE_CLOEXEC,
        ) as i64
    }
}

/// fcntl walk for a kernel that rejects `close_range`. Every descriptor from
/// 3 up to the soft limit is marked. There is no keep-list: `close_range`
/// cannot spare one fd, and no caller needs one spared.
fn cloexec_fallback() -> std::io::Result<()> {
    let mut lim = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `getrlimit` is async-signal-safe. The limit struct is local.
    #[expect(unsafe_code, reason = "async-signal-safe getrlimit")]
    let rc = unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut lim) };
    if rc != 0 {
        return Err(std::io::Error::from_raw_os_error(super::cgroup::raw_errno()));
    }
    let last = cloexec_ceiling(lim.rlim_cur)?;
    let mut fd = CLOEXEC_FIRST;
    while fd < last {
        set_one_cloexec(fd);
        fd += 1;
    }
    Ok(())
}

/// Last descriptor the fcntl walk may touch. An infinite soft limit, or one
/// that does not fit in a descriptor, fails closed. The walk never caps the
/// range at 1024.
fn cloexec_ceiling(rlim_cur: libc::rlim_t) -> std::io::Result<RawFd> {
    if rlim_cur == libc::RLIM_INFINITY || rlim_cur > RawFd::MAX as libc::rlim_t {
        return Err(std::io::Error::from_raw_os_error(libc::ERANGE));
    }
    Ok(rlim_cur as RawFd)
}

fn set_one_cloexec(fd: RawFd) {
    // SAFETY: `fcntl` `F_GETFD` and `F_SETFD` are async-signal-safe. A bad
    // descriptor is ignored so the fallback can walk the whole table.
    #[expect(unsafe_code, reason = "async-signal-safe fcntl")]
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    if flags < 0 {
        return;
    }
    #[expect(unsafe_code, reason = "async-signal-safe fcntl")]
    let _set = unsafe { libc::fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC) };
}
// END SIGNAL_SAFE

#[cfg(test)]
mod tests {
    use super::pre_exec_restrict;
    use std::os::fd::RawFd;

    const ALLOWED_RESTRICT: &[&str] = &[
        "super::cgroup::join",
        "super::git_guard::MountExec",
        "super::git_guard::enter",
        "super::cgroup::raw_errno",
        "std::io::Error::from_raw_os_error",
        "std::io::Result",
        "libc::setsid",
        "libc::prctl",
        "libc::PR_SET_NO_NEW_PRIVS",
        "libc::syscall",
        "libc::SYS_landlock_restrict_self",
        "libc::SYS_close_range",
        "libc::EINTR",
        "libc::ENOSYS",
        "libc::EINVAL",
        "libc::rlimit",
        "libc::getrlimit",
        "libc::RLIMIT_NOFILE",
        "libc::RLIM_INFINITY",
        "libc::rlim_t",
        "libc::ERANGE",
        "libc::fcntl",
        "libc::F_GETFD",
        "libc::F_SETFD",
        "libc::FD_CLOEXEC",
        "RawFd::MAX",
    ];
    const REQUIRED_RESTRICT: &[&str] = &[
        "super::git_guard::enter",
        "libc::setsid",
        "libc::prctl",
        "libc::PR_SET_NO_NEW_PRIVS",
        "libc::syscall",
        "libc::SYS_landlock_restrict_self",
        "libc::SYS_close_range",
        "libc::ERANGE",
        "super::cgroup::join",
    ];
    const SKIP_BARE: &[&str] = &["pub", "crate", "super", "self", "mut", "fn", "unsafe"];
    const BARE_ALLOW: &[&str] = &[
        "setsid",
        "prctl",
        "syscall",
        "from_raw_os_error",
        "close",
        "write",
        "getrlimit",
        "fcntl",
        "__errno_location",
        "join",
        "Some",
        "Ok",
        "Err",
    ];
    const METHOD_ALLOW: &[&str] = &["as_ptr", "cast", "len", "unsigned_abs"];
    const ALLOWED_JOIN: &[&str] = &[
        "libc::write",
        "libc::__errno_location",
        "libc::EIO",
        "std::io::Error::from_raw_os_error",
        "std::io::Result",
    ];
    const REQUIRED_JOIN: &[&str] = &["libc::write"];
    const ALLOWED_GIT_GUARD: &[&str] = &[
        "libc::c_uint",
        "libc::c_int",
        "std::io::Result",
        "std::io::Error::from_raw_os_error",
        "super::cgroup::raw_errno",
        "libc::unshare",
        "libc::CLONE_NEWUSER",
        "libc::CLONE_NEWNS",
        "libc::mount",
        "libc::MS_REC",
        "libc::MS_PRIVATE",
        "std::ptr::null",
        "std::ptr::from_ref",
        "libc::chdir",
        "libc::stat",
        "libc::fstat",
        "std::mem::zeroed",
        "libc::ESTALE",
        "libc::EIO",
        "libc::syscall",
        "libc::SYS_open_tree",
        "libc::SYS_mount_setattr",
        "libc::SYS_move_mount",
        "libc::OPEN_TREE_CLONE",
        "libc::OPEN_TREE_CLOEXEC",
        "libc::AT_EMPTY_PATH",
        "libc::AT_RECURSIVE",
        "libc::mount_attr",
        "libc::MOUNT_ATTR_RDONLY",
        "libc::open",
        "libc::O_PATH",
        "libc::O_NOFOLLOW",
        "libc::O_CLOEXEC",
        "libc::O_WRONLY",
        "libc::write",
        "libc::close",
    ];
    const REQUIRED_GIT_GUARD: &[&str] = &[
        "libc::unshare",
        "libc::SYS_open_tree",
        "libc::SYS_mount_setattr",
        "libc::SYS_move_mount",
        "libc::MOUNT_ATTR_RDONLY",
        "libc::O_NOFOLLOW",
        "libc::chdir",
    ];
    /// libc and std calls the `.git` guard makes by path.
    const BARE_ALLOW_GIT_GUARD: &[&str] = &[
        "unshare", "mount", "chdir", "fstat", "open", "zeroed", "null", "from_ref",
    ];
    const DENIED: &[&str] = &[
        "format!",
        "format_args",
        "println!",
        "eprintln!",
        "panic!",
        "vec!",
        "write!",
        "writeln!",
        "assert!",
        "todo!",
        "unimplemented!",
        "dbg!",
        "String",
        "Vec",
        "Box",
        "PathBuf",
        "to_string",
        "to_owned",
        ".clone(",
        "tracing",
        "log::",
        "Ruleset",
        "OpenOptions",
        "File::",
        "Mutex",
        "unwrap",
        "alloc::",
    ];

    fn marked<'a>(source: &'a str, begin: &str, end: &str) -> &'a str {
        let start = source
            .find(begin)
            .unwrap_or_else(|| panic!("missing {begin}"))
            + begin.len();
        let stop = source[start..]
            .find(end)
            .unwrap_or_else(|| panic!("missing {end}"))
            + start;
        &source[start..stop]
    }

    fn path_tokens(region: &str) -> Vec<&str> {
        let bytes = region.as_bytes();
        let mut tokens = Vec::new();
        let mut index = 0;
        while index < bytes.len() {
            if bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_' {
                let start = index;
                index += 1;
                while index < bytes.len()
                    && (bytes[index].is_ascii_alphanumeric()
                        || bytes[index] == b'_'
                        || bytes[index] == b':')
                {
                    index += 1;
                }
                let token = &region[start..index];
                if token.contains("::") {
                    tokens.push(token);
                }
            } else {
                index += 1;
            }
        }
        tokens
    }

    fn bare_calls(region: &str) -> Vec<(&str, bool)> {
        let bytes = region.as_bytes();
        let mut names = Vec::new();
        let mut index = 0;
        while index < bytes.len() {
            if bytes[index].is_ascii_alphabetic() || bytes[index] == b'_' {
                let start = index;
                index += 1;
                while index < bytes.len()
                    && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_')
                {
                    index += 1;
                }
                let mut next = index;
                while next < bytes.len() && bytes[next].is_ascii_whitespace() {
                    next += 1;
                }
                if next < bytes.len() && bytes[next] == b'(' {
                    let name = &region[start..index];
                    let attribute = start > 0 && bytes[start - 1] == b'[';
                    let method = start > 0 && bytes[start - 1] == b'.';
                    if name.len() > 1 && !attribute && !SKIP_BARE.contains(&name) {
                        names.push((name, method));
                    }
                }
            } else {
                index += 1;
            }
        }
        names
    }

    fn assert_signal_safe(region: &str, allowed: &[&str], required: &[&str]) {
        for token in path_tokens(region) {
            assert!(
                allowed.contains(&token),
                "disallowed path in the pre-exec hook: {token}"
            );
        }
        for path in required {
            assert!(region.contains(path), "pre-exec hook is missing {path}");
        }
        for word in DENIED {
            assert!(
                !region.contains(word),
                "pre-exec hook contains allocation-capable `{word}`"
            );
        }
    }

    fn assert_bare_calls(region: &str, also_defined_in: &str) {
        assert_bare_calls_with(region, also_defined_in, &[]);
    }

    fn assert_bare_calls_with(region: &str, also_defined_in: &str, extra: &[&str]) {
        for (name, method) in bare_calls(region) {
            if method {
                assert!(
                    METHOD_ALLOW.contains(&name),
                    "method `{name}` is outside the signal-safe allowlist"
                );
                continue;
            }
            let defined_here = region.contains(&format!("fn {name}"));
            let defined_join = also_defined_in.contains(&format!("fn {name}"));
            assert!(
                BARE_ALLOW.contains(&name) || extra.contains(&name) || defined_here || defined_join,
                "bare call `{name}` is outside the signal-safe allowlist"
            );
        }
    }

    #[test]
    fn infinite_nofile_returns_erange() {
        let infinite = super::cloexec_ceiling(libc::RLIM_INFINITY);
        assert_eq!(
            infinite.err().and_then(|err| err.raw_os_error()),
            Some(libc::ERANGE),
            "an infinite soft limit must fail closed"
        );
        let wide = super::cloexec_ceiling(RawFd::MAX as libc::rlim_t + 1);
        assert_eq!(
            wide.err().and_then(|err| err.raw_os_error()),
            Some(libc::ERANGE),
            "a limit above a descriptor must fail closed"
        );
        assert_eq!(super::cloexec_ceiling(1024).ok(), Some(1024));
    }

    #[test]
    fn pre_exec_restrict_signature_is_fds_to_io_result() {
        let _check: fn(
            Option<RawFd>,
            Option<&super::super::git_guard::MountExec>,
            RawFd,
        ) -> std::io::Result<()> = pre_exec_restrict;
    }

    #[test]
    fn pre_exec_body_uses_only_signal_safe_calls() {
        let region = marked(
            include_str!("landlock_restrict.rs"),
            "BEGIN SIGNAL_SAFE",
            "END SIGNAL_SAFE",
        );
        let join = marked(
            include_str!("cgroup.rs"),
            "BEGIN SIGNAL_SAFE_JOIN",
            "END SIGNAL_SAFE_JOIN",
        );
        let guard = marked(
            include_str!("git_guard.rs"),
            "BEGIN SIGNAL_SAFE_GIT_GUARD",
            "END SIGNAL_SAFE_GIT_GUARD",
        );
        assert_signal_safe(region, ALLOWED_RESTRICT, REQUIRED_RESTRICT);
        assert_bare_calls(region, &format!("{join}{guard}"));
    }

    #[test]
    fn git_guard_uses_only_signal_safe_calls() {
        let region = marked(
            include_str!("git_guard.rs"),
            "BEGIN SIGNAL_SAFE_GIT_GUARD",
            "END SIGNAL_SAFE_GIT_GUARD",
        );
        let join = marked(
            include_str!("cgroup.rs"),
            "BEGIN SIGNAL_SAFE_JOIN",
            "END SIGNAL_SAFE_JOIN",
        );
        assert_signal_safe(region, ALLOWED_GIT_GUARD, REQUIRED_GIT_GUARD);
        assert_bare_calls_with(region, join, BARE_ALLOW_GIT_GUARD);
    }

    #[test]
    fn cgroup_join_uses_only_a_raw_write() {
        let region = marked(
            include_str!("cgroup.rs"),
            "BEGIN SIGNAL_SAFE_JOIN",
            "END SIGNAL_SAFE_JOIN",
        );
        assert_signal_safe(region, ALLOWED_JOIN, REQUIRED_JOIN);
        assert_bare_calls(region, "");
    }

    #[test]
    fn pre_exec_closure_only_calls_the_signal_safe_helper() {
        let src = include_str!("linux.rs");
        let line = "cmd.pre_exec(move || pre_exec_restrict(cgroup_procs, git_mount.as_ref(), ruleset_fd));";
        assert_eq!(
            src.matches(line).count(),
            1,
            "pre_exec closure must be exactly the signal-safe helper"
        );
        assert!(
            src.contains("let spawned = cmd.spawn();\n    drop(ruleset);"),
            "the parent must close the prepared ruleset fd after spawn"
        );
    }
}

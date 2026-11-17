//! Whether a process id names a running process.

/// True unless `pid` is proven gone.
///
/// A pid that cannot be opened for any reason other than "no such process"
/// counts as running, so a caller never ends work it cannot see. A process
/// that exited but still has an open handle somewhere is gone.
#[cfg(windows)]
#[must_use]
pub fn pid_running(pid: u32) -> bool {
    use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
    use windows_sys::Win32::Foundation::{ERROR_INVALID_PARAMETER, WAIT_OBJECT_0};
    use windows_sys::Win32::System::Threading::{
        OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE, WaitForSingleObject,
    };

    // SAFETY: OpenProcess takes no pointers and returns null on failure.
    #[expect(unsafe_code, reason = "Win32 FFI call to open a process by pid")]
    let raw = unsafe {
        OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_SYNCHRONIZE,
            0,
            pid,
        )
    };
    if raw.is_null() {
        let code = std::io::Error::last_os_error().raw_os_error();
        return code.and_then(|code| u32::try_from(code).ok()) != Some(ERROR_INVALID_PARAMETER);
    }
    // SAFETY: a non-null `raw` is a process handle that nothing else owns.
    #[expect(unsafe_code, reason = "adopt the process handle so it is closed")]
    let process = unsafe { OwnedHandle::from_raw_handle(raw) };
    // SAFETY: `process` is live and was opened with PROCESS_SYNCHRONIZE.
    #[expect(
        unsafe_code,
        reason = "Win32 FFI call to poll a process without waiting"
    )]
    let signaled = unsafe { WaitForSingleObject(process.as_raw_handle(), 0) } == WAIT_OBJECT_0;
    !signaled
}

#[cfg(all(test, windows))]
mod tests {
    use super::pid_running;

    #[test]
    fn a_running_process_is_running_and_an_exited_one_is_not() {
        assert!(pid_running(std::process::id()));

        let mut exited = caliche_policy::program_command("cmd")
            .args(["/C", "exit 0"])
            .spawn()
            .unwrap();
        let exited_pid = exited.id();
        exited.wait().unwrap();
        // `exited` still holds its handle, so the pid cannot be reused yet.
        assert!(!pid_running(exited_pid));
        drop(exited);
    }

    #[test]
    fn a_pid_with_no_process_is_not_running() {
        // Windows pids are multiples of four, so this one is never assigned.
        assert!(!pid_running(u32::MAX - 2));
    }
}

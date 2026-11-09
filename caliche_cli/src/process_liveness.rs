//! Whether a pid is still the process that was recorded.
//!
//! [`windows_process_match`] opens the process, checks that it has not exited,
//! and compares its creation time with the record.
//!
//! The comparisons themselves ([`compare_process_birth`], [`classify_open_error`],
//! [`classify_exit`]) do not call Win32, so they run on every platform.
//!
//! On Unix, [`signal_pid`] is the one place that calls `kill`.

/// Win32 `WAIT_OBJECT_0`: the process handle is signaled, so the process exited.
pub const WAIT_OBJECT_0: u32 = 0;
/// Win32 `WAIT_TIMEOUT`: the process was still running at the poll.
pub const WAIT_TIMEOUT: u32 = 258;
/// Win32 `STILL_ACTIVE`: `GetExitCodeProcess` value while the process is running.
pub const STILL_ACTIVE: u32 = 259;
/// Win32 `ERROR_INVALID_PARAMETER`: `OpenProcess` failed because the pid is gone.
pub const ERROR_INVALID_PARAMETER: i32 = 87;
/// Win32 `WAIT_FAILED`: the wait call itself failed.
pub const WAIT_FAILED: u32 = 0xFFFF_FFFF;
/// `MIB_TCP_STATE_LISTEN`.
pub const TCP_STATE_LISTEN: u32 = 2;

/// Shown when a live creation token is not the token stored on the record.
pub const REASON_TOKEN_MISMATCH: &str = "pid was reused; start token does not match the record";

/// A table size or row count overflowed.
const TCP_TABLE_TOO_LARGE: &str = "tcp table is too large";
/// The table ends before its declared rows.
const TCP_TABLE_TRUNCATED: &str = "tcp table is truncated";
/// A row ends before one of its fields.
const TCP_ROW_TRUNCATED: &str = "tcp row is truncated";

/// Shown when a record has no creation token to compare.
pub const REASON_EMPTY_TOKEN: &str = "record has no process start token";

/// Identity of a pid relative to a recorded birth.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessMatch {
    /// No process with this pid, or it has already exited.
    Dead,
    /// The pid is running and its creation time is the recorded one.
    AliveOriginal,
    /// A process is running, but it was created at a different time.
    Recycled,
    /// The pid could not be proven dead or original. Do not signal it.
    Unreadable { reason: String },
}

/// `OpenProcess` failure, before any creation-time check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenFailure {
    /// The pid does not exist.
    Dead,
    /// The process could not be queried. This is not proof that it is dead.
    Unreadable,
}

/// Result of `WaitForSingleObject` together with `GetExitCodeProcess`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitQuery {
    /// The process has exited.
    Exited,
    /// The process is still running.
    Running,
    /// The wait or exit code could not be read.
    Unreadable,
}

/// Compare a live creation token with the token stored on a record.
///
/// Only the tokens decide: an exact match is [`ProcessMatch::AliveOriginal`].
/// Wall-clock times are not compared, because a clock step-back would make
/// the original process look newer than its record.
#[must_use]
pub fn compare_process_birth(live_token: &str, record_token: &str) -> ProcessMatch {
    if record_token.is_empty() || live_token.is_empty() {
        return ProcessMatch::Unreadable {
            reason: REASON_EMPTY_TOKEN.to_string(),
        };
    }
    if live_token != record_token {
        return ProcessMatch::Recycled;
    }
    ProcessMatch::AliveOriginal
}

/// Join a Windows `FILETIME` high and low dword into 100-nanosecond ticks.
#[must_use]
pub fn filetime_ticks(high: u32, low: u32) -> u64 {
    (u64::from(high) << 32) | u64::from(low)
}

/// Classify an `OpenProcess` failure.
///
/// Error 87 means the pid does not exist. Access denied and every other
/// failure are [`OpenFailure::Unreadable`]: the caller must not treat them
/// as a dead pid and must not signal the process.
#[must_use]
pub fn classify_open_error(raw_os_error: Option<i32>) -> OpenFailure {
    match raw_os_error {
        Some(ERROR_INVALID_PARAMETER) => OpenFailure::Dead,
        _ => OpenFailure::Unreadable,
    }
}

/// Combine a wait result with an optional exit code.
///
/// A signaled wait means the process has exited, even when the exit code is
/// [`STILL_ACTIVE`] (a real process can exit with code 259). A timeout means
/// the process was still running; an exit code other than [`STILL_ACTIVE`]
/// on that path means it exited before the code was read.
#[must_use]
pub fn classify_exit(wait: u32, exit_code: Option<u32>) -> ExitQuery {
    if wait == WAIT_OBJECT_0 {
        return ExitQuery::Exited;
    }
    if wait != WAIT_TIMEOUT {
        return exit_code_or_unreadable(exit_code);
    }
    match exit_code {
        Some(STILL_ACTIVE) => ExitQuery::Running,
        Some(_) => ExitQuery::Exited,
        None => ExitQuery::Unreadable,
    }
}

fn exit_code_or_unreadable(exit_code: Option<u32>) -> ExitQuery {
    match exit_code {
        Some(code) if code != STILL_ACTIVE => ExitQuery::Exited,
        _ => ExitQuery::Unreadable,
    }
}

/// Error text for an unreadable exit query.
///
/// `wait_error` is `GetLastError` captured immediately after `WaitForSingleObject`.
/// `code_error` is `GetLastError` captured immediately after `GetExitCodeProcess`.
/// A later call must not replace the earlier failure.
#[must_use]
pub fn describe_exit_failure(
    wait: u32,
    got_code: bool,
    wait_error: &str,
    code_error: &str,
) -> String {
    let wait_failed = wait != WAIT_OBJECT_0 && wait != WAIT_TIMEOUT;
    match (wait_failed, got_code) {
        (true, false) => format!(
            "could not read process exit state: wait failed: {wait_error}; GetExitCodeProcess failed: {code_error}"
        ),
        (true, true) => format!("could not read process exit state: {wait_error}"),
        (false, false) => format!("could not read process exit state: {code_error}"),
        (false, true) => format!("could not read process exit state: wait={wait:#x}"),
    }
}

/// One row from a `GetExtendedTcpTable` owner-pid table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OwnerPidListener {
    /// `dwState`. Listeners are [`TCP_STATE_LISTEN`].
    pub state: u32,
    /// Host-order local port.
    pub local_port: u16,
    /// `dwOwningPid`.
    pub owning_pid: u32,
}

/// Parse an IPv4 `MIB_TCPTABLE_OWNER_PID` blob.
pub fn parse_ipv4_owner_pid_table(bytes: &[u8]) -> Result<Vec<OwnerPidListener>, &'static str> {
    parse_owner_pid_table(bytes, 24, 0, 8, 20)
}

/// Parse an IPv6 `MIB_TCP6TABLE_OWNER_PID` blob.
pub fn parse_ipv6_owner_pid_table(bytes: &[u8]) -> Result<Vec<OwnerPidListener>, &'static str> {
    parse_owner_pid_table(bytes, 56, 48, 20, 52)
}

/// Send `signal` to the one process `pid`. Signal 0 only checks that it exists.
///
/// Refuses pid 0 and 1, which would signal this process group or init. A pid
/// that does not fit in `pid_t` names no process, so it gets `ESRCH` without a
/// `kill` call; cast to `pid_t` it would name a process group or (as -1) every
/// process the user owns.
///
/// # Errors
///
/// `InvalidInput` for pid 0 or 1, `ESRCH` for a pid above `pid_t::MAX`;
/// otherwise the `kill` error, for example `ESRCH` (no such process) or
/// `EPERM` (another user's process).
#[cfg(unix)]
pub fn signal_pid(pid: u32, signal: i32) -> std::io::Result<()> {
    let Ok(target) = libc::pid_t::try_from(pid) else {
        return Err(std::io::Error::from_raw_os_error(libc::ESRCH));
    };
    if target < 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("refusing to signal pid {pid}"),
        ));
    }
    // SAFETY: kill takes no pointers. `target` is one positive pid of at
    // least 2, so it cannot name a process group, init, or every process.
    #[expect(
        unsafe_code,
        reason = "libc kill is the Unix call that signals or probes one pid"
    )]
    let rc = unsafe { libc::kill(target, signal) };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

/// True when `pid` has a listening row for `port`.
#[must_use]
pub fn listener_matches(rows: &[OwnerPidListener], pid: u32, port: u16) -> bool {
    rows.iter()
        .any(|row| row.owning_pid == pid && row.local_port == port && row.state == TCP_STATE_LISTEN)
}

/// Split a Windows command line using `CommandLineToArgvW` rules.
#[must_use]
pub fn split_windows_command_line(line: &str) -> Vec<String> {
    let chars: Vec<char> = line.chars().collect();
    let mut parser = WindowsCmd {
        chars: &chars,
        index: 0,
    };
    let mut args = Vec::new();
    while parser.skip_whitespace() {
        args.push(parser.parse_argument());
    }
    args
}

/// True when `arg` is one exact argument on a Windows command line.
#[must_use]
pub fn command_line_has_arg(line: &str, arg: &str) -> bool {
    split_windows_command_line(line)
        .iter()
        .any(|item| item == arg)
}

/// Decode a `ProcessCommandLineInformation` buffer.
///
/// `pointer_width` is 4 or 8. `buffer_address` is the allocation address: the
/// kernel writes `UNICODE_STRING.Buffer` as an absolute pointer into that
/// allocation. A pointer outside it is `None`, never a guess.
#[must_use]
pub fn decode_command_line_buffer(
    buffer: &[u8],
    pointer_width: usize,
    buffer_address: usize,
) -> Option<String> {
    let header = unicode_string_header(pointer_width)?;
    if buffer.len() < header {
        return None;
    }
    let length = u16::from_le_bytes([buffer[0], buffer[1]]) as usize;
    if length == 0 || length % 2 != 0 {
        return None;
    }
    let pointer = read_pointer(buffer, pointer_width)?;
    let start = command_line_offset(buffer.len(), header, pointer, length, buffer_address)?;
    let end = start.checked_add(length)?;
    decode_utf16(buffer.get(start..end)?)
}

fn parse_owner_pid_table(
    bytes: &[u8],
    row_len: usize,
    state_off: usize,
    port_off: usize,
    pid_off: usize,
) -> Result<Vec<OwnerPidListener>, &'static str> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    let count = read_u32(bytes, 0).ok_or("tcp table is shorter than its header")? as usize;
    let rows_bytes = count.checked_mul(row_len).ok_or(TCP_TABLE_TOO_LARGE)?;
    let need = 4usize.checked_add(rows_bytes).ok_or(TCP_TABLE_TOO_LARGE)?;
    if bytes.len() < need {
        return Err(TCP_TABLE_TRUNCATED);
    }
    let mut rows = Vec::with_capacity(count);
    for index in 0..count {
        let base = 4 + index * row_len;
        rows.push(OwnerPidListener {
            state: read_u32(bytes, base + state_off).ok_or(TCP_ROW_TRUNCATED)?,
            local_port: u16::from_be(
                read_u32(bytes, base + port_off).ok_or(TCP_ROW_TRUNCATED)? as u16
            ),
            owning_pid: read_u32(bytes, base + pid_off).ok_or(TCP_ROW_TRUNCATED)?,
        });
    }
    Ok(rows)
}

fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let end = offset.checked_add(4)?;
    let chunk = bytes.get(offset..end)?;
    let raw: [u8; 4] = chunk.try_into().ok()?;
    Some(u32::from_le_bytes(raw))
}

struct WindowsCmd<'a> {
    chars: &'a [char],
    index: usize,
}

impl WindowsCmd<'_> {
    fn skip_whitespace(&mut self) -> bool {
        while self.index < self.chars.len() && self.chars[self.index].is_whitespace() {
            self.index += 1;
        }
        self.index < self.chars.len()
    }

    fn parse_argument(&mut self) -> String {
        let mut out = String::new();
        let mut quoted = false;
        while self.index < self.chars.len() {
            if !quoted && self.chars[self.index].is_whitespace() {
                break;
            }
            self.consume(&mut out, &mut quoted);
        }
        out
    }

    fn consume(&mut self, out: &mut String, quoted: &mut bool) {
        match self.chars[self.index] {
            '\\' => self.consume_backslashes(out, quoted),
            '"' => self.consume_quote(out, quoted),
            ch => {
                out.push(ch);
                self.index += 1;
            }
        }
    }

    fn consume_backslashes(&mut self, out: &mut String, quoted: &mut bool) {
        let slashes = self.count_char('\\');
        if self.peek() == Some('"') {
            push_repeat(out, '\\', slashes / 2);
            if slashes % 2 == 1 {
                out.push('"');
                self.index += 1;
            } else {
                self.consume_quote(out, quoted);
            }
            return;
        }
        push_repeat(out, '\\', slashes);
    }

    fn consume_quote(&mut self, out: &mut String, quoted: &mut bool) {
        if *quoted && self.chars.get(self.index + 1) == Some(&'"') {
            out.push('"');
            self.index += 2;
            return;
        }
        *quoted = !*quoted;
        self.index += 1;
    }

    fn count_char(&mut self, expected: char) -> usize {
        let start = self.index;
        while self.index < self.chars.len() && self.chars[self.index] == expected {
            self.index += 1;
        }
        self.index - start
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.index).copied()
    }
}

fn push_repeat(out: &mut String, ch: char, count: usize) {
    for _ in 0..count {
        out.push(ch);
    }
}

fn unicode_string_header(pointer_width: usize) -> Option<usize> {
    match pointer_width {
        8 => Some(16),
        4 => Some(8),
        _ => None,
    }
}

fn read_pointer(buffer: &[u8], pointer_width: usize) -> Option<usize> {
    match pointer_width {
        8 => {
            let raw: [u8; 8] = buffer.get(8..16)?.try_into().ok()?;
            usize::try_from(u64::from_le_bytes(raw)).ok()
        }
        4 => {
            let raw: [u8; 4] = buffer.get(4..8)?.try_into().ok()?;
            Some(u32::from_le_bytes(raw) as usize)
        }
        _ => None,
    }
}

fn command_line_offset(
    buffer_len: usize,
    header: usize,
    pointer: usize,
    length: usize,
    buffer_address: usize,
) -> Option<usize> {
    let offset = pointer.checked_sub(buffer_address)?;
    (offset >= header && offset.checked_add(length)? <= buffer_len).then_some(offset)
}

fn decode_utf16(bytes: &[u8]) -> Option<String> {
    let mut units = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        units.push(u16::from_le_bytes([chunk[0], chunk[1]]));
    }
    Some(String::from_utf16_lossy(&units))
}

/// Windows liveness and creation-time check for one recorded pid.
///
/// Opens the process, treats an exited or missing pid as [`ProcessMatch::Dead`],
/// and compares the `GetProcessTimes` creation time with `record_token`. A pid
/// that cannot be queried is [`ProcessMatch::Unreadable`].
#[cfg(target_os = "windows")]
#[must_use]
pub fn windows_process_match(pid: u32, record_token: &str) -> ProcessMatch {
    let handle = match open_process(pid, query_access()) {
        Ok(handle) => handle,
        Err(OpenError::Dead) => return ProcessMatch::Dead,
        Err(OpenError::Unreadable(reason)) => return ProcessMatch::Unreadable { reason },
    };
    match has_exited(handle.raw()) {
        Ok(true) => return ProcessMatch::Dead,
        Ok(false) => {}
        Err(reason) => return ProcessMatch::Unreadable { reason },
    }
    match creation_ticks(handle.raw()) {
        Ok(ticks) => compare_process_birth(&ticks.to_string(), record_token),
        Err(reason) => ProcessMatch::Unreadable { reason },
    }
}

/// True unless `pid` is proven gone: a pid that cannot be opened or polled
/// for access reasons is still running as far as a lock holder knows.
#[cfg(target_os = "windows")]
#[must_use]
pub fn windows_pid_running(pid: u32) -> bool {
    match open_process(pid, query_access()) {
        Ok(handle) => !matches!(has_exited(handle.raw()), Ok(true)),
        Err(OpenError::Dead) => false,
        Err(OpenError::Unreadable(_)) => true,
    }
}

/// Creation `FILETIME` of `pid`, as a decimal token.
#[cfg(target_os = "windows")]
pub fn creation_token(pid: u32) -> Result<String, String> {
    let handle = open_named_handle(pid, query_access())?;
    let ticks = creation_ticks(handle.raw())?;
    Ok(ticks.to_string())
}

/// Win32 path of `pid`, from `QueryFullProcessImageNameW`.
#[cfg(target_os = "windows")]
pub fn process_image_name(pid: u32) -> Result<String, String> {
    let handle = open_named_handle(pid, query_access())?;
    query_image_name(handle.raw())
}

/// True when `pid`'s command line has `argument` as one exact argument.
#[cfg(target_os = "windows")]
pub fn process_has_argument(pid: u32, argument: &str) -> Result<bool, String> {
    let line = process_command_line(pid)?;
    Ok(command_line_has_arg(&line, argument))
}

/// Command line of `pid`, from `ProcessCommandLineInformation`.
#[cfg(target_os = "windows")]
pub fn process_command_line(pid: u32) -> Result<String, String> {
    let handle = open_named_handle(pid, command_access())?;
    read_process_command_line(handle.raw())
}

/// True when `pid` is listening on `port` in the IPv4 or IPv6 TCP table.
#[cfg(target_os = "windows")]
pub fn process_listens_on(pid: u32, port: u16) -> Result<bool, String> {
    Ok(listener_matches(&tcp_owner_rows()?, pid, port))
}

/// TCP ports in the listen state, from the IPv4 and IPv6 owner-pid tables.
///
/// A port can appear once per family. Callers that compare two snapshots
/// should put the ports in a set.
#[cfg(target_os = "windows")]
pub fn current_listen_ports() -> Result<Vec<u16>, String> {
    Ok(tcp_owner_rows()?
        .into_iter()
        .filter(|row| row.state == TCP_STATE_LISTEN)
        .map(|row| row.local_port)
        .collect())
}

#[cfg(target_os = "windows")]
fn tcp_owner_rows() -> Result<Vec<OwnerPidListener>, String> {
    let ipv4 = extended_tcp_table(AF_INET)?;
    let ipv6 = extended_tcp_table(AF_INET6)?;
    let mut rows = parse_ipv4_owner_pid_table(&ipv4).map_err(str::to_string)?;
    rows.extend(parse_ipv6_owner_pid_table(&ipv6).map_err(str::to_string)?);
    Ok(rows)
}

/// `TerminateProcess` when `pid` is still the process born at `record_token`.
///
/// The handle is opened again for terminate rights and the creation time is
/// read before the call. A token mismatch returns an error and does not
/// terminate. A pid that is already gone is success.
#[cfg(target_os = "windows")]
pub fn terminate_if_same_birth(pid: u32, record_token: &str) -> Result<(), String> {
    if record_token.is_empty() {
        return Err(REASON_EMPTY_TOKEN.to_string());
    }
    let Some(handle) = open_terminate_handle(pid)? else {
        return Ok(());
    };
    if has_exited(handle.raw())? {
        return Ok(());
    }
    let ticks = creation_ticks(handle.raw())?;
    if ticks.to_string() != record_token {
        return Err(REASON_TOKEN_MISMATCH.to_string());
    }
    terminate_handle(handle.raw())
}

#[cfg(target_os = "windows")]
struct ProcessHandle(windows_sys::Win32::Foundation::HANDLE);

#[cfg(target_os = "windows")]
impl ProcessHandle {
    fn raw(&self) -> windows_sys::Win32::Foundation::HANDLE {
        self.0
    }
}

#[cfg(target_os = "windows")]
impl Drop for ProcessHandle {
    fn drop(&mut self) {
        let handle = self.0;
        self.0 = std::ptr::null_mut();
        if handle.is_null() {
            return;
        }
        // SAFETY: `handle` was returned by OpenProcess and has not been closed.
        #[expect(unsafe_code, reason = "CloseHandle releases the OpenProcess handle")]
        unsafe {
            let _ = windows_sys::Win32::Foundation::CloseHandle(handle);
        }
    }
}

#[cfg(target_os = "windows")]
fn has_exited(handle: windows_sys::Win32::Foundation::HANDLE) -> Result<bool, String> {
    use windows_sys::Win32::System::Threading::{GetExitCodeProcess, WaitForSingleObject};
    // SAFETY: `handle` is an open process handle. A zero timeout does not block.
    #[expect(unsafe_code, reason = "Win32 WaitForSingleObject polls process exit")]
    let wait = unsafe { WaitForSingleObject(handle, 0) };
    let wait_error = std::io::Error::last_os_error();
    let mut code = 0_u32;
    // SAFETY: `code` is a writable u32 and `handle` is still open.
    #[expect(unsafe_code, reason = "Win32 GetExitCodeProcess reads the exit code")]
    let got_code = unsafe { GetExitCodeProcess(handle, &raw mut code) } != 0;
    let code_error = std::io::Error::last_os_error();
    let exit_code = got_code.then_some(code);
    match classify_exit(wait, exit_code) {
        ExitQuery::Exited => Ok(true),
        ExitQuery::Running => Ok(false),
        ExitQuery::Unreadable => Err(describe_exit_failure(
            wait,
            got_code,
            &wait_error.to_string(),
            &code_error.to_string(),
        )),
    }
}

#[cfg(target_os = "windows")]
fn creation_ticks(handle: windows_sys::Win32::Foundation::HANDLE) -> Result<u64, String> {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::GetProcessTimes;
    let mut created = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    let mut exited = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    let mut kernel = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    let mut user = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    // SAFETY: the four FILETIME outputs are writable locals and `handle` is open.
    #[expect(
        unsafe_code,
        reason = "Win32 GetProcessTimes reads the creation FILETIME"
    )]
    let ok = unsafe {
        GetProcessTimes(
            handle,
            &raw mut created,
            &raw mut exited,
            &raw mut kernel,
            &raw mut user,
        )
    };
    if ok == 0 {
        return Err(format!(
            "GetProcessTimes failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(filetime_from_parts(created))
}

#[cfg(target_os = "windows")]
fn filetime_from_parts(value: windows_sys::Win32::Foundation::FILETIME) -> u64 {
    filetime_ticks(value.dwHighDateTime, value.dwLowDateTime)
}

#[cfg(target_os = "windows")]
const AF_INET: u32 = 2;
#[cfg(target_os = "windows")]
const AF_INET6: u32 = 23;
#[cfg(target_os = "windows")]
const MAX_COMMAND_LINE_BYTES: u32 = 1 << 20;

#[cfg(target_os = "windows")]
fn query_access() -> u32 {
    use windows_sys::Win32::System::Threading::{
        PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE,
    };
    PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_SYNCHRONIZE
}

#[cfg(target_os = "windows")]
fn command_access() -> u32 {
    use windows_sys::Win32::System::Threading::{
        PROCESS_QUERY_INFORMATION, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_VM_READ,
    };
    PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_QUERY_INFORMATION | PROCESS_VM_READ
}

/// Why `OpenProcess` returned no handle.
#[cfg(target_os = "windows")]
enum OpenError {
    Dead,
    Unreadable(String),
}

#[cfg(target_os = "windows")]
fn open_named_handle(pid: u32, access: u32) -> Result<ProcessHandle, String> {
    open_process(pid, access).map_err(|err| match err {
        OpenError::Dead => format!("pid {pid} is not running"),
        OpenError::Unreadable(reason) => reason,
    })
}

#[cfg(target_os = "windows")]
fn open_process(pid: u32, access: u32) -> Result<ProcessHandle, OpenError> {
    use windows_sys::Win32::System::Threading::OpenProcess;
    // SAFETY: OpenProcess takes no pointers. A null return is a failed open.
    #[expect(unsafe_code, reason = "Win32 OpenProcess is the pid probe")]
    let raw = unsafe { OpenProcess(access, 0, pid) };
    if raw.is_null() {
        let err = std::io::Error::last_os_error();
        return Err(match classify_open_error(err.raw_os_error()) {
            OpenFailure::Dead => OpenError::Dead,
            OpenFailure::Unreadable => {
                OpenError::Unreadable(format!("could not open pid {pid}: {err}"))
            }
        });
    }
    Ok(ProcessHandle(raw))
}

#[cfg(target_os = "windows")]
fn query_image_name(handle: windows_sys::Win32::Foundation::HANDLE) -> Result<String, String> {
    use windows_sys::Win32::Foundation::ERROR_INSUFFICIENT_BUFFER;
    use windows_sys::Win32::System::Threading::QueryFullProcessImageNameW;
    let mut capacity = 512_u32;
    for _ in 0..6 {
        let mut buffer = vec![0_u16; capacity as usize];
        let mut length = capacity;
        // SAFETY: `buffer` is writable for `length` wide characters and `handle` is open.
        #[expect(
            unsafe_code,
            reason = "Win32 QueryFullProcessImageNameW reads the process image path"
        )]
        let ok = unsafe { QueryFullProcessImageNameW(handle, 0, buffer.as_mut_ptr(), &mut length) };
        if ok != 0 {
            let count = (length as usize).min(buffer.len());
            return Ok(String::from_utf16_lossy(&buffer[..count]));
        }
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() != i32::try_from(ERROR_INSUFFICIENT_BUFFER).ok() {
            return Err(format!("QueryFullProcessImageNameW failed: {err}"));
        }
        capacity = capacity.saturating_mul(2).max(1024);
    }
    Err("QueryFullProcessImageNameW failed: path exceeded the buffer".to_string())
}

#[cfg(target_os = "windows")]
fn read_process_command_line(
    handle: windows_sys::Win32::Foundation::HANDLE,
) -> Result<String, String> {
    let mut capacity = 512_u32;
    for _ in 0..8 {
        match query_command_line(handle, capacity)? {
            CommandLineRead::Line(line) => return Ok(line),
            CommandLineRead::Grow(next) if next > capacity && next <= MAX_COMMAND_LINE_BYTES => {
                capacity = next;
            }
            CommandLineRead::Grow(_) => break,
        }
    }
    Err("process command line exceeded the buffer".to_string())
}

#[cfg(target_os = "windows")]
enum CommandLineRead {
    Line(String),
    Grow(u32),
}

#[cfg(target_os = "windows")]
fn query_command_line(
    handle: windows_sys::Win32::Foundation::HANDLE,
    capacity: u32,
) -> Result<CommandLineRead, String> {
    use windows_sys::Wdk::System::Threading::{
        NtQueryInformationProcess, ProcessCommandLineInformation,
    };
    // u64 words keep the UNICODE_STRING header pointer-aligned.
    let mut words = vec![0_u64; (capacity as usize).div_ceil(8)];
    let mut written = 0_u32;
    // SAFETY: `words` is writable for at least `capacity` bytes and `handle` is open.
    #[expect(
        unsafe_code,
        reason = "NtQueryInformationProcess reads the process command line"
    )]
    let status = unsafe {
        NtQueryInformationProcess(
            handle,
            ProcessCommandLineInformation,
            words.as_mut_ptr().cast(),
            capacity,
            &mut written,
        )
    };
    if status < 0 {
        if command_line_needs_room(status) {
            let next = if written > capacity {
                written
            } else {
                capacity.saturating_mul(2)
            };
            return Ok(CommandLineRead::Grow(next));
        }
        return Err(format!("NtQueryInformationProcess failed: {status:#x}"));
    }
    let address = words.as_ptr() as usize;
    let buffer = words_as_bytes(&words);
    decode_command_line_buffer(&buffer, size_of::<usize>(), address)
        .map(CommandLineRead::Line)
        .ok_or_else(|| "process command line was empty".to_string())
}

/// The bytes of `words` in memory order.
#[cfg(target_os = "windows")]
fn words_as_bytes(words: &[u64]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_ne_bytes()).collect()
}

#[cfg(target_os = "windows")]
fn command_line_needs_room(status: i32) -> bool {
    const INFO_LENGTH_MISMATCH: i32 = 0xC000_0004_u32 as i32;
    const BUFFER_TOO_SMALL: i32 = 0xC000_0023_u32 as i32;
    const BUFFER_OVERFLOW: i32 = 0x8000_0005_u32 as i32;
    matches!(
        status,
        INFO_LENGTH_MISMATCH | BUFFER_TOO_SMALL | BUFFER_OVERFLOW
    )
}

#[cfg(target_os = "windows")]
fn extended_tcp_table(family: u32) -> Result<Vec<u8>, String> {
    use windows_sys::Win32::Foundation::ERROR_INSUFFICIENT_BUFFER;
    // u64 words keep the DWORD rows aligned.
    let mut words: Vec<u64> = Vec::new();
    for _ in 0..4 {
        let (status, needed) = query_tcp_table(family, &mut words)?;
        if status == 0 {
            let mut buffer = words_as_bytes(&words);
            buffer.truncate((needed as usize).min(buffer.len()));
            return Ok(buffer);
        }
        if status != ERROR_INSUFFICIENT_BUFFER {
            return Err(format!("GetExtendedTcpTable failed: {status}"));
        }
        let next = usize::try_from(needed).map_err(|_| TCP_TABLE_TOO_LARGE.to_string())?;
        words.resize(next.div_ceil(8), 0);
    }
    Err("GetExtendedTcpTable failed: table size kept changing".to_string())
}

#[cfg(target_os = "windows")]
fn query_tcp_table(family: u32, words: &mut [u64]) -> Result<(u32, u32), String> {
    use windows_sys::Win32::NetworkManagement::IpHelper::{
        GetExtendedTcpTable, TCP_TABLE_OWNER_PID_LISTENER,
    };
    let mut size = words
        .len()
        .checked_mul(8)
        .and_then(|bytes| u32::try_from(bytes).ok())
        .ok_or_else(|| TCP_TABLE_TOO_LARGE.to_string())?;
    let ptr = if words.is_empty() {
        std::ptr::null_mut()
    } else {
        words.as_mut_ptr().cast()
    };
    // SAFETY: `ptr` is null or writable for `size` bytes. The call writes the needed size.
    #[expect(
        unsafe_code,
        reason = "GetExtendedTcpTable reads the TCP listener table"
    )]
    let status = unsafe {
        GetExtendedTcpTable(
            ptr,
            &raw mut size,
            0,
            family,
            TCP_TABLE_OWNER_PID_LISTENER,
            0,
        )
    };
    Ok((status, size))
}

#[cfg(target_os = "windows")]
fn open_terminate_handle(pid: u32) -> Result<Option<ProcessHandle>, String> {
    use windows_sys::Win32::System::Threading::PROCESS_TERMINATE;
    match open_process(pid, query_access() | PROCESS_TERMINATE) {
        Ok(handle) => Ok(Some(handle)),
        Err(OpenError::Dead) => Ok(None),
        Err(OpenError::Unreadable(reason)) => Err(reason),
    }
}

#[cfg(target_os = "windows")]
fn terminate_handle(handle: windows_sys::Win32::Foundation::HANDLE) -> Result<(), String> {
    use windows_sys::Win32::System::Threading::TerminateProcess;
    // SAFETY: `handle` was opened with PROCESS_TERMINATE and re-checked against the record.
    #[expect(unsafe_code, reason = "TerminateProcess stops a verified server pid")]
    let ok = unsafe { TerminateProcess(handle, 1) };
    if ok != 0 {
        return Ok(());
    }
    let err = std::io::Error::last_os_error();
    if has_exited(handle).unwrap_or(false) {
        return Ok(());
    }
    Err(format!("TerminateProcess failed: {err}"))
}

#[cfg(test)]
mod tests {
    use super::{
        ERROR_INVALID_PARAMETER, ExitQuery, OpenFailure, ProcessMatch, STILL_ACTIVE, WAIT_FAILED,
        WAIT_OBJECT_0, WAIT_TIMEOUT, classify_exit, classify_open_error, compare_process_birth,
        filetime_ticks,
    };

    #[test]
    fn matching_token_is_the_original() {
        assert_eq!(
            compare_process_birth("100", "100"),
            ProcessMatch::AliveOriginal
        );
    }

    #[test]
    fn token_mismatch_is_recycled() {
        assert_eq!(compare_process_birth("200", "100"), ProcessMatch::Recycled);
    }

    #[test]
    fn empty_token_is_unreadable() {
        let expected = ProcessMatch::Unreadable {
            reason: super::REASON_EMPTY_TOKEN.to_string(),
        };
        assert_eq!(compare_process_birth("", "token"), expected);
        assert_eq!(compare_process_birth("token", ""), expected);
    }

    #[test]
    fn open_invalid_parameter_is_dead_and_access_denied_is_not() {
        assert_eq!(
            classify_open_error(Some(ERROR_INVALID_PARAMETER)),
            OpenFailure::Dead
        );
        assert_eq!(classify_open_error(Some(5)), OpenFailure::Unreadable);
        assert_eq!(classify_open_error(None), OpenFailure::Unreadable);
    }

    #[test]
    fn signaled_wait_is_exited_even_with_still_active_code() {
        assert_eq!(
            classify_exit(WAIT_OBJECT_0, Some(STILL_ACTIVE)),
            ExitQuery::Exited
        );
        assert_eq!(classify_exit(WAIT_OBJECT_0, Some(0)), ExitQuery::Exited);
    }

    #[test]
    fn timeout_with_still_active_is_running() {
        assert_eq!(
            classify_exit(WAIT_TIMEOUT, Some(STILL_ACTIVE)),
            ExitQuery::Running
        );
    }

    #[test]
    fn timeout_with_a_real_exit_code_is_exited() {
        assert_eq!(classify_exit(WAIT_TIMEOUT, Some(0)), ExitQuery::Exited);
    }

    #[test]
    fn missing_exit_code_on_timeout_is_unreadable() {
        assert_eq!(classify_exit(WAIT_TIMEOUT, None), ExitQuery::Unreadable);
        assert_eq!(classify_exit(0xFFFF_FFFF, None), ExitQuery::Unreadable);
    }

    #[test]
    fn exit_failure_names_the_call_that_failed() {
        let describe =
            |wait, got_code| super::describe_exit_failure(wait, got_code, "wait-err", "code-err");
        assert_eq!(
            describe(WAIT_FAILED, false),
            "could not read process exit state: wait failed: wait-err; GetExitCodeProcess failed: code-err"
        );
        assert_eq!(
            describe(WAIT_FAILED, true),
            "could not read process exit state: wait-err"
        );
        assert_eq!(
            describe(WAIT_TIMEOUT, false),
            "could not read process exit state: code-err"
        );
        assert_eq!(
            describe(WAIT_TIMEOUT, true),
            "could not read process exit state: wait=0x102"
        );
    }

    #[test]
    fn windows_command_line_keeps_an_exact_server_flag() {
        let line = r#""C:\Program Files\caliche.exe" --server --bind 127.0.0.1"#;
        assert!(super::command_line_has_arg(line, "--server"));
        let quoted = r#"caliche.exe "--server""#;
        assert!(super::command_line_has_arg(quoted, "--server"));
        let inside = r#"caliche.exe --bind "has --server inside""#;
        assert!(!super::command_line_has_arg(inside, "--server"));
        let assigned = "caliche.exe --server=1";
        assert!(!super::command_line_has_arg(assigned, "--server"));
        assert_eq!(
            super::split_windows_command_line(r"caliche.exe --server"),
            vec!["caliche.exe".to_string(), "--server".to_string()]
        );
    }

    #[test]
    fn ipv4_listener_table_decodes_network_order_ports() {
        let port = 3000_u16;
        let mut row = [0_u8; 24];
        row[0..4].copy_from_slice(&super::TCP_STATE_LISTEN.to_le_bytes());
        row[8..12].copy_from_slice(&u32::from(port.to_be()).to_le_bytes());
        row[20..24].copy_from_slice(&42_u32.to_le_bytes());
        let mut table = 1_u32.to_le_bytes().to_vec();
        table.extend(row);
        let rows = super::parse_ipv4_owner_pid_table(&table).expect("parse");
        assert!(super::listener_matches(&rows, 42, 3000));
        assert!(!super::listener_matches(&rows, 42, 3001));
        assert!(!super::listener_matches(&rows, 7, 3000));
        assert!(super::parse_ipv4_owner_pid_table(&[]).unwrap().is_empty());
        assert!(super::parse_ipv4_owner_pid_table(&[1, 0, 0, 0]).is_err());
    }

    #[test]
    fn ipv6_listener_table_reads_port_and_pid() {
        let port = 8080_u16;
        let mut row = [0_u8; 56];
        row[20..24].copy_from_slice(&u32::from(port.to_be()).to_le_bytes());
        row[48..52].copy_from_slice(&super::TCP_STATE_LISTEN.to_le_bytes());
        row[52..56].copy_from_slice(&9_u32.to_le_bytes());
        let mut table = 1_u32.to_le_bytes().to_vec();
        table.extend(row);
        let rows = super::parse_ipv6_owner_pid_table(&table).expect("parse");
        assert!(super::listener_matches(&rows, 9, 8080));
    }

    #[test]
    fn command_line_buffer_needs_a_pointer_into_the_buffer() {
        let text: Vec<u16> = "caliche.exe --server".encode_utf16().collect();
        let mut bytes = Vec::new();
        for unit in &text {
            bytes.extend(unit.to_le_bytes());
        }
        let mut buffer = vec![0_u8; 16];
        buffer.extend_from_slice(&bytes);
        let length = u16::try_from(bytes.len()).unwrap_or(0);
        buffer[0..2].copy_from_slice(&length.to_le_bytes());
        let base = 0x1000_u64;
        buffer[8..16].copy_from_slice(&(base + 16).to_le_bytes());
        let decoded =
            super::decode_command_line_buffer(&buffer, 8, base as usize).expect("absolute");
        assert!(super::command_line_has_arg(&decoded, "--server"));
        assert_eq!(decoded, "caliche.exe --server");
        // A bare offset, or a pointer past the allocation, is not decoded.
        buffer[8..16].copy_from_slice(&16_u64.to_le_bytes());
        assert_eq!(
            super::decode_command_line_buffer(&buffer, 8, base as usize),
            None
        );
        buffer[8..16].copy_from_slice(&(base + 0x1000).to_le_bytes());
        assert_eq!(
            super::decode_command_line_buffer(&buffer, 8, base as usize),
            None
        );
        buffer[8..16].copy_from_slice(&(base + 8).to_le_bytes());
        assert_eq!(
            super::decode_command_line_buffer(&buffer, 8, base as usize),
            None
        );
    }

    #[test]
    fn filetime_joins_high_and_low_dwords() {
        assert_eq!(filetime_ticks(0, 1), 1);
        assert_eq!(filetime_ticks(1, 0), 1_u64 << 32);
        assert_eq!(
            filetime_ticks(0x8000_0000, 0x8000_0000),
            0x8000_0000_8000_0000
        );
        assert_eq!(
            filetime_ticks(0x0123_4567, 0x89AB_CDEF),
            0x0123_4567_89AB_CDEF
        );
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn win32_constants_match_this_module() {
        assert_eq!(WAIT_OBJECT_0, windows_sys::Win32::Foundation::WAIT_OBJECT_0);
        assert_eq!(WAIT_TIMEOUT, windows_sys::Win32::Foundation::WAIT_TIMEOUT);
        // windows-sys types STILL_ACTIVE as NTSTATUS (i32). GetExitCodeProcess
        // returns the same 259 as a DWORD.
        assert_eq!(
            i64::from(STILL_ACTIVE),
            i64::from(windows_sys::Win32::Foundation::STILL_ACTIVE)
        );
        assert_eq!(
            ERROR_INVALID_PARAMETER,
            i32::try_from(windows_sys::Win32::Foundation::ERROR_INVALID_PARAMETER).unwrap_or(-1)
        );
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn dead_pid_is_not_the_original() {
        let matched = super::windows_process_match(u32::MAX, "1");
        assert_eq!(matched, ProcessMatch::Dead);
        assert!(!super::windows_pid_running(u32::MAX));
        assert!(super::windows_pid_running(std::process::id()));
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn this_process_matches_its_creation_token() {
        let token = super::creation_token(std::process::id()).expect("creation token");
        let matched = super::windows_process_match(std::process::id(), &token);
        assert_eq!(matched, ProcessMatch::AliveOriginal);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn this_process_with_another_token_is_recycled() {
        let matched = super::windows_process_match(std::process::id(), "1");
        assert_eq!(matched, ProcessMatch::Recycled);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn this_process_image_name_is_an_exe() {
        let name = super::process_image_name(std::process::id()).expect("image");
        assert!(name.to_ascii_lowercase().ends_with(".exe"), "image {name}");
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn this_process_command_line_is_readable() {
        let line = super::process_command_line(std::process::id()).expect("command line");
        assert!(line.to_ascii_lowercase().contains(".exe"), "{line}");
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn this_process_listener_is_visible_in_the_tcp_table() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        assert!(super::process_listens_on(std::process::id(), port).expect("table"));
        assert!(!super::process_listens_on(std::process::id(), 0).expect("other"));
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn terminate_requires_the_creation_token() {
        let mut child = std::process::Command::new("cmd")
            .args(["/c", "ping", "-n", "30", "127.0.0.1"])
            .spawn()
            .expect("spawn");
        let pid = child.id();
        let token = super::creation_token(pid).expect("token");
        let wrong = super::terminate_if_same_birth(pid, "not-the-token");
        assert!(wrong.is_err(), "wrong token must not terminate");
        assert!(child.try_wait().expect("poll").is_none());
        super::terminate_if_same_birth(pid, &token).expect("terminate");
        let _ = child.wait();
    }
}

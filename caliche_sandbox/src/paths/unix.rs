//! Open a file below an allowed root without following any symlink.
//!
//! `O_NOFOLLOW` only covers the last path component, so a directory replaced
//! by a symlink after validation would still be followed by an open of the
//! full path. Here the walk starts at `/` and opens every component of the
//! canonical target, the root's own components included, relative to its
//! parent's descriptor with `openat` and `O_NOFOLLOW`. No component is looked
//! up by a full path string after validation, so a swapped directory, at the
//! root, above it or below it, makes the open fail instead of leaving the root.

use super::{DirEntryBeneath, EntryKind};
use std::ffi::{CStr, CString, OsStr, OsString};
use std::fs::File;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Component, Path};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Access {
    /// Open an existing file for reading.
    Read,
    /// Create missing parent directories, then open or create the file and
    /// truncate it.
    Write,
}

const DIR_FLAGS: libc::c_int =
    libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC;
// `O_NONBLOCK` keeps a FIFO from blocking the open; `require_regular_file`
// then refuses it and clears the flag on a regular file. Writes open without
// `O_TRUNC` and truncate only after the file passed those checks.
const READ_FLAGS: libc::c_int =
    libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK;
const WRITE_FLAGS: libc::c_int =
    libc::O_WRONLY | libc::O_CREAT | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK;

/// Open `target`, a canonical absolute path strictly below the canonical `root`.
///
/// No component of `target`, from `/` down, may be a symlink, so the root
/// cannot be swapped for one either. Writes create missing directories on the
/// way, including a configured root that does not exist yet. The target must
/// be a regular file, and a write target must have a single link.
pub(super) fn open_beneath(root: &Path, target: &Path, access: Access) -> io::Result<File> {
    if components_below(root, target)?.is_empty() {
        return Err(invalid_input(format!(
            "'{}' names the sandbox root, not a file",
            target.display()
        )));
    }
    let names = absolute_components(target)?;
    let Some((file_name, dir_names)) = names.split_last() else {
        return Err(invalid_input(format!(
            "'{}' has no file name",
            target.display()
        )));
    };
    let mut dir: OwnedFd = File::options()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC)
        .open("/")?
        .into();
    for name in dir_names {
        dir = open_dir(&dir, name, access, target)?;
    }
    let name = c_name(file_name)?;
    let file = match access {
        Access::Read => openat(&dir, &name, READ_FLAGS),
        Access::Write => openat(&dir, &name, WRITE_FLAGS),
    }
    .map_err(|err| describe(err, &dir, file_name, target))?;
    let file = require_regular_file(File::from(file), target)?;
    if access == Access::Write {
        refuse_hard_links(&file, target)?;
        file.set_len(0)?;
    }
    Ok(file)
}

/// List the directory `target`, the canonical `root` or a path below it.
///
/// Every component from `/` down is opened with `O_NOFOLLOW` relative to its
/// parent, as in [`open_beneath`]; the entries are then read from that
/// descriptor and described with `fstatat(AT_SYMLINK_NOFOLLOW)`.
pub(super) fn list_dir_beneath(root: &Path, target: &Path) -> io::Result<Vec<DirEntryBeneath>> {
    components_below(root, target)?;
    let mut dir: OwnedFd = File::options()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC)
        .open("/")?
        .into();
    for name in absolute_components(target)? {
        dir = open_dir(&dir, name, Access::Read, target)?;
    }
    read_entries(dir)
}

/// A directory stream from `fdopendir`, closed on drop.
struct DirStream(std::ptr::NonNull<libc::DIR>);

impl Drop for DirStream {
    fn drop(&mut self) {
        // SAFETY: the stream came from `fdopendir` and is closed only here.
        #[expect(unsafe_code, reason = "closedir on the stream this value owns")]
        unsafe {
            libc::closedir(self.0.as_ptr());
        }
    }
}

/// Read every entry of the open directory `dir`, except `.` and `..`.
fn read_entries(dir: OwnedFd) -> io::Result<Vec<DirEntryBeneath>> {
    let fd = dir.into_raw_fd();
    // SAFETY: `fd` is an open directory descriptor that nothing else owns.
    // On success the stream owns it; on failure it is still ours.
    #[expect(unsafe_code, reason = "fdopendir on an owned directory descriptor")]
    let stream = unsafe { libc::fdopendir(fd) };
    let Some(stream) = std::ptr::NonNull::new(stream) else {
        let err = io::Error::last_os_error();
        // SAFETY: `fdopendir` failed, so `fd` is still open and owned here.
        #[expect(unsafe_code, reason = "close the descriptor fdopendir did not take")]
        drop(unsafe { OwnedFd::from_raw_fd(fd) });
        return Err(err);
    };
    let stream = DirStream(stream);
    // SAFETY: `stream` is an open directory stream.
    #[expect(unsafe_code, reason = "dirfd of an open directory stream")]
    let dir_fd = unsafe { libc::dirfd(stream.0.as_ptr()) };
    let mut entries = Vec::new();
    loop {
        clear_errno();
        // SAFETY: `stream` is an open directory stream used by this thread only.
        #[expect(unsafe_code, reason = "readdir on an open directory stream")]
        let entry = unsafe { libc::readdir(stream.0.as_ptr()) };
        if entry.is_null() {
            let err = io::Error::last_os_error();
            if err.raw_os_error() == Some(0) {
                return Ok(entries);
            }
            return Err(err);
        }
        // SAFETY: `readdir` returned a valid entry whose `d_name` is
        // NUL-terminated and lives until the next call on `stream`.
        #[expect(unsafe_code, reason = "read the name of the entry readdir returned")]
        let name = unsafe { CStr::from_ptr((*entry).d_name.as_ptr()) };
        if matches!(name.to_bytes(), b"." | b"..") {
            continue;
        }
        // An entry removed since `readdir` saw it is left out.
        if let Some((kind, len)) = describe_entry(dir_fd, name)? {
            entries.push(DirEntryBeneath {
                name: OsString::from_vec(name.to_bytes().to_vec()),
                kind,
                len,
            });
        }
    }
}

/// What `name` below `dir_fd` is, without following it, and its length.
fn describe_entry(dir_fd: RawFd, name: &CStr) -> io::Result<Option<(EntryKind, u64)>> {
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: `dir_fd` is the open descriptor of a live directory stream,
    // `name` is NUL-terminated, and `stat` points to writable memory of the
    // right size.
    #[expect(unsafe_code, reason = "fstatat relative to a directory descriptor")]
    let rc = unsafe {
        libc::fstatat(
            dir_fd,
            name.as_ptr(),
            stat.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    };
    if rc != 0 {
        let err = io::Error::last_os_error();
        if err.kind() == io::ErrorKind::NotFound {
            return Ok(None);
        }
        return Err(err);
    }
    // SAFETY: `fstatat` returned 0, so it filled in `stat`.
    #[expect(unsafe_code, reason = "read the stat buffer fstatat filled in")]
    let stat = unsafe { stat.assume_init() };
    let kind = match stat.st_mode & libc::S_IFMT {
        libc::S_IFDIR => EntryKind::Dir,
        libc::S_IFREG => EntryKind::File,
        _ => EntryKind::Other,
    };
    let len = match kind {
        EntryKind::File => u64::try_from(stat.st_size).unwrap_or(0),
        EntryKind::Dir | EntryKind::Other => 0,
    };
    Ok(Some((kind, len)))
}

/// Set `errno` to 0, so a null `readdir` can tell the end from an error.
fn clear_errno() {
    // SAFETY: the errno location is valid for the calling thread.
    #[expect(unsafe_code, reason = "write this thread's errno")]
    unsafe {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            *libc::__errno_location() = 0;
        }
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        {
            *libc::__error() = 0;
        }
    }
}

/// Refuse a write to a file with more than one link.
///
/// A hard link inside the root to a file outside it is not a symlink, so the
/// walk cannot see it. Writing through it would change the outside file.
fn refuse_hard_links(file: &File, target: &Path) -> io::Result<()> {
    use std::os::unix::fs::MetadataExt;
    if file.metadata()?.nlink() > 1 {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!(
                "refusing to write '{}': {}",
                target.display(),
                super::HARD_LINK_REFUSAL
            ),
        ));
    }
    Ok(())
}

/// The names of every component of the absolute path `target`, below `/`.
fn absolute_components(target: &Path) -> io::Result<Vec<&OsStr>> {
    let mut components = target.components();
    if components.next() != Some(Component::RootDir) {
        return Err(invalid_input(format!(
            "'{}' is not an absolute path",
            target.display()
        )));
    }
    components
        .map(|component| match component {
            Component::Normal(name) => Ok(name),
            other => Err(invalid_input(format!(
                "unexpected path component {:?} in '{}'",
                other.as_os_str(),
                target.display()
            ))),
        })
        .collect()
}

/// Refuse anything but a regular file, then make the descriptor blocking again.
fn require_regular_file(file: File, target: &Path) -> io::Result<File> {
    if !file.metadata()?.is_file() {
        return Err(invalid_input(format!(
            "'{}' is not a regular file",
            target.display()
        )));
    }
    // SAFETY: `file` owns an open descriptor for the whole call.
    #[expect(unsafe_code, reason = "fcntl F_GETFL on an owned descriptor")]
    let flags = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFL) };
    if flags < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `file` owns an open descriptor for the whole call.
    #[expect(unsafe_code, reason = "fcntl F_SETFL on an owned descriptor")]
    let set = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETFL, flags & !libc::O_NONBLOCK) };
    if set < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(file)
}

fn components_below<'a>(root: &Path, target: &'a Path) -> io::Result<Vec<&'a OsStr>> {
    let relative = target.strip_prefix(root).map_err(|_| {
        invalid_input(format!(
            "'{}' is not below '{}'",
            target.display(),
            root.display()
        ))
    })?;
    relative
        .components()
        .map(|component| match component {
            Component::Normal(name) => Ok(name),
            other => Err(invalid_input(format!(
                "unexpected path component {:?} in '{}'",
                other.as_os_str(),
                target.display()
            ))),
        })
        .collect()
}

/// Open the directory `name` below `parent`, creating it first for writes.
fn open_dir(parent: &OwnedFd, name: &OsStr, access: Access, target: &Path) -> io::Result<OwnedFd> {
    let c = c_name(name)?;
    match openat(parent, &c, DIR_FLAGS) {
        Err(err) if access == Access::Write && err.kind() == io::ErrorKind::NotFound => {
            match mkdirat(parent, &c) {
                Ok(()) => {}
                Err(err) if err.kind() == io::ErrorKind::AlreadyExists => {}
                Err(err) => return Err(describe(err, parent, name, target)),
            }
            openat(parent, &c, DIR_FLAGS).map_err(|err| describe(err, parent, name, target))
        }
        other => other.map_err(|err| describe(err, parent, name, target)),
    }
}

fn openat(dir: &OwnedFd, name: &CString, flags: libc::c_int) -> io::Result<OwnedFd> {
    let mode: libc::c_uint = 0o666;
    // SAFETY: `dir` is an open directory descriptor borrowed for the call and
    // `name` is a NUL-terminated string that outlives it. `mode` is only read
    // when `flags` has `O_CREAT`.
    #[expect(unsafe_code, reason = "openat relative to a directory descriptor")]
    let fd = unsafe { libc::openat(dir.as_raw_fd(), name.as_ptr(), flags, mode) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `openat` returned a new descriptor that nothing else owns.
    #[expect(
        unsafe_code,
        reason = "take ownership of the descriptor openat returned"
    )]
    let owned = unsafe { OwnedFd::from_raw_fd(fd) };
    Ok(owned)
}

fn mkdirat(dir: &OwnedFd, name: &CString) -> io::Result<()> {
    // SAFETY: `dir` is an open directory descriptor borrowed for the call and
    // `name` is a NUL-terminated string that outlives it.
    #[expect(unsafe_code, reason = "mkdirat relative to a directory descriptor")]
    let rc = unsafe { libc::mkdirat(dir.as_raw_fd(), name.as_ptr(), 0o777) };
    if rc < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

fn c_name(name: &OsStr) -> io::Result<CString> {
    CString::new(name.as_bytes()).map_err(|_| invalid_input("path component contains NUL"))
}

/// Name a symlink refusal so callers and tests can tell it from other errors.
/// It is `PermissionDenied`, as on Windows.
///
/// `O_NOFOLLOW` on a symlink fails with `ELOOP`, or with `ENOTDIR` when
/// `O_DIRECTORY` is also set (macOS), so both are checked with `lstat`.
fn describe(err: io::Error, parent: &OwnedFd, name: &OsStr, target: &Path) -> io::Error {
    let maybe_link = matches!(err.raw_os_error(), Some(libc::ELOOP | libc::ENOTDIR));
    if maybe_link && is_symlink_at(parent, name) {
        return io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!(
                "refusing to follow symlink at '{}' in '{}'",
                Path::new(name).display(),
                target.display()
            ),
        );
    }
    err
}

/// Whether `name` below `parent` is a symlink, without following it.
fn is_symlink_at(parent: &OwnedFd, name: &OsStr) -> bool {
    let Ok(c) = c_name(name) else {
        return false;
    };
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: `parent` is an open directory descriptor, `c` is NUL-terminated,
    // and `stat` points to writable memory of the right size.
    #[expect(unsafe_code, reason = "fstatat relative to a directory descriptor")]
    let rc = unsafe {
        libc::fstatat(
            parent.as_raw_fd(),
            c.as_ptr(),
            stat.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    };
    if rc != 0 {
        return false;
    }
    // SAFETY: `fstatat` returned 0, so it filled in `stat`.
    #[expect(unsafe_code, reason = "read the stat buffer fstatat filled in")]
    let stat = unsafe { stat.assume_init() };
    (stat.st_mode & libc::S_IFMT) == libc::S_IFLNK
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

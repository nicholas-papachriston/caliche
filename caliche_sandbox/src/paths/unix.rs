//! Open a file below an allowed root without following any symlink.
//!
//! `O_NOFOLLOW` only covers the last path component, so a directory replaced
//! by a symlink after validation would still be followed by an open of the
//! full path. Here the root is opened once, and each component below it is
//! opened relative to its parent's descriptor with `openat` and `O_NOFOLLOW`.
//! No component is looked up by a full path string after validation, so a
//! swapped directory makes the open fail instead of leaving the root.

use std::ffi::{CString, OsStr};
use std::fs::File;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
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
const READ_FLAGS: libc::c_int = libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC;
const WRITE_FLAGS: libc::c_int =
    libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC | libc::O_NOFOLLOW | libc::O_CLOEXEC;

/// Open `target`, which must lie strictly below the canonical `root`.
///
/// The root itself is trusted and may be reached through symlinks. Every
/// component below it, including the target, must not be one.
pub(super) fn open_beneath(root: &Path, target: &Path, access: Access) -> io::Result<File> {
    let names = components_below(root, target)?;
    let Some((file_name, dir_names)) = names.split_last() else {
        return Err(invalid_input(format!(
            "'{}' names the sandbox root, not a file",
            target.display()
        )));
    };
    let mut dir: OwnedFd = File::options()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC)
        .open(root)?
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
    Ok(File::from(file))
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
///
/// `O_NOFOLLOW` on a symlink fails with `ELOOP`, or with `ENOTDIR` when
/// `O_DIRECTORY` is also set (macOS), so both are checked with `lstat`.
fn describe(err: io::Error, parent: &OwnedFd, name: &OsStr, target: &Path) -> io::Error {
    let maybe_link = matches!(err.raw_os_error(), Some(libc::ELOOP | libc::ENOTDIR));
    if maybe_link && is_symlink_at(parent, name) {
        return io::Error::new(
            err.kind(),
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

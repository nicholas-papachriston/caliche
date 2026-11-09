//! Landlock roots opened once, when the instance is created.
//!
//! Each root is canonicalized against the process cwd at `create`, then opened
//! with `O_PATH|O_NOFOLLOW`. Later execs add rules from those fds, so replacing
//! a root path with a symlink cannot retarget the ruleset. A shared `/tmp` root
//! is replaced with a private directory for this instance.

use crate::config::SandboxConfig;
use anyhow::{Context, Result};
use std::ffi::CString;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub(super) struct Grant {
    pub(super) fd: OwnedFd,
    pub(super) writable: bool,
}

pub(super) struct Prepared {
    pub(super) grants: Vec<Grant>,
    /// The instance stores this and drops the cgroup field first.
    pub(super) private_temp: Option<tempfile::TempDir>,
}

const SYSTEM_DIRS: &[&str] = &["/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc"];
const DEVICE_ROOTS: &[(&str, bool)] = &[
    ("/dev/null", true),
    ("/dev/zero", true),
    ("/dev/full", true),
    ("/dev/urandom", false),
    ("/dev/random", false),
];

/// Writable roots after a shared `/tmp` is swapped for a private directory.
pub(super) struct Roots {
    writable: Vec<PathBuf>,
    private_temp: Option<tempfile::TempDir>,
}

impl Roots {
    pub(super) fn writable(&self) -> &[PathBuf] {
        &self.writable
    }
}

pub(super) fn roots(config: &SandboxConfig, workspace: &Path) -> Result<Roots> {
    let mut writable = config.writable_roots.clone();
    let private_temp = private_temp_dir(config.tmp_write, workspace, &mut writable)?;
    Ok(Roots {
        writable,
        private_temp,
    })
}

/// Open every root. With `excluded` (git targets relative to the
/// workspace), a writable root equal to the workspace becomes writable
/// grants for everything except the targets and the directories leading to
/// them, which stay read-only.
pub(super) fn prepare(
    roots: Roots,
    workspace: &Path,
    excluded: Option<&[PathBuf]>,
) -> Result<Prepared> {
    let mut grants = Vec::new();
    add_system_dirs(&mut grants)?;
    add_required(&mut grants, workspace, false)?;
    for root in &roots.writable {
        match excluded {
            Some(targets) if same_path(root, workspace) => {
                let frozen = workspace
                    .canonicalize()
                    .with_context(|| format!("canonicalize {}", workspace.display()))?;
                add_excluding(&mut grants, &frozen, targets)?;
            }
            _ => add_required(&mut grants, root, true)?,
        }
    }
    add_devices(&mut grants)?;
    Ok(Prepared {
        grants,
        private_temp: roots.private_temp,
    })
}

/// Writable grants for each entry of `dir` except the targets (relative to
/// `dir`). A directory leading to a target is not granted; its entries are
/// granted the same way. A symlink is skipped; its target is granted by its
/// own root, if any.
fn add_excluding(grants: &mut Vec<Grant>, dir: &Path, targets: &[PathBuf]) -> Result<()> {
    let entries = std::fs::read_dir(dir).with_context(|| format!("list {}", dir.display()))?;
    for entry in entries {
        let entry = entry.with_context(|| format!("list {}", dir.display()))?;
        let name = entry.file_name();
        let below: Vec<PathBuf> = targets
            .iter()
            .filter_map(|target| target.strip_prefix(&name).ok())
            .map(Path::to_path_buf)
            .collect();
        if below.is_empty() {
            if let Some(fd) = open_frozen(&entry.path(), false)? {
                grants.push(Grant { fd, writable: true });
            }
            continue;
        }
        let is_target = below.iter().any(|rest| rest.as_os_str().is_empty());
        let kind = entry
            .file_type()
            .with_context(|| format!("stat {}", entry.path().display()))?;
        if !is_target && kind.is_dir() {
            add_excluding(grants, &entry.path(), &below)?;
        }
    }
    Ok(())
}

pub(super) fn duplicate(grants: &[Grant]) -> Result<Vec<Grant>> {
    grants
        .iter()
        .map(|grant| {
            Ok(Grant {
                fd: grant.fd.try_clone().context("dup a landlock root fd")?,
                writable: grant.writable,
            })
        })
        .collect()
}

pub(super) fn is_directory(fd: &OwnedFd) -> bool {
    mode(fd).is_some_and(|bits| (bits & libc::S_IFMT) == libc::S_IFDIR)
}

/// Character devices such as `/dev/null`. `/dev/tty` is not a granted root.
pub(super) fn is_char_device(fd: &OwnedFd) -> bool {
    mode(fd).is_some_and(|bits| (bits & libc::S_IFMT) == libc::S_IFCHR)
}

fn private_temp_dir(
    tmp_write: bool,
    workspace: &Path,
    writable: &mut Vec<PathBuf>,
) -> Result<Option<tempfile::TempDir>> {
    let requested = tmp_write || writable.iter().any(|path| is_shared_temp(path));
    if !requested {
        return Ok(None);
    }
    writable.retain(|path| keep_writable(path, workspace));
    let dir =
        tempfile::TempDir::with_prefix("caliche-").context("create a private temp directory")?;
    require_utf8(dir.path())?;
    writable.push(dir.path().to_path_buf());
    Ok(Some(dir))
}

fn keep_writable(path: &Path, workspace: &Path) -> bool {
    !is_shared_temp(path) || same_path(path, workspace)
}

fn is_shared_temp(path: &Path) -> bool {
    matches!(
        path.to_str(),
        Some("/tmp" | "/tmp/" | "/private/tmp" | "/private/tmp/")
    )
}

fn same_path(left: &Path, right: &Path) -> bool {
    match (left.canonicalize(), right.canonicalize()) {
        (Ok(left), Ok(right)) => left == right,
        _ => left == right,
    }
}

fn require_utf8(path: &Path) -> Result<()> {
    path.to_str().context("private temp path is not UTF-8")?;
    Ok(())
}

fn add_system_dirs(grants: &mut Vec<Grant>) -> Result<()> {
    for path in SYSTEM_DIRS {
        if let Some(fd) = open_optional_dir(Path::new(path))? {
            grants.push(Grant {
                fd,
                writable: false,
            });
        }
    }
    Ok(())
}

fn open_optional_dir(path: &Path) -> Result<Option<OwnedFd>> {
    let frozen = match path.canonicalize() {
        Ok(path) => path,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(err).with_context(|| format!("canonicalize {}", path.display()));
        }
    };
    open_frozen(&frozen, true)
}

fn add_required(grants: &mut Vec<Grant>, path: &Path, writable: bool) -> Result<()> {
    let frozen = path
        .canonicalize()
        .with_context(|| format!("canonicalize {}", path.display()))?;
    let Some(fd) = open_frozen(&frozen, true)? else {
        anyhow::bail!("open {} returned no descriptor", frozen.display());
    };
    grants.push(Grant { fd, writable });
    Ok(())
}

fn add_devices(grants: &mut Vec<Grant>) -> Result<()> {
    for (path, writable) in DEVICE_ROOTS {
        if let Some(fd) = open_device(Path::new(path))? {
            grants.push(Grant {
                fd,
                writable: *writable,
            });
        }
    }
    Ok(())
}

fn open_device(path: &Path) -> Result<Option<OwnedFd>> {
    let frozen = match path.canonicalize() {
        Ok(path) => path,
        Err(err) if skip_device(&err) => return Ok(None),
        Err(err) => {
            return Err(err).with_context(|| format!("canonicalize {}", path.display()));
        }
    };
    open_frozen(&frozen, false)
}

fn open_frozen(path: &Path, required: bool) -> Result<Option<OwnedFd>> {
    let fd = match open_opath(path) {
        Ok(fd) => fd,
        Err(err) if !required && skip_device(&err) => return Ok(None),
        Err(err) => return Err(err).with_context(|| format!("open {}", path.display())),
    };
    if is_symlink(&fd) {
        if required {
            anyhow::bail!(
                "{} is a symlink; refusing to follow it into a landlock rule",
                path.display()
            );
        }
        return Ok(None);
    }
    Ok(Some(fd))
}

fn skip_device(err: &std::io::Error) -> bool {
    err.kind() == std::io::ErrorKind::NotFound
        || err.raw_os_error() == Some(libc::ENXIO)
        || err.raw_os_error() == Some(libc::ENODEV)
        || err.raw_os_error() == Some(libc::ELOOP)
        || err.raw_os_error() == Some(libc::ENOTDIR)
}

fn open_opath(path: &Path) -> std::io::Result<OwnedFd> {
    let text = CString::new(path.as_os_str().as_encoded_bytes())
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;
    #[expect(unsafe_code, reason = "open a root once with O_PATH and O_NOFOLLOW")]
    let fd = unsafe {
        libc::open(
            text.as_ptr(),
            libc::O_PATH | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    #[expect(unsafe_code, reason = "wrap the opened root fd")]
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

fn is_symlink(fd: &OwnedFd) -> bool {
    mode(fd).is_some_and(|bits| (bits & libc::S_IFMT) == libc::S_IFLNK)
}

fn mode(fd: &OwnedFd) -> Option<libc::mode_t> {
    // SAFETY: `fstat` on an `O_PATH` fd is safe. The buffer is zeroed.
    #[expect(unsafe_code, reason = "fstat an O_PATH root fd")]
    unsafe {
        let mut stat = std::mem::zeroed();
        if libc::fstat(fd.as_raw_fd(), &mut stat) == 0 {
            Some(stat.st_mode)
        } else {
            None
        }
    }
}

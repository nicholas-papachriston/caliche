//! Open a file below an allowed root without following any reparse point.
//!
//! `CreateFileW` with `FILE_FLAG_OPEN_REPARSE_POINT` only skips reparse
//! processing for the last path component, so a junction or symlink placed on
//! a parent directory after validation would still be traversed. Here each
//! component below the root is opened relative to its parent's handle with
//! `NtCreateFile` and `FILE_OPEN_REPARSE_POINT`, then refused if it carries
//! `FILE_ATTRIBUTE_REPARSE_POINT`. No component is looked up by a full path
//! string after it has been checked. Every directory handle stays open without
//! `FILE_SHARE_DELETE` until the target is open, so none of them can be
//! renamed, moved, or deleted while the walk is in progress.

use std::ffi::OsStr;
use std::fs::File;
use std::io;
use std::os::windows::ffi::OsStrExt;
use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::path::{Component, Path};
use windows_sys::Wdk::Foundation::OBJECT_ATTRIBUTES;
use windows_sys::Wdk::Storage::FileSystem::{
    FILE_DIRECTORY_FILE, FILE_NON_DIRECTORY_FILE, FILE_OPEN, FILE_OPEN_IF, FILE_OPEN_REPARSE_POINT,
    FILE_SYNCHRONOUS_IO_NONALERT, NtCreateFile,
};
use windows_sys::Win32::Foundation::{
    GENERIC_READ, GENERIC_WRITE, HANDLE, RtlNtStatusToDosError, UNICODE_STRING,
};
use windows_sys::Win32::Storage::FileSystem::{
    FILE_ATTRIBUTE_NORMAL, FILE_ATTRIBUTE_REPARSE_POINT, FILE_FLAG_BACKUP_SEMANTICS,
    FILE_SHARE_READ, FILE_SHARE_WRITE, SYNCHRONIZE,
};
use windows_sys::Win32::System::IO::{IO_STATUS_BLOCK, IO_STATUS_BLOCK_0};
use windows_sys::Win32::System::Kernel::OBJ_CASE_INSENSITIVE;

/// Share mode for every handle in the walk. Leaving out `FILE_SHARE_DELETE`
/// blocks renames and deletes of the held directories and of the target.
const SHARE: u32 = FILE_SHARE_READ | FILE_SHARE_WRITE;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Access {
    /// Open an existing file for reading.
    Read,
    /// Create missing parent directories, then open or create the file and
    /// truncate it.
    Write,
}

/// Open `target`, which must lie strictly below the canonical `root`.
///
/// The root itself is trusted and may be reached through reparse points.
/// Every component below it, including the target, must not be one.
pub(super) fn open_beneath(root: &Path, target: &Path, access: Access) -> io::Result<File> {
    let names = components_below(root, target)?;
    let Some((file_name, dir_names)) = names.split_last() else {
        return Err(invalid_input(format!(
            "'{}' names the sandbox root, not a file",
            target.display()
        )));
    };

    // Ancestors of `parent` stay open until the target is open; dropping
    // them would let a held directory be moved while the walk continues.
    let mut ancestors = Vec::with_capacity(dir_names.len());
    let mut parent = open_root(root)?;
    for name in dir_names {
        let dir = open_dir_child(&parent, name, access)?;
        refuse_reparse_point(&dir, root, target)?;
        ancestors.push(std::mem::replace(&mut parent, dir));
    }

    let file = open_file_child(&parent, file_name, access)?;
    refuse_reparse_point(&file, root, target)?;
    if access == Access::Write {
        file.set_len(0)?;
    }
    drop(ancestors);
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

fn open_root(root: &Path) -> io::Result<File> {
    File::options()
        .read(true)
        .share_mode(SHARE)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(root)
}

fn open_dir_child(parent: &File, name: &OsStr, access: Access) -> io::Result<File> {
    let disposition = match access {
        Access::Read => FILE_OPEN,
        Access::Write => FILE_OPEN_IF,
    };
    // FILE_DIRECTORY_FILE fails the open when `name` is not a directory.
    nt_open_relative(
        parent,
        name,
        GENERIC_READ | SYNCHRONIZE,
        disposition,
        FILE_DIRECTORY_FILE,
    )
}

fn open_file_child(parent: &File, name: &OsStr, access: Access) -> io::Result<File> {
    let (desired, disposition) = match access {
        Access::Read => (GENERIC_READ | SYNCHRONIZE, FILE_OPEN),
        Access::Write => (GENERIC_READ | GENERIC_WRITE | SYNCHRONIZE, FILE_OPEN_IF),
    };
    nt_open_relative(parent, name, desired, disposition, FILE_NON_DIRECTORY_FILE)
}

/// `NtCreateFile` of the single component `name` relative to `parent`.
///
/// `FILE_OPEN_REPARSE_POINT` opens a reparse point itself instead of its
/// target, so the caller can inspect and refuse it.
fn nt_open_relative(
    parent: &File,
    name: &OsStr,
    desired_access: u32,
    disposition: u32,
    kind_option: u32,
) -> io::Result<File> {
    let mut wide: Vec<u16> = name.encode_wide().collect();
    let byte_len = wide
        .len()
        .checked_mul(2)
        .and_then(|len| u16::try_from(len).ok())
        .ok_or_else(|| invalid_input("path component is too long"))?;
    let object_name = UNICODE_STRING {
        Length: byte_len,
        MaximumLength: byte_len,
        Buffer: wide.as_mut_ptr(),
    };
    let attributes = OBJECT_ATTRIBUTES {
        Length: u32::try_from(size_of::<OBJECT_ATTRIBUTES>()).unwrap_or(u32::MAX),
        RootDirectory: parent.as_raw_handle(),
        ObjectName: &raw const object_name,
        // Match Win32 name lookup, which is case-insensitive.
        Attributes: OBJ_CASE_INSENSITIVE.cast_unsigned(),
        SecurityDescriptor: std::ptr::null(),
        SecurityQualityOfService: std::ptr::null(),
    };
    let mut io_status = IO_STATUS_BLOCK {
        Anonymous: IO_STATUS_BLOCK_0 { Status: 0 },
        Information: 0,
    };
    let mut handle: HANDLE = std::ptr::null_mut();
    // SAFETY: `handle` and `io_status` are writable locals. `attributes`
    // points at `object_name`, whose buffer `wide` holds `byte_len` bytes of
    // UTF-16 and outlives the call; `Length` is the struct size and
    // `RootDirectory` is a live directory handle borrowed from `parent`. A
    // null allocation size and a null EA buffer with length 0 are valid.
    #[expect(unsafe_code, reason = "NtCreateFile FFI for a parent-relative open")]
    let status = unsafe {
        NtCreateFile(
            &raw mut handle,
            desired_access,
            &raw const attributes,
            &raw mut io_status,
            std::ptr::null(),
            FILE_ATTRIBUTE_NORMAL,
            SHARE,
            disposition,
            kind_option | FILE_OPEN_REPARSE_POINT | FILE_SYNCHRONOUS_IO_NONALERT,
            std::ptr::null(),
            0,
        )
    };
    if status < 0 {
        // SAFETY: RtlNtStatusToDosError takes a plain status value and reads
        // no memory.
        #[expect(unsafe_code, reason = "Win32 FFI to translate an NTSTATUS")]
        let code = unsafe { RtlNtStatusToDosError(status) };
        return Err(io::Error::from_raw_os_error(
            i32::try_from(code).unwrap_or(i32::MAX),
        ));
    }
    // SAFETY: NtCreateFile succeeded, so `handle` is a new open file handle
    // that nothing else owns; `OwnedHandle` takes it and closes it on drop.
    #[expect(unsafe_code, reason = "adopt the handle returned by NtCreateFile")]
    let owned = unsafe { OwnedHandle::from_raw_handle(handle) };
    Ok(File::from(owned))
}

fn refuse_reparse_point(opened: &File, root: &Path, target: &Path) -> io::Result<()> {
    if opened.metadata()?.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT == 0 {
        return Ok(());
    }
    Err(io::Error::new(
        io::ErrorKind::PermissionDenied,
        format!(
            "refusing to follow reparse point between '{}' and '{}'",
            root.display(),
            target.display()
        ),
    ))
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

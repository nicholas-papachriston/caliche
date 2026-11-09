//! Absolute paths for the system tools the agent runs itself.
//!
//! `Command::new("kill")` searches `PATH`, so a `kill` planted in the
//! workspace or in an earlier `PATH` entry would run in its place. These
//! lookups check fixed system directories only and never read `PATH`.

use std::path::{Path, PathBuf};

/// System directories searched in order. macOS keeps `lsof` in `/usr/sbin`;
/// NixOS keeps everything in `/run/current-system/sw/bin`.
const SYSTEM_BIN_DIRS: [&str; 5] = [
    "/bin",
    "/usr/bin",
    "/usr/sbin",
    "/sbin",
    "/run/current-system/sw/bin",
];

/// A system tool the agent runs by absolute path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SystemBinary {
    Hostname,
    Kill,
    Lsof,
    Ps,
}

impl SystemBinary {
    const fn file_name(self) -> &'static str {
        match self {
            Self::Hostname => "hostname",
            Self::Kill => "kill",
            Self::Lsof => "lsof",
            Self::Ps => "ps",
        }
    }

    /// The first system directory that holds this tool, or `None` when no
    /// system directory does. `PATH` is never consulted.
    pub fn path(self) -> Option<PathBuf> {
        SYSTEM_BIN_DIRS
            .iter()
            .map(|dir| Path::new(dir).join(self.file_name()))
            .find(|candidate| candidate.is_file())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_each_tool_in_a_system_directory() {
        for binary in [
            SystemBinary::Hostname,
            SystemBinary::Kill,
            SystemBinary::Lsof,
            SystemBinary::Ps,
        ] {
            let path = binary.path().expect("system tool present");
            assert!(
                SYSTEM_BIN_DIRS.iter().any(|dir| path.starts_with(dir)),
                "{} is outside the system directories",
                path.display()
            );
            assert_eq!(path.file_name().unwrap(), binary.file_name());
        }
    }
}

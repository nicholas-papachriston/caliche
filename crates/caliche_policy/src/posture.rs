//! Layers, presets, overlays, facets, and sandbox obligations.
//!
//! Preset tables are named here. They are not wired through the live tool
//! path. Workstation default is [`Preset::Standard`]. New deployments
//! default to [`Preset::Locked`]. [`Preset::Trusted`] leaves shell analyzers
//! off and still requires a workspace-write sandbox.

use serde::{Deserialize, Serialize};

/// Policy layer. Lower layers cannot raise a deployment deny.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Layer {
    /// Compiled-in denies and hard asks. Plugins cannot replace this layer.
    Floor,
    /// Operator policy. Full allow / ask / deny and the preset default.
    Deployment,
    /// Agent profile. Narrow only.
    Profile,
    /// Repo `caliche.toml`. Narrow only unless the deployment opts in.
    Project,
    /// Live session grants and mode overlays. Narrow, or resolve a non-hard ask.
    Session,
}

/// Named deployment preset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Preset {
    /// Unattended use, production, and new deploys. Default deny.
    Locked,
    /// Plan / ask posture. Default deny.
    Review,
    /// Interactive workstation default. Default ask. Shell analyzer on once
    /// content rules exist.
    Standard,
    /// Relaxed preset. Tool-level allow, no shell content rules, sandbox stays on.
    Trusted,
    /// Explicit full access. Refused for production profiles.
    Danger,
}

impl Preset {
    /// Preset for a developer workstation.
    #[must_use]
    pub const fn workstation_default() -> Self {
        Self::Standard
    }

    /// Preset compiled into a new deployment.
    #[must_use]
    pub const fn new_deploy_default() -> Self {
        Self::Locked
    }

    /// Effect used when no rule in the deployment layer matches.
    #[must_use]
    pub const fn default_effect(self) -> crate::Effect {
        match self {
            Self::Locked | Self::Review => crate::Effect::Deny,
            Self::Standard => crate::Effect::Ask,
            Self::Trusted | Self::Danger => crate::Effect::Allow,
        }
    }

    /// Sandbox obligation stub for this preset.
    ///
    /// `trusted` keeps the sandbox on. `danger` asks for none and is refused
    /// when the deployment is a production profile.
    #[must_use]
    pub const fn sandbox(self) -> SandboxRequirement {
        match self {
            Self::Danger => SandboxRequirement::None,
            Self::Trusted | Self::Standard => SandboxRequirement::WorkspaceWrite,
            Self::Locked | Self::Review => SandboxRequirement::Required,
        }
    }

    /// Obligations that accompany [`Self::sandbox`].
    #[must_use]
    pub const fn obligations(self) -> Obligations {
        match self {
            Self::Locked | Self::Review => Obligations::required(),
            Self::Standard | Self::Trusted => Obligations::workspace_write(),
            Self::Danger => Obligations::none(),
        }
    }

    /// Wire name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Locked => "locked",
            Self::Review => "review",
            Self::Standard => "standard",
            Self::Trusted => "trusted",
            Self::Danger => "danger",
        }
    }
}

/// Session mode overlay. These never change a deny.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModeOverlay {
    /// No overlay.
    #[default]
    Default,
    /// Lower a non-hard ask to allow for filesystem writes and patches.
    AcceptEdits,
    /// Lower a non-hard ask to allow. This is the `--yolo` overlay.
    AcceptAll,
    /// Deny every action that is not a read.
    DenyAll,
    /// Raise allow to ask for side-effecting tools.
    Interactive,
}

/// How much sandbox the preset requires.
///
/// `Noop` is an explicit opt-in. Production compilation rejects it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SandboxRequirement {
    /// No sandbox obligation (`danger`).
    None,
    /// Confine writes to the workspace. Network honesty is per host.
    WorkspaceWrite,
    /// Sandbox required (`locked`, `review`).
    Required,
    /// Explicit noop backend. Refused for unattended production use.
    Noop,
}

impl SandboxRequirement {
    /// Wire name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::WorkspaceWrite => "workspace_write",
            Self::Required => "required",
            Self::Noop => "noop",
        }
    }
}

/// Filesystem confinement an obligation can name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FsConfine {
    /// No filesystem obligation.
    None,
    /// Writes stay inside the workspace.
    WorkspaceWrite,
    /// Reads only.
    ReadOnly,
}

/// Network confinement an obligation can name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NetConfine {
    /// No network obligation.
    None,
    /// Deny outbound network.
    DenyAll,
    /// Public hosts only. The SSRF floor still applies.
    AllowPublic,
}

/// Process confinement an obligation can name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcConfine {
    /// No process obligation.
    None,
    /// Kill the process tree when the sandbox handle drops.
    KillOnDrop,
}

/// Sandbox requirements attached to a decision.
///
/// A backend must report what it cannot enforce. The snapshot records the
/// obligation. It does not launch a sandbox.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Obligations {
    /// Filesystem confinement.
    pub fs: FsConfine,
    /// Network confinement.
    pub net: NetConfine,
    /// Process confinement.
    pub proc: ProcConfine,
}

impl Obligations {
    /// No sandbox requirements.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            fs: FsConfine::None,
            net: NetConfine::None,
            proc: ProcConfine::None,
        }
    }

    /// Workspace write confinement and public egress.
    #[must_use]
    pub const fn workspace_write() -> Self {
        Self {
            fs: FsConfine::WorkspaceWrite,
            net: NetConfine::AllowPublic,
            proc: ProcConfine::KillOnDrop,
        }
    }

    /// Locked / review: workspace writes, deny network, kill on drop.
    #[must_use]
    pub const fn required() -> Self {
        Self {
            fs: FsConfine::WorkspaceWrite,
            net: NetConfine::DenyAll,
            proc: ProcConfine::KillOnDrop,
        }
    }
}

/// Bitset of facets that are active for a compiled snapshot.
///
/// Shell bits are set only when a layer has a shell *content* rule. Tool-level
/// allow / ask / deny does not set them. When a bit is clear, the shell
/// analyzer is not called and does not allocate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ActiveFacets(u32);

impl ActiveFacets {
    /// POSIX bash/zsh/sh content rules.
    pub const SHELL_POSIX: Self = Self(1 << 0);
    /// `PowerShell` content rules.
    pub const SHELL_PWSH: Self = Self(1 << 1);
    /// cmd.exe content rules.
    pub const SHELL_CMD: Self = Self(1 << 2);
    /// Path canonicalization.
    pub const FS_PATHS: Self = Self(1 << 3);
    /// Egress classification.
    pub const EGRESS: Self = Self(1 << 4);
    /// MCP tool classification.
    pub const MCP_TOOLS: Self = Self(1 << 5);
    /// Hash bindings even for allows.
    pub const AUDIT_BINDING: Self = Self(1 << 6);

    /// No facets.
    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Whether every bit in `other` is set.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Union of two bitsets.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Raw bits. One integer compare is the off-switch.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }
}

/// Host isolation the caller claims.
///
/// `decide` does not probe the OS and does not enforce Windows isolation.
/// The variants exist so obligations can be joined with capabilities without
/// inventing a Windows kernel sandbox.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostCapability {
    /// Caller did not name a host.
    #[default]
    Unspecified,
    /// macOS Seatbelt (files and network deny).
    MacOsSeatbelt,
    /// Linux Landlock. `network_enforceable` is true only when the caller
    /// probed TCP bind and connect (`landlock-net`, kernel ABI 4+).
    LinuxLandlock {
        /// Whether this kernel can deny TCP connect.
        network_enforceable: bool,
    },
    /// Windows Job object. Process and memory caps only. Files and network
    /// are not confined. Unattended use does not meet the v1 bar.
    WindowsJobOnly,
    /// Docker or WSL2. Required for unattended and production Windows.
    WindowsDockerOrWsl2,
    /// Docker on any OS that has a daemon.
    Docker,
    /// Explicit noop. Refused for locked production use.
    Noop,
}

impl HostCapability {
    /// Whether unattended shell or code execution has a real isolator.
    ///
    /// `WindowsJobOnly` is false: Job does not confine files or network, and
    /// v1 does not add a Windows kernel sandbox. Use Docker or WSL2.
    #[must_use]
    pub const fn unattended_shell_supported(self) -> bool {
        match self {
            Self::Unspecified | Self::WindowsJobOnly | Self::Noop => false,
            Self::MacOsSeatbelt
            | Self::WindowsDockerOrWsl2
            | Self::Docker
            | Self::LinuxLandlock { .. } => true,
        }
    }

    /// Stable name for `policy show` and `policy.toml` host queries.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Unspecified => "unspecified",
            Self::MacOsSeatbelt => "macos_seatbelt",
            Self::LinuxLandlock {
                network_enforceable: true,
            } => "linux_landlock",
            Self::LinuxLandlock {
                network_enforceable: false,
            } => "linux_landlock_no_network",
            Self::WindowsJobOnly => "windows_job_only",
            Self::WindowsDockerOrWsl2 => "windows_docker_or_wsl2",
            Self::Docker => "docker",
            Self::Noop => "noop",
        }
    }
}

/// Sandbox backend names.
///
/// `WindowsJob` is resource-only. There is no Windows kernel sandbox.
/// Unattended Windows requires Docker or WSL2.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SandboxBackend {
    /// macOS `sandbox-exec` / Seatbelt.
    Seatbelt,
    /// Linux Landlock (filesystem). Network deny is not recorded here.
    Landlock,
    /// Docker, including the Windows unattended path.
    Docker,
    /// wasmtime for wasm modules, not arbitrary shell.
    Wasm,
    /// Windows Job object. Not a filesystem or network sandbox.
    WindowsJob,
    /// No isolation.
    Noop,
}

/// Shell family. Each family has a denylist analyzer behind its facet bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShellDialect {
    /// bash, zsh, or sh.
    Posix,
    /// `PowerShell` 7 or Windows `PowerShell` 5.1.
    PowerShell,
    /// `cmd.exe`.
    Cmd,
}

impl ShellDialect {
    /// Facet bit that gates this dialect's analyzer.
    #[must_use]
    pub const fn facet(self) -> ActiveFacets {
        match self {
            Self::Posix => ActiveFacets::SHELL_POSIX,
            Self::PowerShell => ActiveFacets::SHELL_PWSH,
            Self::Cmd => ActiveFacets::SHELL_CMD,
        }
    }
}

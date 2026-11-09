# Caliche Sandbox

Trait-based sandbox isolation for agent tool execution. Used by the CLI (`--sandbox`) and core tool runtime to confine bash, file, and related operations.

## Backends

| Backend | Platform | Cargo feature |
| ------- | -------- | ------------- |
| macOS Seatbelt | macOS | `sandbox-macos` |
| Linux Landlock | Linux | `sandbox-linux` |
| Landlock TCP deny | Linux, kernel ABI 4+ | `landlock-net` |
| Windows Job Object | Windows | `sandbox-windows` |
| Docker | cross-platform | `docker` |
| Wasmtime WASM | cross-platform | `sandbox-wasm` |
| Noop | testing only | (always available) |

`native_backend_for_platform()` selects Seatbelt, Landlock, or Windows Job for the current OS. Unsupported platforms return an error instead of silently falling back to Noop.

## Policies

Named policies (CLI `--sandbox` values):

- `read-only` — no writes. Network isolation is TCP bind and connect only.
- `workspace-write` — writes under the workspace and a private temp dir. Network isolation is TCP bind and connect only. With `landlock-net` on a kernel that accepts TCP rules, Landlock denies bind and connect. No port is allowed. If that deny cannot be applied, Landlock logs once and leaves TCP bind and connect allowed. UDP, raw sockets, and sockets inherited from the parent stay outside that claim.
- `network-restricted` — workspace writes; network allowlists are not enforced by backends (use read-only or workspace-write instead)
- `danger-full-access` — explicit opt-out of sandboxing (logged warning)

## Fail-closed behavior

- Default CLI sandbox backend is `native`, not Noop. Use `--sandbox-backend noop` only for explicit testing.
- Backend feature or platform mismatches return errors at initialization rather than degrading silently.
- `validate_sandbox_config()` rejects `network-restricted` egress allowlists because no backend enforces them.
- On Linux, `NetworkPolicy::HostOnly` is refused even when the `landlock-net` feature is off. Landlock cannot express a host-only allowlist, so the backend does not run the command with an open network.
- On Linux, each exec runs in its own cgroup v2 cgroup inside a cgroup for the instance. A timeout writes `cgroup.kill`, which also kills processes that left the process group with `setsid`. Processes still running when the instance is dropped are killed with it. This needs cgroup v2 at `/sys/fs/cgroup`, kernel 5.14+, and write access to the caller's cgroup (root, or a systemd-delegated cgroup). Without them the backend logs a warning once and a timeout kills the exec's process group. The child calls `setsid`, which sets that group id to the child pid, so the child is included. A grandchild that calls `setsid` again leaves the group.
- On Linux, each root is canonicalized and opened once at instance creation with `O_PATH|O_NOFOLLOW`. Every exec builds the Landlock ruleset from those fds in the parent, before `spawn`. The forked child calls `setsid` (so it is not in the agent's session and `/dev/tty` is not the user's terminal), then `prctl(PR_SET_NO_NEW_PRIVS)`, `landlock_restrict_self` on that already-open ruleset fd, `write` of `0` to the exec cgroup, and `close_range` to mark descriptors from 3 upward close-on-exec. Those calls are async-signal-safe. The parent does not put the child in its own process group first, because that would make `setsid` fail. The parent closes the ruleset fd after `spawn`. Stdin is `/dev/null`. When the kernel ABI supports them, abstract-unix and signal scopes are handled best-effort and do not weaken the filesystem or TCP rules. `tcp_isolation` reports TCP bind and connect only. Read-write is allowed on `/dev/null`, `/dev/zero`, and `/dev/full`; read is allowed on `/dev/urandom` and `/dev/random`. `/dev/tty` is not granted, and character devices do not receive `IoctlDev`. A workspace-write policy gets a private temp directory and `TMPDIR` instead of a shared `/tmp`.
- On Windows, each exec runs in its own Job Object nested in the instance job. A timeout terminates only that exec's processes. The instance job applies the limits and kills what is left when the instance is dropped. Breakaway stays off (`JOB_OBJECT_LIMIT_BREAKAWAY_OK` and `JOB_OBJECT_LIMIT_SILENT_BREAKAWAY` are not set), so `CreateProcess`, `start /b`, and `Start-Process` children stay in the job and share its cap. `CREATE_BREAKAWAY_FROM_JOB` fails. WMI `Win32_Process.Create` and Task Scheduler (`schtasks`) are a known escape: those services start the new process outside the job, and it is not CPU-capped.
- `resource_limits.cpu_rate_hard_cap_percent` is a percentage of the whole machine (every logical processor), from 1 through 100. It is not a percentage of one CPU. The default is no CPU-rate control (`None`). When set, the instance job gets `JOBOBJECT_CPU_RATE_CONTROL_INFORMATION` with `JOB_OBJECT_CPU_RATE_CONTROL_ENABLE | JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP` and `CpuRate = percent * 100` (hundredths of a percent of processor cycles; 100 is 1% of the machine, 10_000 is 100%). Nested per-exec jobs run under that parent cap. On a 16-CPU host a 20% cap measured about 19% of the machine. A 1% cap measured about 0.1% and did not run over the ceiling; the low end undershoots, and the rate is not scaled up to compensate. If Windows rejects the cap, the instance is not created. Values outside 1 through 100 fail `validate_sandbox_config`. `create_backend` and every backend `create` call that validation, so Docker, Landlock, Seatbelt, Wasm, Noop, and AgentFS reject a set cap instead of ignoring it.

## Git metadata

Host `git` runs programs that repository config and hooks name (filters, `gpg.program`, `core.fsmonitor`, `core.hooksPath`, diff drivers). The agent must not choose them.

- **Sandboxed commands.** Git metadata stays read-only: the root `.git` (a command cannot create one in a workspace without it) and the `.git` path of every submodule in the root index and, recursively, in each submodule's own index (checked out or not). macOS reads only the indexes and never walks the workspace tree, so build output such as `target/` or `node_modules/` and directories nobody can list cost nothing and refuse nothing there. Linux and Docker also add every nested `.git` a bounded scan finds (below).
  - macOS Seatbelt denies writes to any `.git` path component (any case) in the workspace, including one created later, and to a `.git` entry directly in any parent of the workspace. Git discovery also takes a directory holding `HEAD`, `objects` and `refs` as a bare repository, so macOS denies writes to `HEAD` (any case) in every parent and in the workspace root, including when the workspace already has a `.git`. A `.git` git later rejects would otherwise fall through to that `HEAD`. Host `git` discovery therefore finds no repository a command plants in the workspace root or a parent. It also denies writes to the directory entries of the workspace, each of its parents, and every directory on the path to a submodule in any index (nested submodules, and submodules that are not checked out, included). So a writable root around the workspace (such as `/tmp`) cannot be used to move the workspace or a submodule aside, rewrite its `.git`, and move it back, or to fill in a missing submodule directory. Entries inside those directories stay writable.
  - macOS refuses a workspace where a directory on the path to a submodule is a symlink.
  - macOS computes the directories on the path to a submodule when the instance is created. A submodule the host adds later gets no such deny until a new instance is created.
  - macOS residuals: names that only HFS+ folds to `.git` (a zero-width joiner or another ignorable code point inside `.git`) are not matched; APFS does not fold them. Config that host `git` reads through `include.path` or `includeIf` pointing at a workspace file is not blocked. With `$HOME` as a writable root, `~/.gitconfig` is writable. A bare repository planted in a subdirectory (for example `src/`) is found by `git` run from inside it, the same residual as `src/.git` on Linux and Docker; the REPL's own `git` never uses one (below).
  - Inside the macOS sandbox, `git submodule deinit`, renaming a directory that leads to a submodule or nested repository, and `rm -rf` of such a directory fail. `rm -rf` deletes the ordinary files it reaches first, so a partial delete can happen; git metadata is not touched.
  - Linux with only a root `.git`: each exec binds it read-only in a new mount namespace. When user namespaces are blocked, when there is a submodule or a nested repository, or when the workspace has no `.git`, the workspace root and every directory leading to git metadata get no Landlock grant, and every other entry gets its own. A command cannot write, delete, or rename that metadata, rename a directory that leads to it, or create a `.git` at a submodule path. It also cannot create, rename, or delete a file directly in the workspace root or in a directory that leads to git metadata (for example a submodule's top directory). Edit existing files there, or work in a subdirectory.
  - Linux refuses to start when the targets cannot be read (a damaged or split index, or submodules nested too deep) and when a writable root overlaps them.
  - Docker binds the root `.git` again with `:ro`. A bind cannot keep a submodule's `.git` read-only, because renaming its parent moves the bind aside, and cannot cover a `.git` that does not exist yet. So Docker refuses a workspace with a submodule or a nested repository, and a workspace without a root `.git`.
  - Linux refuses a writable root that contains the workspace from outside it, in both modes. Docker refuses a writable mount that contains the workspace or lies inside its `.git`.
  - A git dir or common dir that a `.git` file names (`git init --separate-git-dir`, a linked worktree, a submodule) is refused on every backend when it is inside the workspace under a name other than `.git`, or under a writable root or writable Docker mount, or when a writable root or mount lies inside it. The same refusal applies when `HEAD`, `objects`, `refs`, or `commondir` inside that dir is a symlink the sandbox could retarget or whose target it could delete. A workspace without a `.git` of its own is checked the same way against the repository `git` discovery finds above it (a parent's `.git`, such as a linked worktree's gitfile, or a parent that is a bare repository); a workspace that is a bare repository itself is refused. Each is resolved one component at a time, as the kernel does, and a symlink on the way that lies inside the workspace or under a writable root is refused, because the sandbox could retarget it.
  - A symlinked root `.git` is refused.
  - On the way up from the workspace, a `.git` that is not a valid gitfile or git dir (an empty `.git` from an interrupted clone, for example) is refused on every backend, and so is a symlinked `.git`, which `git` would follow. A `.git` counts only when git's own loose test would use it. A `HEAD` in the workspace or any parent that is not a git dir is refused too, and so is a `HEAD` beside a `.git` that was accepted, because a command could complete it into a repository if that `.git` were later rejected. The error names the path. Create classifies the workspace once.
  - On Linux, a writable workspace below the root of the repository that holds it (a monorepo package, for example) is refused, because the guard reads only the workspace's own index and would miss the enclosing repository's submodules. Open the repository root instead; the error names it. macOS allows it.
  - Windows Job Objects do not keep git metadata read-only.
  - `.gitmodules` in the workspace stays writable. A URL planted there reaches the host only when you run `git submodule init`, `sync`, or `update`; `ext::` URLs are off by default in git.
  - Linux and Docker scan the writable roots for nested `.git` entries (a nested repository that is not a submodule, a copied gitfile, any case) once, when the first sandboxed tool runs, never at startup. The scan does not follow symlinks or read inside a `.git`. Each `.git` it finds in the workspace is a target like a submodule's (Linux keeps it read-only, Docker refuses the workspace); one under a writable root outside the workspace is refused. The scan stops at 1,000,000 entries, 64 directories deep, or 10 seconds. A bound hit, or a directory it cannot list, refuses sandboxed tools with a one-line reason naming the path; the rest of the session keeps working. So on Linux and in Docker a large tree such as `~` gets no sandboxed tools, and neither does a workspace with a gitfile copied without its git dir (for example `node_modules/<pkg>/.git` after `bun install` of a package with submodules), because a command could create that git dir. A nested repository under `target/` or `node_modules/` is guarded like a submodule on Linux, so files cannot be created directly in the directories leading to it, and Docker refuses the workspace.
  - **Residual (Linux and Docker):** a `.git` created after the scan (for example `git init src` by a sandboxed command) is not blocked until a new session starts. Host `git` in the workspace root does not read it, but `git` run from inside that directory would use it as a repository. macOS blocks it by name.
- **Host write tools.** `fs` write, edit and delete, `write_file`, `edit_file`, and `apply_patch` (add, update, delete, and the `Move to` target) refuse a path inside git metadata: a `.git` component at any depth (any case, also with a trailing dot or space), a nested repository, a submodule git dir, or a git dir that a `.git` file points to. The path is resolved first, so `..` and symlinks do not get around it. Approval modes, including accept-all, do not turn this off.
- **Agent git reads.** The workspace root, dirty files, and git context come from libgit2, which runs no filter, diff driver, hook, or signing program. Without the `tools-git` feature these reads return nothing. They do not fall back to host `git`.
- **User git commands.** The REPL commands `/commit`, `/pr`, and `/review` run your own `git`, with your hooks and signing, plus guards that narrow what a backend gap reaches: `core.fsmonitor` is off for that `git` and every `git` it starts in a submodule; `safe.bareRepository` is `explicit`, so discovery never uses a bare repository, and a directory with a git dir but no work tree is refused; and the `.git` of each submodule in the root index must be a gitfile into the superproject's `.git/modules`. A submodule `.git` that is a directory, a symlink, or a gitfile whose named git dir is outside `.git/modules` (as `/commit` reads the gitfile) is refused with an error naming it, before any `git` reads that submodule. A work tree whose own `.git` is a symlink is refused the same way: the error says that entry is not a valid gitfile or git dir. Git would follow the link. These guards do not turn off filters, hooks, or other programs a submodule's config names; keeping that config read-only is the sandbox's job.
- **Limit.** Inside the sandbox, `git add` and `git commit` fail because `.git` is read-only. Use `/commit` from the REPL.

## Installation

From the workspace root:

```bash
cargo build -p caliche_sandbox
```

Enable platform backends as needed:

```bash
cargo build -p caliche_sandbox --features sandbox-linux
cargo build -p caliche_sandbox --features landlock-net
cargo build -p caliche_sandbox --features sandbox-macos
cargo build -p caliche_sandbox --features sandbox-windows
cargo build -p caliche_sandbox --features docker
```

## Examples

```bash
cargo run -p caliche_sandbox --example noop_backend
cargo run -p caliche_sandbox --features sandbox-linux --example linux_landlock_backend
cargo run -p caliche_sandbox --features sandbox-macos --example macos_seatbelt_backend
cargo run -p caliche_sandbox --features sandbox-windows --example windows_job_backend
```

## License

Licensed under either of [Apache-2.0](../LICENSE-APACHE) or [MIT](../LICENSE-MIT), at your option.

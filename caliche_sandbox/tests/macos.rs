#![cfg(all(feature = "sandbox-macos", target_os = "macos"))]

use caliche_policy::{SAFE_BARE_REPOSITORY_KEY, clear_git_config, plant_linked_worktree};
use caliche_sandbox::backends::macos::{MacOSSeatbeltBackend, build_seatbelt_profile};
use caliche_sandbox::paths::{
    CONFIG_FILE, GIT_DIR_NAME, GITFILE_PREFIX, HEAD_FILE, OBJECTS_DIR, REFS_DIR, git_refusal,
};
use caliche_sandbox::traits::SandboxBackend;
use caliche_sandbox::{NamedPolicy, SandboxBackendType, policy_to_config};
use std::time::Duration;

mod git_shapes;

#[test]
fn workspace_write_profile_denies_unlisted_writes() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let workspace = dir.path().to_path_buf();
    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &workspace,
        None,
        SandboxBackendType::MacOSSeatbelt {
            policy: NamedPolicy::WorkspaceWrite,
        },
    );
    let profile = build_seatbelt_profile(&config, workspace.as_path()).expect("profile");
    assert!(profile.contains("(deny default)"));
    assert!(!profile.contains("(allow default)"));
    assert!(
        !profile
            .lines()
            .any(|line| line.trim() == "(allow file-read*)"),
        "global file-read* must not be granted: {profile}"
    );
    assert!(profile.contains("(allow file-read* (literal \"/\"))"));
    assert!(profile.contains("(allow file-map-executable)"));
    assert!(profile.contains("(deny file-write*)"));
    assert!(profile.contains("(allow file-write* (subpath"));
}

#[test]
fn writable_root_quotes_are_escaped_in_profile() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let workspace = dir.path().to_path_buf();
    let mut config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &workspace,
        None,
        SandboxBackendType::MacOSSeatbelt {
            policy: NamedPolicy::WorkspaceWrite,
        },
    );
    config
        .writable_roots
        .push(std::path::PathBuf::from("/tmp/quote\"root"));
    let profile = build_seatbelt_profile(&config, workspace.as_path()).expect("profile");
    assert!(profile.contains("quote\\\"root"));
    assert!(!profile.contains("(allow default)"));
    assert!(
        !profile
            .lines()
            .any(|line| line.trim() == "(allow file-read*)")
    );
}

#[tokio::test]
async fn exec_echo_ok() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let workspace = dir.path().to_path_buf();
    let config = policy_to_config(
        NamedPolicy::ReadOnly,
        &workspace,
        None,
        SandboxBackendType::MacOSSeatbelt {
            policy: NamedPolicy::ReadOnly,
        },
    );
    let backend = MacOSSeatbeltBackend;
    let instance = backend
        .create(&config, workspace.as_path())
        .await
        .expect("seatbelt instance");
    let result = instance
        .exec(&["sh", "-c", "echo ok"], None, None, Duration::from_secs(5))
        .await
        .expect("exec");
    assert_eq!(
        result.exit_code, 0,
        "echo failed: stdout={:?} stderr={:?}",
        result.stdout, result.stderr
    );
    assert!(result.stdout.trim().contains("ok"));
}

#[tokio::test]
async fn exec_cannot_read_file_outside_workspace() {
    let workspace = tempfile::TempDir::new().expect("workspace");
    let secret_dir = tempfile::TempDir::new().expect("secret dir");
    let secret_path = secret_dir.path().join("secret.txt");
    std::fs::write(&secret_path, "leaked").expect("write secret");

    let config = policy_to_config(
        NamedPolicy::ReadOnly,
        workspace.path(),
        None,
        SandboxBackendType::MacOSSeatbelt {
            policy: NamedPolicy::ReadOnly,
        },
    );
    let backend = MacOSSeatbeltBackend;
    let instance = backend
        .create(&config, workspace.path())
        .await
        .expect("seatbelt instance");
    let script = format!("cat '{}'", secret_path.display());
    let result = instance
        .exec(&["sh", "-c", &script], None, None, Duration::from_secs(5))
        .await
        .expect("exec");
    assert_ne!(
        result.exit_code, 0,
        "sandbox must not read files outside workspace; stdout={} stderr={}",
        result.stdout, result.stderr
    );
    assert!(
        !result.stdout.contains("leaked"),
        "secret contents leaked via exec stdout: {}",
        result.stdout
    );
}

#[tokio::test]
async fn exec_never_runs_a_sandbox_exec_planted_on_its_path() {
    use std::os::unix::fs::PermissionsExt;

    let workspace = tempfile::TempDir::new().expect("workspace");
    let secret_dir = tempfile::TempDir::new().expect("secret dir");
    let secret_path = secret_dir.path().join("secret.txt");
    std::fs::write(&secret_path, "leaked").expect("write secret");
    // The planted tool leaves a marker, drops `-f <profile> --`, and runs the
    // command with no sandbox.
    let planted = tempfile::TempDir::new().expect("planted dir");
    let marker = planted.path().join("ran");
    let fake = planted.path().join("sandbox-exec");
    std::fs::write(
        &fake,
        format!(
            "#!/bin/sh\n/usr/bin/touch '{}'\nshift 3\nexec \"$@\"\n",
            marker.display()
        ),
    )
    .expect("write planted sandbox-exec");
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).expect("chmod");

    let config = policy_to_config(
        NamedPolicy::ReadOnly,
        workspace.path(),
        None,
        SandboxBackendType::MacOSSeatbelt {
            policy: NamedPolicy::ReadOnly,
        },
    );
    let instance = MacOSSeatbeltBackend
        .create(&config, workspace.path())
        .await
        .expect("seatbelt instance");
    let env = std::collections::HashMap::from([(
        "PATH".to_string(),
        format!("{}:/usr/bin:/bin", planted.path().display()),
    )]);
    let script = format!("cat '{}'", secret_path.display());
    let result = instance
        .exec(
            &["/bin/sh", "-c", &script],
            None,
            Some(&env),
            Duration::from_secs(5),
        )
        .await
        .expect("exec");
    assert_eq!(
        (marker.exists(), result.stdout.contains("leaked")),
        (false, false),
        "planted sandbox-exec ran or the read escaped; stdout={} stderr={}",
        result.stdout,
        result.stderr
    );
    assert_ne!(result.exit_code, 0, "the read must be denied");
}

/// The python3 behind the `/usr/bin/python3` shim lives in the selected
/// developer directory. With Xcode.app selected, that directory must be
/// readable or the interpreter cannot load its own framework.
#[tokio::test]
async fn exec_runs_the_selected_developer_python3() {
    let found = std::process::Command::new("/usr/bin/xcrun")
        .args(["--find", "python3"])
        .env_clear()
        .output()
        .ok()
        .filter(|output| output.status.success());
    let Some(found) = found else {
        eprintln!("skipped: no developer tools with python3 are selected");
        return;
    };
    let python3 = String::from_utf8_lossy(&found.stdout).trim().to_string();
    let workspace = tempfile::TempDir::new().expect("workspace");
    let config = policy_to_config(
        NamedPolicy::ReadOnly,
        workspace.path(),
        None,
        SandboxBackendType::MacOSSeatbelt {
            policy: NamedPolicy::ReadOnly,
        },
    );
    let instance = MacOSSeatbeltBackend
        .create(&config, workspace.path())
        .await
        .expect("seatbelt instance");
    let result = instance
        .exec(
            &[&python3, "-I", "-c", "print(2 + 2)"],
            None,
            None,
            Duration::from_secs(30),
        )
        .await
        .expect("exec");
    assert_eq!(
        (result.exit_code, result.stdout.as_str()),
        (0, "4\n"),
        "{python3}: stderr={}",
        result.stderr
    );
}

/// Run `command` on its own thread and fail if exec does not return within
/// `timeout` plus a guard, so a stalled exec fails instead of hanging.
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "test helper panics with context"
)]
fn exec_guarded(
    command: &[&str],
    timeout: Duration,
) -> (Duration, anyhow::Result<caliche_sandbox::ExecResult>) {
    let owned: Vec<String> = command.iter().map(|arg| (*arg).to_string()).collect();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let command: Vec<&str> = owned.iter().map(String::as_str).collect();
        let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
        let outcome = runtime.block_on(async {
            let dir = tempfile::TempDir::new().expect("tempdir");
            let workspace = dir.path().to_path_buf();
            let config = policy_to_config(
                NamedPolicy::ReadOnly,
                &workspace,
                None,
                SandboxBackendType::MacOSSeatbelt {
                    policy: NamedPolicy::ReadOnly,
                },
            );
            let instance = MacOSSeatbeltBackend
                .create(&config, workspace.as_path())
                .await
                .expect("instance");
            let started = std::time::Instant::now();
            let result = instance.exec(&command, None, None, timeout).await;
            (started.elapsed(), result)
        });
        let _ = tx.send(outcome);
    });
    let guard = timeout + Duration::from_secs(30);
    rx.recv_timeout(guard).unwrap_or_else(|_| {
        panic!("exec did not return within {guard:?}; output drain is unbounded")
    })
}

#[test]
fn output_flood_is_capped_and_exec_returns_at_its_timeout() {
    let timeout = Duration::from_secs(4);
    let (elapsed, result) = exec_guarded(
        &["/bin/sh", "-c", "echo err >&2; exec /usr/bin/yes"],
        timeout,
    );

    assert!(
        elapsed < timeout + Duration::from_secs(2),
        "exec took {elapsed:?} with a {timeout:?} timeout"
    );
    let err = result.expect_err("yes must time out");
    let timed_out = err
        .downcast_ref::<caliche_sandbox::ExecTimeout>()
        .unwrap_or_else(|| panic!("expected ExecTimeout, got {err:#}"));
    assert_eq!(timed_out.stdout.len(), caliche_sandbox::EXEC_OUTPUT_LIMIT);
    assert!(
        timed_out.stdout_truncated,
        "stdout past the cap must be flagged"
    );
    assert_eq!(
        (timed_out.stderr.as_str(), timed_out.stderr_truncated),
        ("err\n", false)
    );
}

#[test]
fn output_under_the_cap_is_returned_in_full() {
    let (_, result) = exec_guarded(
        &["/bin/sh", "-c", "printf %01000d 0; echo done >&2"],
        Duration::from_secs(10),
    );

    let result = result.expect("exec");
    assert_eq!(result.stdout, "0".repeat(1000));
    assert_eq!(
        (
            result.stdout_truncated,
            result.stderr.as_str(),
            result.stderr_truncated
        ),
        (false, "done\n", false)
    );
}

#[tokio::test]
async fn exec_cannot_write_the_workspace_git_dir() {
    let workspace = tempfile::TempDir::new().expect("workspace");
    let root = workspace
        .path()
        .canonicalize()
        .expect("canonical workspace");
    git_shapes::git_dir_at(&root.join(GIT_DIR_NAME));
    std::fs::create_dir_all(root.join("sub")).expect("sub dir");
    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &root,
        None,
        SandboxBackendType::MacOSSeatbelt {
            policy: NamedPolicy::WorkspaceWrite,
        },
    );
    let instance = MacOSSeatbeltBackend
        .create(&config, &root)
        .await
        .expect("create");
    let script = "for p in .git/config .GIT/hooks sub/.git; do \
                  (mkdir -p \"$p\" 2>/dev/null || : > \"$p\") 2>/dev/null \
                  && printf 'wrote:%s ' \"$p\"; done; \
                  mv .git moved 2>/dev/null && printf 'renamed '; \
                  : > ok.txt && : > .gitignore && printf 'allowed'";
    let result = instance
        .exec(
            &["sh", "-c", script],
            Some(root.to_str().expect("utf-8 workspace")),
            None,
            Duration::from_secs(5),
        )
        .await
        .expect("exec");
    assert_eq!(result.stdout, "allowed", "stderr={}", result.stderr);
    assert!(!root.join(GIT_DIR_NAME).join(CONFIG_FILE).exists());
    assert!(root.join(GIT_DIR_NAME).is_dir());
}

/// With a writable root around the workspace, a command cannot move the
/// workspace (or a parent) aside, write `.git` under the new name, and
/// move it back.
#[tokio::test]
async fn exec_cannot_move_the_workspace_aside() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let base = dir.path().canonicalize().expect("canonical");
    let root = base.join("parent").join("ws");
    git_shapes::git_dir_at(&root.join(GIT_DIR_NAME));
    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &base,
        None,
        SandboxBackendType::MacOSSeatbelt {
            policy: NamedPolicy::WorkspaceWrite,
        },
    );
    let instance = MacOSSeatbeltBackend
        .create(&config, &root)
        .await
        .expect("create");
    let script = "mv parent/ws parent/old 2>/dev/null && printf 'workspace '; \
                  mv parent other 2>/dev/null && printf 'parent '; \
                  : > parent/ws/ok.txt && mkdir parent/ws/d && printf 'allowed'";
    let result = instance
        .exec(
            &["sh", "-c", script],
            Some(base.to_str().expect("utf-8 base")),
            None,
            Duration::from_secs(5),
        )
        .await
        .expect("exec");
    assert_eq!(result.stdout, "allowed", "stderr={}", result.stderr);
    assert!(root.join(GIT_DIR_NAME).is_dir());
}

/// `git init --separate-git-dir <ws>/meta <ws>`: the profile denies only
/// `.git`, so a git dir under another name is refused at create.
#[tokio::test]
async fn create_refuses_a_git_dir_inside_the_workspace_outside_git() {
    let dir = tempfile::TempDir::new().expect("workspace");
    let root = dir.path().canonicalize().expect("canonical");
    std::fs::create_dir(root.join("meta")).expect("meta");
    git_shapes::git_dir_at(&root.join("meta"));
    std::fs::write(root.join(GIT_DIR_NAME), format!("{GITFILE_PREFIX} meta\n")).expect("gitfile");
    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &root,
        None,
        SandboxBackendType::MacOSSeatbelt {
            policy: NamedPolicy::WorkspaceWrite,
        },
    );

    let err = MacOSSeatbeltBackend
        .create(&config, &root)
        .await
        .err()
        .expect("refused");

    assert_eq!(
        err.to_string(),
        git_refusal::git_dir_in_workspace(&root.join("meta"))
    );
}

/// A linked worktree whose main repository is under a writable root: the
/// sandbox could plant config in the worktree git dir it shares.
#[tokio::test]
async fn create_refuses_a_worktree_git_dir_under_a_writable_root() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let base = dir.path().canonicalize().expect("canonical");
    let root = base.join("ws");
    let main = base.join("main");
    std::fs::create_dir_all(&root).expect("ws");
    let git_dir = plant_linked_worktree(&main.join(GIT_DIR_NAME), "ws").expect("linked worktree");
    std::fs::write(
        root.join(GIT_DIR_NAME),
        format!("{GITFILE_PREFIX} {}\n", git_dir.display()),
    )
    .expect("gitfile");
    let mut config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &root,
        None,
        SandboxBackendType::MacOSSeatbelt {
            policy: NamedPolicy::WorkspaceWrite,
        },
    );
    config.writable_roots.push(main.clone());

    let err = MacOSSeatbeltBackend
        .create(&config, &root)
        .await
        .err()
        .expect("refused");

    assert_eq!(
        err.to_string(),
        git_refusal::git_dir_under_writable_root(&git_dir, &main)
    );
}

/// Runs host `git` for a test fixture, without the user's config.
#[expect(clippy::expect_used, reason = "test fixture")]
fn git(dir: &std::path::Path, args: &[&str]) {
    let mut command = std::process::Command::new("git");
    let status = clear_git_config(&mut command)
        .args([
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "protocol.file.allow=always",
        ])
        .args(args)
        .current_dir(dir)
        .status()
        .expect("host git must be installed to run these tests");
    assert!(status.success(), "git {args:?}");
}

/// A checked-out submodule cannot be moved aside, given a gitfile that
/// points elsewhere, and moved back, even when a writable root holds the
/// workspace. Writes inside the submodule still work.
#[tokio::test]
async fn exec_cannot_swap_a_submodule_gitfile() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let base = dir.path().canonicalize().expect("canonical");
    let origin = base.join("origin");
    let root = base.join("parent").join("ws");
    std::fs::create_dir_all(&origin).expect("origin");
    std::fs::create_dir_all(&root).expect("ws");
    git(&origin, &["init", "-q"]);
    git(&origin, &["commit", "-q", "--allow-empty", "-m", "init"]);
    git(&root, &["init", "-q"]);
    git(
        &root,
        &[
            "submodule",
            "add",
            "-q",
            origin.to_str().expect("utf-8 origin"),
            "sub",
        ],
    );
    let gitfile = std::fs::read_to_string(root.join("sub").join(GIT_DIR_NAME)).expect("gitfile");
    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &base,
        None,
        SandboxBackendType::MacOSSeatbelt {
            policy: NamedPolicy::WorkspaceWrite,
        },
    );
    let instance = MacOSSeatbeltBackend
        .create(&config, &root)
        .await
        .expect("create");
    let script = format!(
        "mv parent/ws/sub moved 2>/dev/null && printf 'moved '; \
         [ -d moved ] && {{ printf '{GITFILE_PREFIX} /tmp/evil\\n' > moved/{GIT_DIR_NAME} 2>/dev/null && printf 'rewritten '; mv moved parent/ws/sub 2>/dev/null; }}; \
         : > parent/ws/sub/ok.txt && printf 'allowed'"
    );
    let result = instance
        .exec(
            &["sh", "-c", &script],
            Some(base.to_str().expect("utf-8 base")),
            None,
            Duration::from_secs(5),
        )
        .await
        .expect("exec");
    assert_eq!(result.stdout, "allowed", "stderr={}", result.stderr);
    assert_eq!(
        std::fs::read_to_string(root.join("sub").join(GIT_DIR_NAME)).expect("gitfile"),
        gitfile
    );
}

/// A workspace without `.git` cannot get one, in itself or in a parent
/// that a writable root holds, so host git never discovers a planted
/// repository.
#[tokio::test]
async fn exec_cannot_create_a_git_dir_for_a_workspace_without_one() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let base = dir.path().canonicalize().expect("canonical");
    let root = base.join("parent").join("ws");
    std::fs::create_dir_all(&root).expect("ws");
    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &base,
        None,
        SandboxBackendType::MacOSSeatbelt {
            policy: NamedPolicy::WorkspaceWrite,
        },
    );
    let instance = MacOSSeatbeltBackend
        .create(&config, &root)
        .await
        .expect("create");
    let script = format!(
        "mkdir parent/{GIT_DIR_NAME} 2>/dev/null && printf 'parent '; \
         mkdir parent/.GIT 2>/dev/null && printf 'parent-case '; \
         mkdir parent/ws/{GIT_DIR_NAME} 2>/dev/null && printf 'workspace '; \
         mkdir d && mv d parent/{GIT_DIR_NAME} 2>/dev/null && printf 'renamed '; \
         : > parent/ws/ok.txt && printf 'allowed'"
    );
    let result = instance
        .exec(
            &["sh", "-c", &script],
            Some(base.to_str().expect("utf-8 base")),
            None,
            Duration::from_secs(5),
        )
        .await
        .expect("exec");
    assert_eq!(result.stdout, "allowed", "stderr={}", result.stderr);
    assert!(!base.join("parent").join(GIT_DIR_NAME).exists());
    assert!(!root.join(GIT_DIR_NAME).exists());
}

/// A writable root inside a linked worktree's common dir (here the main
/// repository's hooks) is refused.
#[tokio::test]
async fn create_refuses_a_writable_root_inside_the_common_dir() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let base = dir.path().canonicalize().expect("canonical");
    let root = base.join("ws");
    let common = base.join("main").join(GIT_DIR_NAME);
    let hooks = common.join("hooks");
    std::fs::create_dir_all(&root).expect("ws");
    let git_dir = plant_linked_worktree(&common, "ws").expect("linked worktree");
    std::fs::create_dir_all(&hooks).expect("hooks");
    std::fs::write(
        root.join(GIT_DIR_NAME),
        format!("{GITFILE_PREFIX} {}\n", git_dir.display()),
    )
    .expect("gitfile");
    let mut config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &root,
        None,
        SandboxBackendType::MacOSSeatbelt {
            policy: NamedPolicy::WorkspaceWrite,
        },
    );
    config.writable_roots.push(hooks.clone());

    let err = MacOSSeatbeltBackend
        .create(&config, &root)
        .await
        .err()
        .expect("refused");

    assert_eq!(
        err.to_string(),
        git_refusal::overlapping_root(&hooks, &common)
    );
}

/// A superproject at `<base>/parent/ws` with submodules `full` (checked
/// out), `empty` (deinit: an empty directory) and `gone` (deinit, then the
/// directory removed), from a repository at `<base>/origin`.
#[expect(clippy::expect_used, reason = "test fixture")]
fn submodule_fixture(base: &std::path::Path) -> std::path::PathBuf {
    let origin = base.join("origin");
    let root = base.join("parent").join("ws");
    std::fs::create_dir_all(&origin).expect("origin");
    std::fs::create_dir_all(&root).expect("ws");
    git(&origin, &["init", "-q"]);
    git(&origin, &["commit", "-q", "--allow-empty", "-m", "init"]);
    git(&root, &["init", "-q"]);
    let url = origin.to_str().expect("utf-8 origin");
    for name in ["full", "empty", "gone"] {
        git(&root, &["submodule", "add", "-q", url, name]);
    }
    git(&root, &["submodule", "deinit", "-q", "-f", "empty", "gone"]);
    std::fs::remove_dir(root.join("gone")).expect("remove gone");
    root
}

fn writable_base_config(base: &std::path::Path) -> caliche_sandbox::SandboxConfig {
    policy_to_config(
        NamedPolicy::WorkspaceWrite,
        base,
        None,
        SandboxBackendType::MacOSSeatbelt {
            policy: NamedPolicy::WorkspaceWrite,
        },
    )
}

/// A submodule that is not checked out cannot be filled in from outside:
/// its missing directory cannot be made or moved in, and its empty one
/// cannot be removed or replaced. Writes inside the empty one still work.
#[tokio::test]
async fn exec_cannot_fill_in_a_submodule_that_is_not_checked_out() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let base = dir.path().canonicalize().expect("canonical");
    let root = submodule_fixture(&base);
    let instance = MacOSSeatbeltBackend
        .create(&writable_base_config(&base), &root)
        .await
        .expect("create");
    let script = format!(
        "mkdir m1 m2 && printf '{GITFILE_PREFIX} /tmp/evil\\n' > m1/{GIT_DIR_NAME} && cp m1/{GIT_DIR_NAME} m2/; \
         mkdir parent/ws/gone 2>/dev/null && printf 'gone-made '; \
         mv m1 parent/ws/gone 2>/dev/null && printf 'gone-moved '; \
         rmdir parent/ws/empty 2>/dev/null && printf 'empty-removed '; \
         perl -e 'print rename(\"m2\", \"parent/ws/empty\") ? \"empty-replaced \" : \"replace-denied \"'; \
         : > parent/ws/empty/ok.txt && printf 'allowed'"
    );
    let result = instance
        .exec(
            &["sh", "-c", &script],
            Some(base.to_str().expect("utf-8 base")),
            None,
            Duration::from_secs(5),
        )
        .await
        .expect("exec");
    assert_eq!(
        result.stdout, "replace-denied allowed",
        "stderr={}",
        result.stderr
    );
    assert!(!root.join("gone").exists());
    assert!(!root.join("empty").join(GIT_DIR_NAME).exists());
}

/// A symlinked directory on the path to a submodule is refused at create:
/// a deny on the path would not cover where the kernel resolves it.
#[cfg(unix)]
#[tokio::test]
async fn create_refuses_a_symlink_on_the_path_to_a_submodule() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let base = dir.path().canonicalize().expect("canonical");
    let origin = base.join("origin");
    let root = base.join("ws");
    std::fs::create_dir_all(&origin).expect("origin");
    std::fs::create_dir_all(&root).expect("ws");
    git(&origin, &["init", "-q"]);
    git(&origin, &["commit", "-q", "--allow-empty", "-m", "init"]);
    git(&root, &["init", "-q"]);
    git(
        &root,
        &[
            "submodule",
            "add",
            "-q",
            origin.to_str().expect("utf-8 origin"),
            "libs/a",
        ],
    );
    std::fs::rename(root.join("libs"), base.join("real-libs")).expect("move libs");
    std::os::unix::fs::symlink(base.join("real-libs"), root.join("libs")).expect("symlink");

    let err = MacOSSeatbeltBackend
        .create(&writable_base_config(&root), &root)
        .await
        .err()
        .expect("refused");

    assert_eq!(
        err.to_string(),
        git_refusal::symlink_leads_to_git(std::path::Path::new("libs"))
    );
}

/// `path` through the `/var` or `/tmp` alias when it is under `/private`.
fn alias(path: &std::path::Path) -> std::path::PathBuf {
    path.strip_prefix("/private").map_or_else(
        |_| path.to_path_buf(),
        |rest| std::path::Path::new("/").join(rest),
    )
}

/// A superproject at `<base>/parent/ws` with a checked-out submodule at
/// `libs/a` and a submodule at `mods/b` whose directories do not exist.
#[expect(clippy::expect_used, reason = "test fixture")]
fn attack_fixture(base: &std::path::Path) -> std::path::PathBuf {
    let origin = base.join("origin");
    let root = base.join("parent").join("ws");
    std::fs::create_dir_all(&origin).expect("origin");
    std::fs::create_dir_all(&root).expect("ws");
    git(&origin, &["init", "-q"]);
    git(&origin, &["commit", "-q", "--allow-empty", "-m", "init"]);
    git(&root, &["init", "-q"]);
    let url = origin.to_str().expect("utf-8 origin");
    git(&root, &["submodule", "add", "-q", url, "libs/a"]);
    git(&root, &["submodule", "add", "-q", url, "mods/b"]);
    git(&root, &["submodule", "deinit", "-q", "-f", "mods/b"]);
    std::fs::remove_dir(root.join("mods").join("b")).expect("remove mods/b");
    std::fs::remove_dir(root.join("mods")).expect("remove mods");
    std::fs::write(root.join("libs").join("a").join("f"), "file").expect("libs/a/f");
    root
}

/// Runs `script` from `<base>` in a sandbox on `workspace` with `<base>`
/// writable, and returns its stdout.
#[expect(clippy::expect_used, reason = "test helper")]
async fn run_attack(base: &std::path::Path, workspace: &std::path::Path, script: &str) -> String {
    let instance = MacOSSeatbeltBackend
        .create(&writable_base_config(base), workspace)
        .await
        .expect("create");
    let result = instance
        .exec(
            &["sh", "-c", script],
            Some(base.to_str().expect("utf-8 base")),
            None,
            Duration::from_secs(5),
        )
        .await
        .expect("exec");
    result.stdout
}

/// Each attack on a submodule or the workspace `.git`, with a writable
/// root around the workspace, runs in its own sandbox on a fresh fixture.
/// Each prints its name only if it succeeds; none may.
#[tokio::test]
async fn every_attack_on_submodule_metadata_is_denied() {
    let evil = format!("{GITFILE_PREFIX} /tmp/evil\\n");
    let attacks: Vec<(&str, String)> = vec![
        ("move the submodule out", "mv parent/ws/libs/a moved".into()),
        (
            "move the directory above it out",
            "mv parent/ws/libs moved".into(),
        ),
        (
            "make the missing submodule directory",
            "mkdir parent/ws/mods".into(),
        ),
        (
            "move a prepared submodule in",
            format!("mkdir -p x/b && printf '{evil}' > x/b/{GIT_DIR_NAME} && mv x parent/ws/mods"),
        ),
        (
            "hard-link the gitfile out",
            format!("ln parent/ws/libs/a/{GIT_DIR_NAME} l"),
        ),
        (
            "hard-link the config out",
            format!("ln parent/ws/{GIT_DIR_NAME}/config l"),
        ),
        (
            "put a symlink over the gitfile",
            format!("ln -sf /tmp/evil parent/ws/libs/a/{GIT_DIR_NAME}"),
        ),
        (
            "remove the directory above it",
            "rm -rf parent/ws/libs".into(),
        ),
        (
            "move the submodule out through an alias",
            "mv \"$ALIAS/parent/ws/libs/a\" moved".into(),
        ),
        (
            "move the workspace .git out",
            format!("mv parent/ws/{GIT_DIR_NAME} moved"),
        ),
        (
            "hard-link a file over the gitfile",
            format!("printf '{evil}' > f && ln -f f parent/ws/libs/a/{GIT_DIR_NAME}"),
        ),
        (
            "clone a file over the gitfile",
            format!("printf '{evil}' > f && cp -c f parent/ws/libs/a/{GIT_DIR_NAME}"),
        ),
    ];
    let mut succeeded = Vec::new();
    for (name, attack) in &attacks {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let base = dir.path().canonicalize().expect("canonical");
        let root = attack_fixture(&base);
        let gitfile = root.join("libs").join("a").join(GIT_DIR_NAME);
        let before = std::fs::read(&gitfile).expect("gitfile");
        let script = format!(
            "ALIAS='{}'; {{ {attack}; }} 2>/dev/null && printf ok",
            alias(&base).display()
        );

        let stdout = run_attack(&base, &root, &script).await;

        let intact = std::fs::read(&gitfile).ok().as_ref() == Some(&before)
            && root.join(GIT_DIR_NAME).is_dir()
            && !root.join("mods").exists();
        if stdout == "ok" || !intact {
            succeeded.push(*name);
        }
    }
    assert_eq!(succeeded, Vec::<&str>::new());
}

/// Each attempt to give a workspace without `.git` one, in itself or a
/// parent, with a writable root around it, runs in its own sandbox. Each
/// prints its name only if it succeeds; none may.
#[tokio::test]
async fn every_attempt_to_plant_a_git_dir_is_denied() {
    let attacks: Vec<(&str, String)> = vec![
        (
            "make it in the parent",
            format!("mkdir parent/{GIT_DIR_NAME}"),
        ),
        (
            "create a file at it",
            format!("touch parent/{GIT_DIR_NAME}"),
        ),
        (
            "rename a directory onto it",
            format!("mkdir d && mv d parent/{GIT_DIR_NAME}"),
        ),
        (
            "put a symlink at it",
            format!("ln -s /tmp/evil parent/{GIT_DIR_NAME}"),
        ),
        ("make it in another case", "mkdir parent/.GIT".into()),
        (
            "hard-link a file to it",
            format!("touch f && ln f parent/{GIT_DIR_NAME}"),
        ),
        (
            "make it in a higher parent",
            format!("mkdir {GIT_DIR_NAME}"),
        ),
        (
            "make it in the workspace",
            format!("mkdir parent/ws/{GIT_DIR_NAME}"),
        ),
        (
            "make it through an alias",
            format!("mkdir \"$ALIAS/parent/{GIT_DIR_NAME}\""),
        ),
    ];
    let mut succeeded = Vec::new();
    for (name, attack) in &attacks {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let base = dir.path().canonicalize().expect("canonical");
        let root = base.join("parent").join("ws");
        std::fs::create_dir_all(&root).expect("ws");
        let script = format!(
            "ALIAS='{}'; {{ {attack}; }} 2>/dev/null && printf ok",
            alias(&base).display()
        );

        let stdout = run_attack(&base, &root, &script).await;

        let planted = [
            base.join(GIT_DIR_NAME),
            base.join("parent").join(GIT_DIR_NAME),
            root.join(GIT_DIR_NAME),
        ]
        .iter()
        .any(|path| path.symlink_metadata().is_ok());
        if stdout == "ok" || planted {
            succeeded.push(*name);
        }
    }
    assert_eq!(succeeded, Vec::<&str>::new());
}

include!("macos_parts/planted.rs");

/// A package manager can copy a submodule checkout, gitfile and all, into
/// build output: `node_modules/pkg/.git` then names
/// `node_modules/.git/modules/pkg`, which does not exist. Host `git` in the
/// workspace root never reads it, and the profile denies creating that
/// path, so the dangling gitfile does not stop the sandbox.
#[tokio::test]
async fn create_accepts_a_dangling_gitfile_in_build_output() {
    let workspace = tempfile::TempDir::new().expect("workspace");
    let root = workspace
        .path()
        .canonicalize()
        .expect("canonical workspace");
    git_shapes::git_dir_at(&root.join(GIT_DIR_NAME));
    let pkg = root.join("node_modules").join("pkg");
    std::fs::create_dir_all(&pkg).expect("pkg dir");
    std::fs::write(
        pkg.join(GIT_DIR_NAME),
        format!("{GITFILE_PREFIX} ../{GIT_DIR_NAME}/modules/pkg\n"),
    )
    .expect("dangling gitfile");
    let config = writable_base_config(&root);

    let instance = MacOSSeatbeltBackend
        .create(&config, &root)
        .await
        .expect("create");

    let script = format!(
        "mkdir -p node_modules/{GIT_DIR_NAME}/modules/pkg 2>/dev/null && printf 'made '; \
         : > node_modules/ok.txt && printf 'allowed'"
    );
    let result = instance
        .exec(
            &["sh", "-c", &script],
            Some(root.to_str().expect("utf-8 workspace")),
            None,
            Duration::from_secs(5),
        )
        .await
        .expect("exec");
    assert_eq!(result.stdout, "allowed", "stderr={}", result.stderr);
    assert!(!root.join("node_modules").join(GIT_DIR_NAME).exists());
}

/// A directory the user cannot list (macOS privacy protection on
/// `~/.Trash`, or plain mode 000) does not stop the sandbox: host `git` in
/// the workspace root reads only the root `.git` and the submodules in its
/// index, and the profile denies a `.git` at any depth.
#[tokio::test]
async fn create_accepts_a_directory_it_cannot_list() {
    use std::os::unix::fs::PermissionsExt;

    let workspace = tempfile::TempDir::new().expect("workspace");
    let root = workspace
        .path()
        .canonicalize()
        .expect("canonical workspace");
    git_shapes::git_dir_at(&root.join(GIT_DIR_NAME));
    let locked = root.join("locked");
    std::fs::create_dir(&locked).expect("locked dir");
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).expect("chmod 000");
    let config = writable_base_config(&root);

    let created = MacOSSeatbeltBackend.create(&config, &root).await;
    let result = match &created {
        Ok(instance) => instance
            .exec(
                &["sh", "-c", ": > ok.txt && printf 'allowed'"],
                Some(root.to_str().expect("utf-8 workspace")),
                None,
                Duration::from_secs(5),
            )
            .await
            .map(|result| result.stdout),
        Err(_) => Ok(String::new()),
    };
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).expect("chmod back");

    let instance_err = created.err().map(|err| err.to_string());
    assert_eq!(instance_err, None);
    assert_eq!(result.expect("exec"), "allowed");
}

/// The macOS backend does no nested git scan: a tree past the Linux and
/// Docker scan's depth limit and a directory nobody can list do not stop
/// it, and the profile's name rule still denies a write to a nested
/// repository's hooks.
#[tokio::test]
async fn create_does_no_nested_git_scan() {
    use std::os::unix::fs::PermissionsExt;

    let workspace = tempfile::TempDir::new().expect("workspace");
    let root = workspace
        .path()
        .canonicalize()
        .expect("canonical workspace");
    git_shapes::git_dir_at(&root.join(GIT_DIR_NAME));
    let nested = root.join("vendor").join("dep").join(GIT_DIR_NAME);
    git_shapes::git_dir_at(&nested);
    std::fs::create_dir_all(nested.join("hooks")).expect("hooks dir");
    let mut deep = root.join("deep");
    for _ in 0..65 {
        deep.push("d");
    }
    std::fs::create_dir_all(&deep).expect("deep dirs");
    let locked = root.join("locked");
    std::fs::create_dir(&locked).expect("locked dir");
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).expect("chmod 000");
    let config = writable_base_config(&root);

    let created = MacOSSeatbeltBackend.create(&config, &root).await;
    let result = match &created {
        Ok(instance) => instance
            .exec(
                &[
                    "sh",
                    "-c",
                    "(: > vendor/dep/.git/hooks/post-checkout) 2>/dev/null && printf 'hook '; \
                     : > ok.txt && printf 'allowed'",
                ],
                Some(root.to_str().expect("utf-8 workspace")),
                None,
                Duration::from_secs(5),
            )
            .await
            .map(|result| result.stdout),
        Err(_) => Ok(String::new()),
    };
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).expect("chmod back");

    let instance_err = created.err().map(|err| err.to_string());
    assert_eq!(instance_err, None);
    assert_eq!(result.expect("exec"), "allowed");
    assert!(!nested.join("hooks").join("post-checkout").exists());
}

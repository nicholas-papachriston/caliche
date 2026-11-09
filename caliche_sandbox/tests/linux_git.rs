#![cfg(all(feature = "sandbox-linux", target_os = "linux"))]
//! The workspace `.git` stays read-only in both Linux modes.
//!
//! `CLS_EXPECT_GIT_GUARD` names the mode the host must pick
//! (`mount_namespace` or `top_level_grants`), so a run meant for one mode
//! cannot pass in the other.

use caliche_sandbox::paths::{
    CONFIG_FILE, GIT_DIR_NAME, GITFILE_PREFIX, GITLINK_MODE, GitGuardMode, HEAD_FILE, INDEX_FILE,
    MODULES_DIR, git_refusal,
};
use caliche_sandbox::traits::SandboxInstance;
use caliche_sandbox::{
    DeferredInstance, NamedPolicy, SANDBOX_UNAVAILABLE, SandboxBackendType, SandboxConfig,
    create_backend, policy_to_config,
};
use std::fs;
use std::path::Path;
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

include!("common/mod.rs");

mod git_shapes;

const EXPECT_GIT_GUARD: &str = "CLS_EXPECT_GIT_GUARD";
const MOUNT_NAMESPACE: &str = GitGuardMode::MountNamespace.label();
const TOP_LEVEL_GRANTS: &str = GitGuardMode::TopLevelGrants.label();

fn lock_process_state() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|err| err.into_inner())
}

fn workspace_write(workspace: &Path) -> SandboxConfig {
    policy_to_config(
        NamedPolicy::WorkspaceWrite,
        workspace,
        None,
        SandboxBackendType::LinuxLandlock {
            policy: NamedPolicy::WorkspaceWrite,
        },
    )
}

#[expect(clippy::expect_used, reason = "test fixture")]
fn repo() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().expect("workspace");
    let root = dir.path().canonicalize().expect("canonical");
    let git = root.join(GIT_DIR_NAME);
    git_shapes::git_dir_at(&git);
    fs::create_dir_all(git.join("hooks")).expect("hooks dir");
    fs::write(git.join(CONFIG_FILE), b"[core]\n").expect("config");
    fs::create_dir(root.join("src")).expect("src");
    fs::write(root.join("top.txt"), b"top").expect("top file");
    (dir, root)
}

#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "test helper panics with context"
)]
fn run(root: &Path, script: &str, cwd: &Path) -> Option<String> {
    let config = workspace_write(root);
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    runtime.block_on(async {
        let backend = create_backend(&config).await.expect("landlock backend");
        let instance = match backend.create(&config, root).await {
            Ok(instance) => instance,
            Err(err) if err.to_string().contains("not available") => {
                require_landlock();
                return None;
            }
            Err(err) => panic!("landlock instance: {err:#}"),
        };
        let result = instance
            .exec(
                &["/bin/sh", "-c", script],
                Some(cwd.to_str().expect("utf-8")),
                None,
                Duration::from_secs(10),
            )
            .await
            .expect("exec");
        Some(result.stdout)
    })
}

/// The mode this host picked, read from whether a new top-level file can be
/// created, and checked against `CLS_EXPECT_GIT_GUARD` when it is set.
fn mode(root: &Path) -> Option<&'static str> {
    let created = run(
        root,
        "(: > .mode-probe) 2>/dev/null && rm .mode-probe && printf yes",
        root,
    )?;
    let mode = if created == "yes" {
        MOUNT_NAMESPACE
    } else {
        TOP_LEVEL_GRANTS
    };
    if let Ok(expected) = std::env::var(EXPECT_GIT_GUARD) {
        assert_eq!(mode, expected, "{EXPECT_GIT_GUARD} names another mode");
    }
    Some(mode)
}

#[test]
fn exec_cannot_change_the_workspace_git_dir() {
    let _process = lock_process_state();
    let (_dir, root) = repo();
    let script = "(: > .git/config) 2>/dev/null && printf 'config '; \
                  (: > .git/hooks/pre-commit) 2>/dev/null && printf 'hook '; \
                  mkdir .git/x 2>/dev/null && printf 'mkdir '; \
                  rm .git/HEAD 2>/dev/null && printf 'delete '; \
                  mv .git moved 2>/dev/null && printf 'rename '; \
                  (cd src && : > ../.git/via-parent) 2>/dev/null && printf 'parent '; \
                  ln -s ../.git src/g 2>/dev/null; (: > src/g/via-link) 2>/dev/null && printf 'link '; \
                  ln .git/config src/h 2>/dev/null && (echo x >> src/h) 2>/dev/null && printf 'hardlink '; \
                  : > src/ok.rs && : > top.txt && printf allowed";
    let Some(stdout) = run(&root, script, &root) else {
        return;
    };
    assert_eq!(stdout, "allowed");
    assert_eq!(
        fs::read(root.join(GIT_DIR_NAME).join(CONFIG_FILE)).expect("config"),
        b"[core]\n"
    );
    assert!(
        root.join(GIT_DIR_NAME).join(HEAD_FILE).is_file(),
        "HEAD was deleted"
    );
    assert!(!root.join(GIT_DIR_NAME).join("hooks/pre-commit").exists());
    assert!(!root.join(GIT_DIR_NAME).join("via-parent").exists());
    assert!(!root.join(GIT_DIR_NAME).join("via-link").exists());
    assert!(root.join("src/ok.rs").is_file(), "workspace write failed");
}

#[test]
fn exec_started_inside_git_cannot_write_there() {
    let _process = lock_process_state();
    let (_dir, root) = repo();
    let script = "(: > config) 2>/dev/null && printf wrote || printf refused";
    let Some(stdout) = run(&root, script, &root.join(GIT_DIR_NAME)) else {
        return;
    };
    assert_eq!(stdout, "refused");
    assert_eq!(
        fs::read(root.join(GIT_DIR_NAME).join(CONFIG_FILE)).expect("config"),
        b"[core]\n"
    );
}

/// Creating `.gitmodules` runs on a workspace without one; renaming and
/// deleting run after the host has written it.
#[test]
fn top_level_git_entries_follow_the_guard_mode() {
    let _process = lock_process_state();
    let (_dir, root) = repo();
    let Some(mode) = mode(&root) else {
        return;
    };
    let create = run(&root, "(: > .gitmodules) 2>/dev/null && printf ok", &root);
    fs::write(root.join(".gitmodules"), b"").expect("gitmodules");
    let rename = run(
        &root,
        "mv .gitmodules gm 2>/dev/null && mv gm .gitmodules && printf ok",
        &root,
    );
    let delete = run(&root, "rm .gitmodules 2>/dev/null && printf ok", &root);
    let git = run(
        &root,
        "{ mv .git moved || rm -r .git; } 2>/dev/null; [ -f .git/HEAD ] && printf kept",
        &root,
    );
    let outcome = [create, rename, delete, git].map(Option::unwrap_or_default);
    let expected = match mode {
        MOUNT_NAMESPACE => ["ok", "ok", "ok", "kept"],
        _ => ["", "", "", "kept"],
    };
    assert_eq!(outcome, expected, "mode {mode}");
}

/// A version 2 index whose only entries are submodules at `paths`.
fn index_with_submodules(paths: &[&str]) -> Vec<u8> {
    let mut data = b"DIRC".to_vec();
    data.extend_from_slice(&2_u32.to_be_bytes());
    data.extend_from_slice(&u32::try_from(paths.len()).unwrap_or(0).to_be_bytes());
    for path in paths {
        let start = data.len();
        data.extend_from_slice(&[0; 24]);
        data.extend_from_slice(&GITLINK_MODE.to_be_bytes());
        data.extend_from_slice(&[0; 12 + 20]);
        data.extend_from_slice(&u16::try_from(path.len()).unwrap_or(0).to_be_bytes());
        data.extend_from_slice(path.as_bytes());
        let padded = (data.len() - start + 8) & !7;
        data.resize(start + padded, 0);
    }
    data.extend_from_slice(&[0; 20]);
    data
}

fn gitfile(target: &str) -> String {
    format!("{GITFILE_PREFIX} {target}\n")
}

/// `repo()` plus a checked-out submodule at `libs/a` (a gitfile into
/// `.git/modules/a`) and one at `vendored` with its own `.git` dir.
#[expect(clippy::expect_used, reason = "test fixture")]
fn superproject() -> (tempfile::TempDir, std::path::PathBuf) {
    let (dir, root) = repo();
    let git = root.join(GIT_DIR_NAME);
    fs::create_dir_all(git.join(MODULES_DIR).join("a")).expect("modules");
    fs::write(
        git.join(MODULES_DIR).join("a").join(CONFIG_FILE),
        b"[core]\n",
    )
    .expect("module config");
    fs::write(
        git.join(INDEX_FILE),
        index_with_submodules(&["libs/a", "vendored"]),
    )
    .expect("index");
    fs::create_dir_all(root.join("libs/a/src")).expect("libs/a");
    fs::write(
        root.join("libs/a").join(GIT_DIR_NAME),
        gitfile(&format!("../../{GIT_DIR_NAME}/{MODULES_DIR}/a")),
    )
    .expect("gitfile");
    fs::write(root.join("libs/a/lib.rs"), b"lib").expect("lib.rs");
    let vendored = root.join("vendored").join(GIT_DIR_NAME);
    fs::create_dir_all(&vendored).expect("vendored");
    fs::write(vendored.join(CONFIG_FILE), b"[core]\n").expect("vendored config");
    (dir, root)
}

#[test]
fn exec_cannot_plant_submodule_or_nested_git_metadata() {
    let _process = lock_process_state();
    let (_dir, root) = superproject();
    let script = format!(
        "(printf '%s\\n' '{GITFILE_PREFIX} .evil' > libs/a/.git) 2>/dev/null && printf 'gitfile '; \
                  rm -f libs/a/.git 2>/dev/null; [ -f libs/a/.git ] || printf 'delete '; \
                  mv libs/a/.git libs/a/g 2>/dev/null && printf 'move '; \
                  mv libs/a libs/moved 2>/dev/null && printf 'rename-parent '; \
                  mv libs other 2>/dev/null && printf 'rename-grandparent '; \
                  (: > vendored/.git/config) 2>/dev/null && printf 'nested '; \
                  mv vendored v2 2>/dev/null && printf 'rename-nested '; \
                  : > libs/a/lib.rs && : > libs/a/src/new.rs && : > src/ok.rs && printf allowed"
    );
    let Some(stdout) = run(&root, &script, &root) else {
        return;
    };
    assert_eq!(stdout, "allowed");
    assert_eq!(
        fs::read_to_string(root.join("libs/a").join(GIT_DIR_NAME)).expect("gitfile"),
        gitfile(&format!("../../{GIT_DIR_NAME}/{MODULES_DIR}/a"))
    );
    assert_eq!(
        fs::read(root.join("vendored").join(GIT_DIR_NAME).join(CONFIG_FILE))
            .expect("nested config"),
        b"[core]\n"
    );
    assert!(
        root.join("libs/a/src/new.rs").is_file(),
        "submodule write failed"
    );
}

/// A workspace without a `.git`: the root gets no grant, so no command can
/// create one there (a directory, a gitfile, or a symlink).
#[test]
fn exec_cannot_create_a_root_git_in_a_workspace_without_one() {
    let _process = lock_process_state();
    let (_dir, root) = repo();
    fs::remove_dir_all(root.join(GIT_DIR_NAME)).expect("drop .git");
    let script = format!(
        "mkdir .git 2>/dev/null && printf 'dir '; \
         (printf '%s\\n' '{GITFILE_PREFIX} /tmp/evil' > .git) 2>/dev/null && printf 'gitfile '; \
         ln -s /tmp .git 2>/dev/null && printf 'symlink '; \
         (cd src && mkdir ../.git) 2>/dev/null && printf 'via-child '; \
         : > src/ok.rs && : > top.txt && printf allowed"
    );
    let Some(stdout) = run(&root, &script, &root) else {
        return;
    };
    assert_eq!(stdout, "allowed");
    assert!(
        fs::symlink_metadata(root.join(GIT_DIR_NAME)).is_err(),
        "a root .git was created"
    );
}

/// The error `create` returns for `root` under `config`, or `None` when
/// Landlock is not available here.
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "test helper panics with context"
)]
fn create_error_with(root: &Path, config: &SandboxConfig) -> Option<String> {
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    runtime.block_on(async {
        let backend = create_backend(config).await.expect("landlock backend");
        match backend.create(config, root).await {
            Ok(_) => panic!("an instance started in {}", root.display()),
            Err(err) if err.to_string().contains("not available") => {
                require_landlock();
                None
            }
            Err(err) => Some(format!("{err:#}")),
        }
    })
}

/// A submodule that is not checked out (`libs/b`) and a submodule's own
/// submodule (`libs/a/inner`, recorded in the module's index) are targets
/// read from the indexes. A command cannot create a `.git` at either path.
#[test]
fn exec_cannot_create_git_at_an_uninitialized_or_nested_submodule() {
    let _process = lock_process_state();
    let (_dir, root) = superproject();
    let git = root.join(GIT_DIR_NAME);
    fs::write(
        git.join(INDEX_FILE),
        index_with_submodules(&["libs/a", "libs/b", "vendored"]),
    )
    .expect("index");
    fs::create_dir(root.join("libs/b")).expect("libs/b");
    fs::write(
        git.join(MODULES_DIR).join("a").join(INDEX_FILE),
        index_with_submodules(&["inner"]),
    )
    .expect("module index");
    let script = "mkdir libs/b/.git 2>/dev/null && printf 'uninit '; \
                  mkdir libs/a/inner 2>/dev/null && printf 'inner-dir '; \
                  mkdir libs/a/inner/.git 2>/dev/null && printf 'inner '; \
                  : > libs/a/lib.rs && printf allowed";
    let Some(stdout) = run(&root, script, &root) else {
        return;
    };
    assert_eq!(stdout, "allowed");
    assert!(fs::symlink_metadata(root.join("libs/b").join(GIT_DIR_NAME)).is_err());
    assert!(fs::symlink_metadata(root.join("libs/a/inner")).is_err());
}

/// A workspace without `.git` inside a linked worktree, whose main
/// repository a writable root holds, is refused: host git would find the
/// worktree's git dir above the workspace and run hooks the sandbox wrote.
#[test]
fn a_workspace_in_a_worktree_of_a_writable_repository_is_refused() {
    let _process = lock_process_state();
    let dir = tempfile::tempdir().expect("base");
    let base = dir.path().canonicalize().expect("canonical");
    let main = base.join("m");
    let git_dir = caliche_policy::plant_linked_worktree(&main.join(GIT_DIR_NAME), "wt")
        .expect("linked worktree");
    let root = base.join("wt").join("sub");
    fs::create_dir_all(&root).expect("ws");
    fs::write(
        base.join("wt").join(GIT_DIR_NAME),
        format!("{GITFILE_PREFIX} {}\n", git_dir.display()),
    )
    .expect("gitfile");
    let mut config = workspace_write(&root);
    config.writable_roots.push(main.clone());

    let Some(err) = create_error_with(&root, &config) else {
        return;
    };

    assert_eq!(
        err,
        git_refusal::git_dir_under_writable_root(&git_dir, &main)
    );
}

/// A writable workspace below a repository's root is refused at create,
/// with the repository read-only: a submodule of the enclosing repository
/// under the workspace, not checked out, is listed only in the enclosing
/// index.
#[test]
fn create_refuses_a_workspace_inside_an_enclosing_repository() {
    let _process = lock_process_state();
    let dir = tempfile::tempdir().expect("base");
    let repo = dir.path().canonicalize().expect("canonical");
    git_shapes::git_dir_at(&repo.join(GIT_DIR_NAME));
    fs::write(
        repo.join(GIT_DIR_NAME).join(INDEX_FILE),
        index_with_submodules(&["ws/lib"]),
    )
    .expect("index");
    let ws = repo.join("ws");
    fs::create_dir_all(ws.join("lib")).expect("submodule dir");

    let Some(err) = create_error_with(&ws, &workspace_write(&ws)) else {
        return;
    };

    assert_eq!(err, git_refusal::enclosing_repository(&ws, &repo));
}

/// Every layout where host git would find a repository the sandbox could
/// write, or could finish planting one, is refused at create.
#[test]
fn create_refuses_every_planted_repository_shape() {
    let _process = lock_process_state();
    let dir = tempfile::tempdir().expect("base");
    let base = dir.path().canonicalize().expect("canonical");
    let mut wrong = Vec::new();

    for shape in git_shapes::planted_repository_shapes(&base) {
        let mut config = workspace_write(&shape.workspace);
        if shape.writable != shape.workspace {
            config.writable_roots.push(shape.writable.clone());
        }
        let Some(refused) = create_error_with(&shape.workspace, &config) else {
            return;
        };
        if refused != shape.refusal {
            wrong.push((shape.name, refused));
        }
    }

    assert_eq!(wrong, Vec::<(&str, String)>::new());
}

/// The guard resolves the workspace first, so one opened through a
/// symlink is checked against the repository git finds from where it
/// really is.
#[test]
fn a_workspace_opened_through_a_symlink_is_checked_where_it_resolves() {
    let _process = lock_process_state();
    let dir = tempfile::tempdir().expect("base");
    let base = dir.path().canonicalize().expect("canonical");
    let repo = base.join("repo");
    git_shapes::git_dir_at(&repo.join(GIT_DIR_NAME));
    let sub = repo.join("sub");
    fs::create_dir(&sub).expect("sub");
    let link = base.join("link");
    std::os::unix::fs::symlink(&sub, &link).expect("symlink");
    let mut config = workspace_write(&link);
    config.writable_roots.push(base.clone());

    let Some(err) = create_error_with(&link, &config) else {
        return;
    };

    assert_eq!(
        err,
        git_refusal::git_dir_under_writable_root(&repo.join(GIT_DIR_NAME), &base)
    );
}

/// `repo()` plus a nested repository at `vendor/dep` that no index lists,
/// with a hooks dir.
#[expect(clippy::expect_used, reason = "test fixture")]
fn nested_repo(root: &Path) {
    let git = root.join("vendor").join("dep").join(GIT_DIR_NAME);
    git_shapes::git_dir_at(&git);
    fs::create_dir_all(git.join("hooks")).expect("hooks dir");
    fs::write(git.join(CONFIG_FILE), b"[core]\n").expect("config");
}

/// A nested repository that is not a submodule: host `git` run inside it
/// would read its `.git`, so a command cannot write a hook or config
/// there, nor move the repository aside.
#[test]
fn exec_cannot_write_the_git_dir_of_a_nested_repository() {
    let _process = lock_process_state();
    let (_dir, root) = repo();
    nested_repo(&root);
    let script = "(: > vendor/dep/.git/hooks/post-checkout) 2>/dev/null && printf 'hook '; \
                  (: > vendor/dep/.git/config) 2>/dev/null && printf 'config '; \
                  mv vendor/dep/.git vendor/dep/g 2>/dev/null && printf 'move '; \
                  mv vendor/dep vendor/d2 2>/dev/null && printf 'rename-parent '; \
                  : > src/ok.rs && printf allowed";
    let Some(stdout) = run(&root, script, &root) else {
        return;
    };
    assert_eq!(stdout, "allowed");
    let git = root.join("vendor/dep").join(GIT_DIR_NAME);
    assert!(!git.join("hooks/post-checkout").exists());
    assert_eq!(
        fs::read(git.join(CONFIG_FILE)).expect("config"),
        b"[core]\n"
    );
}

/// The error a [`DeferredInstance`] returns for an exec in `root`, as the
/// CLI creates it, or `None` when Landlock is not available here.
#[expect(clippy::expect_used, reason = "test helper panics with context")]
fn deferred_exec(root: &Path) -> Option<Result<String, String>> {
    let config = workspace_write(root);
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    runtime.block_on(async {
        let backend = create_backend(&config).await.expect("landlock backend");
        let instance = DeferredInstance::new(backend, config, root.to_path_buf());
        let result = instance
            .exec(
                &["/bin/sh", "-c", "printf ok"],
                Some(root.to_str().expect("utf-8")),
                None,
                Duration::from_secs(10),
            )
            .await
            .map(|result| result.stdout)
            .map_err(|err| err.to_string());
        if result
            .as_ref()
            .is_err_and(|err| err.contains("not available"))
        {
            require_landlock();
            return None;
        }
        Some(result)
    })
}

/// A directory past the scan's depth limit, or one nobody can list,
/// refuses the sandboxed call with the one-line reason. The instance is
/// still made, so the session keeps running.
#[test]
fn a_scan_bound_or_an_unlistable_directory_refuses_sandboxed_calls() {
    use std::os::unix::fs::PermissionsExt;
    let _process = lock_process_state();
    let (_deep_dir, deep) = repo();
    let mut past = deep.clone();
    for _ in 0..65 {
        past.push("d");
    }
    fs::create_dir_all(&past).expect("deep dirs");
    let (_locked_dir, root) = repo();
    let locked = root.join("locked");
    fs::create_dir(&locked).expect("locked");
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).expect("chmod 000");
    fs::read_dir(&locked).expect_err("mode 000 is unlistable (run as a user, not root)");

    let too_deep = deferred_exec(&deep);
    let unlistable = deferred_exec(&root);

    fs::set_permissions(&locked, fs::Permissions::from_mode(0o755)).expect("chmod back");
    let (Some(too_deep), Some(unlistable)) = (too_deep, unlistable) else {
        return;
    };
    assert_eq!(
        too_deep,
        Err(format!(
            "{SANDBOX_UNAVAILABLE}: {}",
            git_refusal::scan_too_deep(&past, 64)
        ))
    );
    assert_eq!(
        unlistable,
        Err(format!(
            "{SANDBOX_UNAVAILABLE}: {}",
            git_refusal::unlistable_dir(&locked)
        ))
    );
}

/// The scan runs at the first sandboxed call, not when the instance is
/// made, and only once: a nested repository added after the instance was
/// made is guarded, and a directory nobody can list, added after the
/// first call, does not refuse a later one.
#[test]
fn the_scan_runs_once_at_the_first_sandboxed_call() {
    use std::os::unix::fs::PermissionsExt;
    let _process = lock_process_state();
    let (_dir, root) = repo();
    let config = workspace_write(&root);
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    let exec = |instance: &DeferredInstance, script: &str| {
        runtime.block_on(instance.exec(
            &["/bin/sh", "-c", script],
            Some(root.to_str().expect("utf-8")),
            None,
            Duration::from_secs(10),
        ))
    };
    let backend = runtime
        .block_on(create_backend(&config))
        .expect("landlock backend");
    let instance = DeferredInstance::new(backend, config, root.clone());
    nested_repo(&root);

    let first = exec(
        &instance,
        "(: > vendor/dep/.git/hooks/post-checkout) 2>/dev/null && printf 'hook '; printf first",
    );
    if first
        .as_ref()
        .is_err_and(|err| err.to_string().contains("not available"))
    {
        require_landlock();
        return;
    }
    let locked = root.join("locked");
    fs::create_dir(&locked).expect("locked");
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).expect("chmod 000");
    let second = exec(&instance, "printf second");
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o755)).expect("chmod back");

    assert_eq!(first.expect("first exec").stdout, "first");
    assert_eq!(second.expect("second exec").stdout, "second");
    assert!(
        !root
            .join("vendor/dep")
            .join(GIT_DIR_NAME)
            .join("hooks/post-checkout")
            .exists()
    );
}

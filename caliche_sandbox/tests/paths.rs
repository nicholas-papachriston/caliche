use std::path::Path;

use caliche_sandbox::{
    NamedPolicy, SandboxBackendType, policy_to_config, resolve_sandbox_path, tmp_writable_roots,
    validate_read_path, validate_write_path,
};
#[cfg(any(unix, windows))]
use caliche_sandbox::{read_validated_file, write_validated_file};
use std::fs;
use tempfile::TempDir;

/// A directory outside every root the policies grant. `/tmp` (and `/private/tmp`
/// on macOS) is writable by design, so a checkout or `$TMPDIR` under it cannot
/// host the fixture: a write there is allowed and the escape tests would fail.
#[expect(clippy::expect_used, reason = "test fixture setup panics with context")]
fn outside_tempdir() -> TempDir {
    let granted = tmp_writable_roots();
    let parent = [
        Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf(),
        std::env::temp_dir(),
    ]
    .into_iter()
    .filter_map(|candidate| candidate.canonicalize().ok())
    .find(|candidate| !granted.iter().any(|root| candidate.starts_with(root)))
    .expect("no fixture directory outside /tmp: run the tests from a checkout outside /tmp");
    tempfile::Builder::new()
        .prefix("caliche-sandbox-outside-")
        .tempdir_in(parent)
        .expect("outside dir")
}

#[test]
fn read_within_workspace_allowed() {
    let dir = TempDir::new().expect("tempdir");
    let workspace = dir.path().canonicalize().expect("canonical workspace");
    let config = policy_to_config(
        NamedPolicy::ReadOnly,
        &workspace,
        None,
        SandboxBackendType::Noop,
    );
    let file = workspace.join("a.txt");
    fs::write(&file, b"ok").expect("write fixture");
    let resolved = validate_read_path(&workspace, &config, "a.txt").expect("read allowed");
    assert_eq!(resolved, file.canonicalize().expect("canonical file"));
}

#[test]
fn write_outside_roots_denied_for_read_only_policy() {
    let dir = TempDir::new().expect("tempdir");
    let workspace = dir.path();
    let config = policy_to_config(
        NamedPolicy::ReadOnly,
        workspace,
        None,
        SandboxBackendType::Noop,
    );
    let err = validate_write_path(workspace, &config, "a.txt").expect_err("write denied");
    assert!(err.to_string().contains("does not allow file writes"));
}

#[test]
fn read_outside_workspace_denied() {
    let dir = TempDir::new().expect("tempdir");
    let workspace = dir.path();
    let outside = TempDir::new().expect("outside tempdir");
    let outside_file = outside.path().join("secret.txt");
    fs::write(&outside_file, b"nope").expect("write outside fixture");

    let config = policy_to_config(
        NamedPolicy::ReadOnly,
        workspace,
        None,
        SandboxBackendType::Noop,
    );
    let err = validate_read_path(
        workspace,
        &config,
        outside_file.to_str().expect("utf8 path"),
    )
    .expect_err("read outside workspace");
    assert!(err.to_string().contains("outside sandbox-allowed"));
}

#[test]
fn write_within_workspace_allowed_for_workspace_write_policy() {
    let dir = TempDir::new().expect("tempdir");
    let workspace = dir.path().canonicalize().expect("canonical workspace");
    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &workspace,
        None,
        SandboxBackendType::Noop,
    );
    let resolved = validate_write_path(&workspace, &config, "out.txt").expect("write allowed");
    assert_eq!(resolved, workspace.join("out.txt"));
}

#[test]
fn resolve_sandbox_path_joins_relative_paths() {
    let workspace = Path::new("/tmp/workspace");
    let resolved = resolve_sandbox_path(workspace, "src/main.rs").expect("resolve relative");
    assert_eq!(resolved, Path::new("/tmp/workspace/src/main.rs"));
}

#[test]
fn resolve_sandbox_path_preserves_absolute_paths() {
    let workspace = Path::new("/tmp/workspace");
    let resolved = resolve_sandbox_path(workspace, "/etc/hosts").expect("resolve absolute");
    assert_eq!(resolved, Path::new("/etc/hosts"));
}

#[test]
fn read_dot_dot_traversal_denied() {
    let dir = TempDir::new().expect("tempdir");
    let workspace = dir.path().canonicalize().expect("canonical workspace");
    let config = policy_to_config(
        NamedPolicy::ReadOnly,
        &workspace,
        None,
        SandboxBackendType::Noop,
    );
    let err = validate_read_path(workspace.as_path(), &config, "../secret.txt")
        .expect_err("dot-dot traversal should be denied");
    assert!(err.to_string().contains("outside sandbox-allowed"));
}

#[test]
fn write_dot_dot_traversal_denied() {
    let dir = TempDir::new().expect("tempdir");
    let workspace = dir.path().canonicalize().expect("canonical workspace");
    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &workspace,
        None,
        SandboxBackendType::Noop,
    );
    let err = validate_write_path(workspace.as_path(), &config, "../../outside.txt")
        .expect_err("dot-dot write traversal should be denied");
    assert!(err.to_string().contains("outside sandbox writable"));
}

#[test]
fn read_nested_dot_dot_traversal_denied() {
    let dir = TempDir::new().expect("tempdir");
    let workspace = dir.path().canonicalize().expect("canonical workspace");
    let nested = workspace.join("nested");
    fs::create_dir_all(&nested).expect("create nested dir");
    let config = policy_to_config(
        NamedPolicy::ReadOnly,
        &workspace,
        None,
        SandboxBackendType::Noop,
    );
    let err = validate_read_path(workspace.as_path(), &config, "nested/../../outside.txt")
        .expect_err("nested dot-dot traversal should be denied");
    assert!(err.to_string().contains("outside sandbox-allowed"));
}

#[test]
fn sibling_prefix_path_is_not_inside_workspace() {
    let dir = TempDir::new().expect("tempdir");
    let parent = dir.path().canonicalize().expect("canonical parent");
    let workspace = parent.join("workspace");
    let sibling = parent.join("workspace-evil");
    fs::create_dir_all(&workspace).expect("workspace");
    fs::create_dir_all(&sibling).expect("sibling");
    fs::write(sibling.join("secret.txt"), b"nope").expect("sibling secret");
    let config = policy_to_config(
        NamedPolicy::ReadOnly,
        &workspace,
        None,
        SandboxBackendType::Noop,
    );
    let err = validate_read_path(
        &workspace,
        &config,
        sibling.join("secret.txt").to_str().expect("utf8"),
    )
    .expect_err("prefix sibling must not count as inside workspace");
    assert!(err.to_string().contains("outside sandbox-allowed"));
}

#[test]
fn nul_path_is_rejected() {
    let dir = TempDir::new().expect("tempdir");
    let workspace = dir.path().canonicalize().expect("canonical workspace");
    let config = policy_to_config(
        NamedPolicy::ReadOnly,
        &workspace,
        None,
        SandboxBackendType::Noop,
    );
    let err = validate_read_path(&workspace, &config, "a\0.txt").expect_err("NUL");
    assert!(err.to_string().contains("NUL"));
}

#[test]
fn dotted_filename_is_not_traversal() {
    let dir = TempDir::new().expect("tempdir");
    let workspace = dir.path().canonicalize().expect("canonical workspace");
    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &workspace,
        None,
        SandboxBackendType::Noop,
    );
    let resolved =
        validate_write_path(&workspace, &config, "foo..bar.txt").expect("dots in name are fine");
    assert_eq!(resolved, workspace.join("foo..bar.txt"));
}

#[cfg(unix)]
#[test]
fn write_through_directory_symlink_outside_denied() {
    use std::os::unix::fs::symlink;

    let dir = TempDir::new().expect("tempdir");
    let workspace = dir.path().canonicalize().expect("canonical workspace");
    let outside_dir = outside_tempdir();
    let outside = outside_dir
        .path()
        .canonicalize()
        .expect("canonical outside");
    symlink(&outside, workspace.join("escape")).expect("symlink");

    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &workspace,
        None,
        SandboxBackendType::Noop,
    );
    let err = validate_write_path(&workspace, &config, "escape/pwned.txt")
        .expect_err("symlink write escape must be denied");
    assert!(err.to_string().contains("outside sandbox writable"));
    assert!(
        !outside.join("pwned.txt").exists(),
        "validator must not create the escaped file"
    );
}

#[cfg(unix)]
#[test]
fn write_through_nested_missing_dirs_on_symlink_denied() {
    use std::os::unix::fs::symlink;

    let dir = TempDir::new().expect("tempdir");
    let workspace = dir.path().canonicalize().expect("canonical workspace");
    let outside_dir = outside_tempdir();
    let outside = outside_dir
        .path()
        .canonicalize()
        .expect("canonical outside");
    symlink(&outside, workspace.join("escape")).expect("symlink");

    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &workspace,
        None,
        SandboxBackendType::Noop,
    );
    let err = validate_write_path(&workspace, &config, "escape/a/b/c.txt")
        .expect_err("nested symlink write escape must be denied");
    assert!(err.to_string().contains("outside sandbox writable"));
}

#[cfg(unix)]
#[test]
fn read_symlink_to_outside_file_denied() {
    use std::os::unix::fs::symlink;

    let dir = TempDir::new().expect("tempdir");
    let workspace = dir.path().canonicalize().expect("canonical workspace");
    let outside_dir = outside_tempdir();
    let outside = outside_dir
        .path()
        .canonicalize()
        .expect("canonical outside");
    let secret = outside.join("secret.txt");
    fs::write(&secret, b"classified").expect("outside secret");
    symlink(&secret, workspace.join("link.txt")).expect("file symlink");

    let config = policy_to_config(
        NamedPolicy::ReadOnly,
        &workspace,
        None,
        SandboxBackendType::Noop,
    );
    let err = validate_read_path(&workspace, &config, "link.txt")
        .expect_err("symlink to outside file must be denied");
    assert!(err.to_string().contains("outside sandbox-allowed"));
}

#[cfg(unix)]
#[test]
fn write_through_symlink_into_workspace_allowed() {
    use std::os::unix::fs::symlink;

    let dir = TempDir::new().expect("tempdir");
    let workspace = dir.path().canonicalize().expect("canonical workspace");
    let nested = workspace.join("nested");
    fs::create_dir_all(&nested).expect("nested");
    symlink(&nested, workspace.join("alias")).expect("in-workspace symlink");

    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &workspace,
        None,
        SandboxBackendType::Noop,
    );
    let resolved =
        validate_write_path(&workspace, &config, "alias/out.txt").expect("in-workspace symlink");
    assert_eq!(resolved, nested.join("out.txt"));
}

#[test]
fn workspace_write_does_not_allow_host_tmpdir_siblings() {
    let dir = TempDir::new().expect("tempdir");
    let workspace = dir.path().canonicalize().expect("canonical workspace");
    let sibling_dir = outside_tempdir();
    let sibling = sibling_dir
        .path()
        .canonicalize()
        .expect("canonical sibling");
    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &workspace,
        None,
        SandboxBackendType::Noop,
    );
    let err = validate_write_path(
        &workspace,
        &config,
        sibling.join("other.txt").to_str().expect("utf8"),
    )
    .expect_err("host TMPDIR siblings must not be writable");
    assert!(err.to_string().contains("outside sandbox writable"));
}

#[cfg(unix)]
#[test]
fn write_validated_file_refuses_last_component_symlink() {
    use std::os::unix::fs::symlink;

    let dir = TempDir::new().expect("tempdir");
    let workspace = dir.path().canonicalize().expect("canonical workspace");
    let outside = outside_tempdir();
    let dangling = outside.path().join("missing-target");
    symlink(&dangling, workspace.join("link.txt")).expect("dangling symlink");

    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &workspace,
        None,
        SandboxBackendType::Noop,
    );
    let err =
        write_validated_file(&workspace, &config, "link.txt", b"pwned").expect_err("symlink write");
    assert!(
        err.to_string().contains("symlink"),
        "unexpected error: {err}"
    );
    assert!(!dangling.exists());
}

#[cfg(unix)]
#[test]
fn read_validated_file_refuses_last_component_symlink_to_outside() {
    use std::os::unix::fs::symlink;

    let dir = TempDir::new().expect("tempdir");
    let workspace = dir.path().canonicalize().expect("canonical workspace");
    let outside = outside_tempdir();
    let target = outside.path().join("secret.txt");
    fs::write(&target, b"secret").expect("outside file");
    symlink(&target, workspace.join("link.txt")).expect("symlink");

    let config = policy_to_config(
        NamedPolicy::ReadOnly,
        &workspace,
        None,
        SandboxBackendType::Noop,
    );
    let err = read_validated_file(&workspace, &config, "link.txt").expect_err("symlink read");
    assert!(
        err.to_string().contains("symlink") || err.to_string().contains("outside"),
        "unexpected error: {err}"
    );
}

#[test]
fn writes_into_the_workspace_git_dir_are_refused() {
    let dir = TempDir::new().expect("tempdir");
    let workspace = dir.path().canonicalize().expect("canonical workspace");
    std::fs::create_dir_all(workspace.join(".git/hooks")).expect("git dir");
    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &workspace,
        None,
        SandboxBackendType::Noop,
    );
    for path in [
        ".git/config",
        ".git/hooks/pre-commit",
        ".git",
        ".GIT/config",
        "vendor/dep/.git/config",
        "src/../.git/info/attributes",
    ] {
        let err = validate_write_path(&workspace, &config, path).expect_err(path);
        assert_eq!(
            err.to_string(),
            format!("write path '{path}' is in the repository's .git directory")
        );
    }
    for path in [".gitignore", ".github/ci.yml", "src/git/mod.rs"] {
        validate_write_path(&workspace, &config, path).expect(path);
    }
}

#[cfg(windows)]
#[test]
fn write_refuses_a_hard_link_and_leaves_the_linked_file() {
    let dir = TempDir::new().expect("tempdir");
    let workspace = dir.path().canonicalize().expect("canonical workspace");
    let outside = outside_tempdir();
    let outside_file = outside.path().join("file.txt");
    fs::write(&outside_file, b"orig").expect("outside file");
    fs::hard_link(&outside_file, workspace.join("linked.txt")).expect("hard link");
    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &workspace,
        None,
        SandboxBackendType::Noop,
    );

    let err = write_validated_file(&workspace, &config, "linked.txt", b"evil")
        .expect_err("a hard-linked file must not be written");
    let message = format!("{err:#}");
    assert!(
        message.ends_with(caliche_sandbox::HARD_LINK_REFUSAL),
        "unexpected error: {message}"
    );
    assert_eq!(fs::read(&outside_file).expect("read"), b"orig");
    assert_eq!(
        read_validated_file(&workspace, &config, "linked.txt").expect("reads are allowed"),
        b"orig"
    );
}

/// A canonical Windows path is verbatim; the form for external tools is
/// plain and names the same directory.
#[cfg(windows)]
#[test]
fn a_canonical_windows_path_is_plain_for_external_tools() {
    let dir = tempfile::tempdir().expect("tempdir");
    let canonical = dir.path().canonicalize().expect("canonical");
    assert!(
        canonical.to_string_lossy().starts_with(r"\\?\"),
        "{canonical:?}"
    );

    let plain = caliche_sandbox::external_tool_path(&canonical);

    assert!(!plain.to_string_lossy().starts_with(r"\\?\"), "{plain:?}");
    assert_eq!(
        plain.to_string_lossy(),
        canonical.to_string_lossy().trim_start_matches(r"\\?\")
    );
    assert_eq!(plain.canonicalize().expect("plain resolves"), canonical);
}

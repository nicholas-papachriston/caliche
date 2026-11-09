/// The git dir host git discovers from `dir` with bare repositories
/// allowed, as a user's own git would, or `None`.
#[expect(clippy::expect_used, reason = "test helper")]
fn discovered_git_dir(dir: &std::path::Path) -> Option<String> {
    let mut command = std::process::Command::new("git");
    let output = clear_git_config(&mut command)
        .args(["-c", &format!("{SAFE_BARE_REPOSITORY_KEY}=all")])
        .args(["rev-parse", "--absolute-git-dir"])
        .env("GIT_CEILING_DIRECTORIES", "")
        .current_dir(dir)
        .output()
        .expect("host git must be installed to run these tests");
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Each attempt to plant a bare repository (`HEAD`, `objects`, `refs`)
/// where host git discovery would find it, in a workspace without `.git`
/// or a parent of it, with a writable root around it, runs in its own
/// sandbox. Each prints its name only if it succeeds; none may, and host
/// git finds no repository afterwards.
#[tokio::test]
async fn every_attempt_to_plant_a_bare_repository_is_denied() {
    let head = "printf 'ref: refs/heads/main\\n'";
    let parts = |dir: &str| format!("mkdir -p {dir}/{OBJECTS_DIR} {dir}/{REFS_DIR}");
    let attacks: Vec<(&str, String)> = vec![
        (
            "plant it in the workspace",
            format!("{} && {head} > parent/ws/{HEAD_FILE}", parts("parent/ws")),
        ),
        (
            "plant it in the parent",
            format!("{} && {head} > parent/{HEAD_FILE}", parts("parent")),
        ),
        (
            "plant it in a higher parent",
            format!("{} && {head} > {HEAD_FILE}", parts(".")),
        ),
        (
            "plant it in another case",
            format!("{} && {head} > parent/head", parts("parent")),
        ),
        (
            "rename a file onto it",
            format!(
                "{} && {head} > h && mv h parent/{HEAD_FILE}",
                parts("parent")
            ),
        ),
        (
            "hard-link a file to it",
            format!(
                "{} && {head} > h && ln h parent/{HEAD_FILE}",
                parts("parent")
            ),
        ),
        (
            "put a symlink at it",
            format!(
                "{} && ln -s refs/heads/main parent/{HEAD_FILE}",
                parts("parent")
            ),
        ),
        (
            "plant it through an alias",
            format!(
                "{} && {head} > \"$ALIAS/parent/{HEAD_FILE}\"",
                parts("parent")
            ),
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

        if stdout == "ok" || discovered_git_dir(&root).is_some() {
            succeeded.push(*name);
        }
    }
    assert_eq!(succeeded, Vec::<&str>::new());
}

/// A workspace inside a linked worktree, whose main repository a writable
/// root holds, is refused at create: the sandbox could write the hooks
/// host git runs for that worktree.
#[tokio::test]
async fn create_refuses_a_workspace_in_a_worktree_of_a_writable_repository() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let base = dir.path().canonicalize().expect("canonical");
    let main = base.join("m").join("r");
    std::fs::create_dir_all(&main).expect("main");
    git(&main, &["init", "-q"]);
    git(&main, &["commit", "-q", "--allow-empty", "-m", "init"]);
    let wt = base.join("wt");
    git(
        &main,
        &["worktree", "add", "-q", wt.to_str().expect("utf-8")],
    );
    let root = wt.join("sub");
    std::fs::create_dir(&root).expect("sub");
    let git_dir = main.join(GIT_DIR_NAME).join("worktrees").join("wt");

    let err = MacOSSeatbeltBackend
        .create(&writable_base_config(&base), &root)
        .await
        .err()
        .expect("refused");

    assert_eq!(
        err.to_string(),
        git_refusal::git_dir_under_writable_root(&git_dir, &base)
    );
}

/// Every layout where host git would find a repository the sandbox could
/// write, or could finish planting one, is refused at create.
#[tokio::test]
async fn create_refuses_every_planted_repository_shape() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let base = dir.path().canonicalize().expect("canonical");
    let mut wrong = Vec::new();

    for shape in git_shapes::planted_repository_shapes(&base) {
        let refused = MacOSSeatbeltBackend
            .create(&writable_base_config(&shape.writable), &shape.workspace)
            .await
            .err()
            .map(|err| err.to_string());
        if refused.as_deref() != Some(shape.refusal.as_str()) {
            wrong.push((shape.name, refused));
        }
    }

    assert_eq!(wrong, Vec::<(&str, Option<String>)>::new());
}

/// A `.git` git would skip is refused even when nothing is writable: the
/// answer about which repository host git uses must not rest on it.
#[tokio::test]
async fn create_refuses_an_invalid_git_entry_with_nothing_writable() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let root = dir.path().canonicalize().expect("canonical");
    std::fs::create_dir(root.join(GIT_DIR_NAME)).expect(".git");
    let config = policy_to_config(
        NamedPolicy::ReadOnly,
        &root,
        None,
        SandboxBackendType::MacOSSeatbelt {
            policy: NamedPolicy::ReadOnly,
        },
    );

    let err = MacOSSeatbeltBackend
        .create(&config, &root)
        .await
        .err()
        .expect("refused");

    assert_eq!(
        err.to_string(),
        git_refusal::invalid_git_entry(&root.join(GIT_DIR_NAME))
    );
}

/// Create resolves the workspace first, so one opened through a symlink
/// is checked against the repository git finds from where it really is.
#[tokio::test]
async fn create_checks_a_workspace_opened_through_a_symlink_where_it_resolves() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let base = dir.path().canonicalize().expect("canonical");
    let repo = base.join("repo");
    git_shapes::git_dir_at(&repo.join(GIT_DIR_NAME));
    let sub = repo.join("sub");
    std::fs::create_dir(&sub).expect("sub");
    let link = base.join("link");
    std::os::unix::fs::symlink(&sub, &link).expect("symlink");

    let err = MacOSSeatbeltBackend
        .create(&writable_base_config(&base), &link)
        .await
        .err()
        .expect("refused");

    assert_eq!(
        err.to_string(),
        git_refusal::git_dir_under_writable_root(&repo.join(GIT_DIR_NAME), &base)
    );
}

/// A workspace with a valid `.git` still cannot gain a root `HEAD`. That
/// file would be a bare repository if git later rejected the `.git`.
#[tokio::test]
async fn exec_cannot_write_head_beside_a_valid_git_dir() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let base = dir.path().canonicalize().expect("canonical");
    let root = base.join("ws");
    std::fs::create_dir(&root).expect("ws");
    git_shapes::git_dir_at(&root.join(GIT_DIR_NAME));
    let script = format!(
        "printf x > ws/note && printf '{LOOSE}' > ws/{HEAD_FILE} && printf wrote",
        LOOSE = "ref: refs/heads/main\\n"
    );

    let stdout = run_attack(&base, &root, &script).await;

    assert_ne!(stdout, "wrote");
    assert!(
        root.join("note").is_file(),
        "an ordinary file stays writable"
    );
    assert!(!root.join(HEAD_FILE).exists(), "HEAD was created");
}

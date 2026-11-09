use super::*;
use caliche_policy::{
    COMMONDIR_FILE, Discovery, GIT_DIR_LINKS, GITFILE_PREFIX, HEAD_FILE, LOOSE_HEAD_CONTENTS,
    MODULES_DIR, OBJECTS_DIR, REFS_DIR, clear_git_config, discover, plant_linked_worktree,
    write_loose_git_dir,
};
use std::process::Command;

/// Runs host `git`. A test host without it fails these tests rather
/// than skipping them, so they always run.
fn git(dir: &Path, args: &[&str]) {
    let mut command = Command::new("git");
    let status = clear_git_config(&mut command)
        .args([
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "init.defaultBranch=main",
        ])
        .args(["-c", "protocol.file.allow=always"])
        .args(args)
        .current_dir(dir)
        .status()
        .expect("host git must be installed to run these tests");
    assert!(status.success(), "git {args:?}");
}

/// A superproject with a checked-out submodule at `libs/a`, an
/// uninitialized one at `libs/b`, and a nested repo at `tmp` that is not
/// a submodule.
/// `object_format` is `sha1` or `sha256`; `intent_to_add` stages
/// `a.txt` with `git add -N`, which sets the extended flag on an
/// entry before the submodules.
fn superproject(
    index_version: &str,
    object_format: &str,
    intent_to_add: bool,
) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path();
    let init = format!("--object-format={object_format}");
    std::fs::create_dir(root.join("lib")).expect("lib");
    git(&root.join("lib"), &["init", "-q", &init]);
    git(
        &root.join("lib"),
        &["commit", "-q", "--allow-empty", "-m", "init"],
    );
    std::fs::create_dir(root.join("top")).expect("top");
    let top = root.join("top");
    git(&top, &["init", "-q", &init]);
    git(&top, &["submodule", "add", "-q", "../lib", "libs/a"]);
    git(&top, &["submodule", "add", "-q", "../lib", "libs/b"]);
    git(&top, &["commit", "-q", "-m", "add"]);
    git(&top, &["submodule", "deinit", "-q", "-f", "libs/b"]);
    if intent_to_add {
        std::fs::write(top.join("a.txt"), "n").expect("a.txt");
        git(&top, &["add", "-N", "a.txt"]);
    }
    git(&top, &["update-index", "--index-version", index_version]);
    std::fs::create_dir(top.join("tmp")).expect("tmp");
    git(&top.join("tmp"), &["init", "-q"]);
    dir
}

/// The root `.git` and every submodule's, checked out or not. The nested
/// repo at `tmp` is not in the index, so it is not a target.
fn expected() -> Vec<PathBuf> {
    ["", "libs/a", "libs/b"]
        .iter()
        .map(|dir| Path::new(dir).join(GIT_DIR_NAME))
        .collect()
}

#[test]
fn every_submodule_path_in_the_index_is_a_target() {
    for version in ["2", "3", "4"] {
        let dir = superproject(version, "sha1", false);
        assert_eq!(
            index_targets(&dir.path().join("top")).expect("targets"),
            expected(),
            "index v{version}"
        );
    }
}

#[test]
fn a_sha256_index_lists_its_submodules() {
    let dir = superproject("2", "sha256", false);
    assert_eq!(
        index_targets(&dir.path().join("top")).expect("targets"),
        expected()
    );
}

#[test]
fn an_intent_to_add_entry_does_not_hide_a_submodule() {
    for version in ["3", "4"] {
        let dir = superproject(version, "sha1", true);
        assert_eq!(
            index_targets(&dir.path().join("top")).expect("targets"),
            expected(),
            "index v{version}"
        );
    }
}

#[test]
fn a_split_index_is_refused() {
    let dir = superproject("2", "sha1", false);
    let top = dir.path().join("top");
    git(&top, &["update-index", "--split-index"]);
    let err = index_targets(&top).expect_err("split index");
    assert_eq!(err.root_cause().to_string(), git_refusal::SPLIT_INDEX);
}

/// `top` holds a checked-out submodule `sub` whose own index records
/// `sub/inner`, which is not checked out.
pub(crate) fn nested_submodules() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path();
    for name in ["leaf", "mid", "top"] {
        std::fs::create_dir(root.join(name)).expect("repo dir");
        git(&root.join(name), &["init", "-q"]);
    }
    git(
        &root.join("leaf"),
        &["commit", "-q", "--allow-empty", "-m", "leaf"],
    );
    git(
        &root.join("mid"),
        &["submodule", "add", "-q", "../leaf", "inner"],
    );
    git(&root.join("mid"), &["commit", "-q", "-m", "mid"]);
    let top = root.join("top");
    git(&top, &["submodule", "add", "-q", "../mid", "sub"]);
    git(&top, &["commit", "-q", "-m", "top"]);
    dir
}

/// The checked-out submodule's index records `sub/inner`, which is not
/// checked out; it is a target too.
#[test]
fn a_submodule_with_its_own_submodules_adds_their_targets() {
    let dir = nested_submodules();

    let targets = index_targets(&dir.path().join("top")).expect("targets");

    assert_eq!(
        targets,
        vec![
            PathBuf::from(GIT_DIR_NAME),
            Path::new("sub").join(GIT_DIR_NAME),
            Path::new("sub/inner").join(GIT_DIR_NAME),
        ]
    );
}

#[test]
fn a_workspace_without_git_still_guards_the_root_git() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir(dir.path().join("src")).expect("src");
    assert_eq!(
        index_targets(dir.path()).expect("targets"),
        vec![PathBuf::from(GIT_DIR_NAME)]
    );
    assert!(only_root(&[PathBuf::from(GIT_DIR_NAME)]));
    assert!(!only_root(&expected()));
}

#[test]
fn a_damaged_index_fails_closed() {
    let dir = superproject("2", "sha1", false);
    let index = dir.path().join("top").join(GIT_DIR_NAME).join(INDEX_FILE);
    let data = std::fs::read(&index).expect("index");
    std::fs::write(&index, &data[..40]).expect("truncate");
    let err = index_targets(&dir.path().join("top")).expect_err("damaged index");
    assert!(format!("{err:#}").contains("index ends early"), "{err:#}");
}

/// One directory per shape git's loose git dir test decides on.
#[cfg(unix)]
fn git_dir_cases(base: &Path) -> Vec<(&'static str, PathBuf)> {
    use std::os::unix::fs::PermissionsExt;
    let case = |name: &'static str, head: Option<&str>, parts: bool| {
        let dir = base.join(name);
        std::fs::create_dir(&dir).expect("case");
        if parts {
            for part in [OBJECTS_DIR, REFS_DIR] {
                std::fs::create_dir(dir.join(part)).expect("part");
            }
        }
        if let Some(head) = head {
            std::fs::write(dir.join(HEAD_FILE), head).expect("HEAD");
        }
        (name, dir)
    };
    let mut cases = vec![
        case("ref-head", Some(LOOSE_HEAD_CONTENTS), true),
        case("ref-head-no-space", Some("ref:refs/heads/main\n"), true),
        case(
            "oid-head",
            Some("0123456789abcdef0123456789abcdef01234567\n"),
            true,
        ),
        case(
            "short-oid-head",
            Some("0123456789abcdef0123456789abcdef0123456\n"),
            true,
        ),
        case("ref-outside-refs", Some("ref: heads/main\n"), true),
        case("no-objects-or-refs", Some(LOOSE_HEAD_CONTENTS), false),
    ];
    let (name, dir) = case("exec-file-parts", Some(LOOSE_HEAD_CONTENTS), false);
    for part in [OBJECTS_DIR, REFS_DIR] {
        std::fs::write(dir.join(part), "").expect("part file");
        std::fs::set_permissions(dir.join(part), std::fs::Permissions::from_mode(0o755))
            .expect("chmod");
    }
    cases.push((name, dir));
    let (name, dir) = case("plain-file-parts", Some(LOOSE_HEAD_CONTENTS), false);
    for part in [OBJECTS_DIR, REFS_DIR] {
        std::fs::write(dir.join(part), "").expect("part file");
    }
    cases.push((name, dir));
    for (name, target) in [
        ("dangling-head-link", "refs/heads/main"),
        ("head-link-outside-refs", "elsewhere"),
    ] {
        let (name, dir) = case(name, None, true);
        std::os::unix::fs::symlink(target, dir.join(HEAD_FILE)).expect("HEAD link");
        cases.push((name, dir));
    }
    let (name, dir) = case("commondir", Some(LOOSE_HEAD_CONTENTS), false);
    for part in [OBJECTS_DIR, REFS_DIR] {
        std::fs::create_dir_all(dir.join("common").join(part)).expect("common part");
    }
    std::fs::write(dir.join(COMMONDIR_FILE), "common\n").expect("commondir");
    cases.push((name, dir));
    let (name, dir) = case("empty-dot-git", None, false);
    std::fs::create_dir(dir.join(GIT_DIR_NAME)).expect(".git");
    cases.push((name, dir));
    let (name, dir) = case("invalid-gitfile", None, false);
    std::fs::write(dir.join(GIT_DIR_NAME), "gitdir: nowhere\n").expect("gitfile");
    cases.push((name, dir));
    let (name, dir) = case("gitfile-to-a-git-dir", None, false);
    std::fs::write(
        dir.join(GIT_DIR_NAME),
        format!("{GITFILE_PREFIX} ../ref-head\n"),
    )
    .expect("gitfile");
    cases.push((name, dir));
    let (name, dir) = case("symlinked-dot-git", None, false);
    std::os::unix::fs::symlink("../ref-head", dir.join(GIT_DIR_NAME)).expect(".git link");
    cases.push((name, dir));
    // A symlink whose text is not `refs/…`, even when the file it
    // names holds a valid symref. Git does not read through it.
    let (name, dir) = case("head-link-to-ref-text", None, true);
    std::fs::write(dir.join("x"), LOOSE_HEAD_CONTENTS).expect("target");
    std::os::unix::fs::symlink("x", dir.join(HEAD_FILE)).expect("HEAD link");
    cases.push((name, dir));
    cases.push(case(
        "ref-vertical-tab",
        Some("ref:\u{000B}refs/heads/main\n"),
        true,
    ));
    let (name, dir) = case(
        "objects-not-owner-searchable",
        Some(LOOSE_HEAD_CONTENTS),
        true,
    );
    for part in [OBJECTS_DIR, REFS_DIR] {
        std::fs::set_permissions(dir.join(part), std::fs::Permissions::from_mode(0o010))
            .expect("chmod");
    }
    cases.push((name, dir));
    cases
}

/// The git dir `git rev-parse` finds from `dir`, not looking above
/// `ceiling`, with no user or system config, or `None`.
#[cfg(unix)]
fn git_finds(dir: &Path, ceiling: &Path) -> Option<PathBuf> {
    let mut command = Command::new("git");
    let output = clear_git_config(&mut command)
        .args(["rev-parse", "--absolute-git-dir"])
        .env("GIT_CEILING_DIRECTORIES", ceiling)
        .current_dir(dir)
        .output()
        .expect("run git");
    output.status.success().then(|| {
        PathBuf::from(String::from_utf8_lossy(&output.stdout).trim_end())
            .canonicalize()
            .expect("canonical git dir")
    })
}

/// Discovery takes a directory as a git dir exactly when git does, for
/// every `HEAD`, `objects`/`refs`, `commondir` and `.git` shape, except
/// a symlinked `.git`: git follows it, discovery refuses it.
#[cfg(unix)]
#[test]
fn discovery_agrees_with_git_on_every_git_dir_shape() {
    let (_dir, base, _ws) = scratch();
    let cases = git_dir_cases(&base);
    let ours = |dir: &Path| match discover(dir) {
        Discovery::Repository { git_dir, .. } => {
            Some(git_dir.canonicalize().expect("canonical git dir"))
        }
        _ => None,
    };

    let disagree: Vec<(&str, Option<PathBuf>, Option<PathBuf>)> = cases
        .iter()
        .map(|(name, dir)| (*name, ours(dir), git_finds(dir, &base)))
        .filter(|(_, ours, git)| ours != git)
        .collect();
    let found: Vec<&str> = cases
        .iter()
        .filter(|(_, dir)| ours(dir).is_some())
        .map(|(name, _)| *name)
        .collect();

    assert_eq!(
        disagree,
        [(
            "symlinked-dot-git",
            None,
            Some(base.join("ref-head").canonicalize().expect("canonical"))
        )]
    );
    assert_eq!(
        found,
        [
            "ref-head",
            "ref-head-no-space",
            "oid-head",
            "exec-file-parts",
            "dangling-head-link",
            "commondir",
            "gitfile-to-a-git-dir"
        ]
    );
}

/// What discovery reports for a `.git` git would not use and for a
/// `HEAD` that does not make its directory a git dir.
#[cfg(unix)]
#[test]
fn discovery_reports_an_invalid_git_entry_and_a_stray_head() {
    let (_dir, base, _ws) = scratch();
    let cases = git_dir_cases(&base);
    let at = |name: &str| {
        let dir = &cases
            .iter()
            .find(|(case, _)| *case == name)
            .expect("case")
            .1;
        discover(dir)
    };

    assert_eq!(
        at("empty-dot-git"),
        Discovery::InvalidGitEntry(base.join("empty-dot-git").join(GIT_DIR_NAME))
    );
    assert_eq!(
        at("invalid-gitfile"),
        Discovery::InvalidGitEntry(base.join("invalid-gitfile").join(GIT_DIR_NAME))
    );
    assert_eq!(
        at("symlinked-dot-git"),
        Discovery::InvalidGitEntry(base.join("symlinked-dot-git").join(GIT_DIR_NAME))
    );
    assert_eq!(
        at("short-oid-head"),
        Discovery::StrayHead(base.join("short-oid-head").join(HEAD_FILE))
    );
    assert_eq!(
        at("head-link-outside-refs"),
        Discovery::StrayHead(base.join("head-link-outside-refs").join(HEAD_FILE))
    );
    assert_eq!(
        at("head-link-to-ref-text"),
        Discovery::StrayHead(base.join("head-link-to-ref-text").join(HEAD_FILE))
    );
    assert_eq!(
        at("ref-vertical-tab"),
        Discovery::StrayHead(base.join("ref-vertical-tab").join(HEAD_FILE))
    );
    assert_eq!(
        at("objects-not-owner-searchable"),
        Discovery::StrayHead(base.join("objects-not-owner-searchable").join(HEAD_FILE))
    );
}

/// Git uses the `.git`. A `HEAD` beside it would be a bare repository
/// if that `.git` were later rejected, so discovery reports it.
#[cfg(unix)]
#[test]
fn a_head_beside_an_accepted_git_dir_is_stray() {
    let (_dir, base, ws) = scratch();
    real_git_dir(&ws.join(GIT_DIR_NAME));
    std::fs::write(ws.join(HEAD_FILE), LOOSE_HEAD_CONTENTS).expect("HEAD");

    assert_eq!(discover(&ws), Discovery::StrayHead(ws.join(HEAD_FILE)));
    assert_eq!(
        git_finds(&ws, &base).expect("git uses .git"),
        ws.join(GIT_DIR_NAME).canonicalize().expect("canonical")
    );
}

/// A git dir git accepts: `HEAD`, `objects/` and `refs/`.
pub(crate) fn real_git_dir(dir: &Path) {
    write_loose_git_dir(dir).expect("git dir");
}

/// A temp dir holding an empty workspace `ws`, both canonical.
fn scratch() -> (tempfile::TempDir, PathBuf, PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let base = dir.path().canonicalize().expect("canonical");
    let ws = base.join("ws");
    std::fs::create_dir(&ws).expect("ws");
    (dir, base, ws)
}

fn gitfile(at: &Path, git_dir: &Path) {
    std::fs::create_dir_all(at).expect("gitfile parent");
    std::fs::write(
        at.join(GIT_DIR_NAME),
        format!("{GITFILE_PREFIX} {}\n", git_dir.display()),
    )
    .expect("gitfile");
}

fn check(ws: &Path, writable: &[PathBuf]) -> Result<()> {
    let targets = index_targets(ws).expect("targets");
    refuse_exposed_git_dirs(ws, &targets, writable)?;
    Ok(())
}

/// `git init --separate-git-dir ws/meta ws`.
#[test]
fn a_git_dir_inside_the_workspace_outside_git_is_refused() {
    let (_dir, _base, ws) = scratch();
    real_git_dir(&ws.join("meta"));
    gitfile(&ws, Path::new("meta"));

    let err = check(&ws, std::slice::from_ref(&ws)).expect_err("refused");

    assert_eq!(
        err.to_string(),
        git_refusal::git_dir_in_workspace(&ws.join("meta"))
    );
}

/// A linked worktree whose main repository is under a writable root.
#[test]
fn a_worktree_git_dir_under_a_writable_root_is_refused() {
    let (_dir, base, ws) = scratch();
    let common = base.join("main").join(GIT_DIR_NAME);
    let git_dir = plant_linked_worktree(&common, "ws").expect("linked worktree");
    gitfile(&ws, &git_dir);
    let main = base.join("main");

    let refused = check(&ws, &[ws.clone(), main.clone()]).expect_err("refused");
    let allowed = check(&ws, std::slice::from_ref(&ws));

    assert_eq!(
        refused.to_string(),
        git_refusal::git_dir_under_writable_root(&git_dir, &main)
    );
    assert!(allowed.is_ok(), "{allowed:?}");
}

/// A workspace without a `.git` inside a linked worktree: host git
/// finds the worktree's git dir above the workspace, so a writable
/// root holding the main repository is refused.
#[test]
fn an_enclosing_worktree_git_dir_under_a_writable_root_is_refused() {
    let (_dir, base, _ws) = scratch();
    let common = base.join("main").join(GIT_DIR_NAME);
    let git_dir = plant_linked_worktree(&common, "wt").expect("linked worktree");
    let ws = base.join("wt").join("sub");
    std::fs::create_dir_all(&ws).expect("ws");
    gitfile(&base.join("wt"), &git_dir);
    let main = base.join("main");

    let refused = check(&ws, &[ws.clone(), main.clone()]).expect_err("refused");
    let allowed = check(&ws, std::slice::from_ref(&ws));

    assert_eq!(
        refused.to_string(),
        git_refusal::git_dir_under_writable_root(&git_dir, &main)
    );
    assert!(allowed.is_ok(), "{allowed:?}");
}

/// A bare repository above the workspace is what host git finds, so a
/// writable root holding it is refused, and so is a workspace that is
/// a bare repository itself.
#[test]
fn an_enclosing_bare_repository_is_checked() {
    let (_dir, base, ws) = scratch();
    for dir in [OBJECTS_DIR, REFS_DIR] {
        std::fs::create_dir(base.join(dir)).expect("git dir part");
        std::fs::create_dir(ws.join(dir)).expect("git dir part");
    }
    std::fs::write(base.join(HEAD_FILE), LOOSE_HEAD_CONTENTS).expect("HEAD");

    let parent = check(&ws, &[ws.clone(), base.clone()]).expect_err("parent refused");
    std::fs::write(ws.join(HEAD_FILE), LOOSE_HEAD_CONTENTS).expect("HEAD");
    let itself = check(&ws, std::slice::from_ref(&ws)).expect_err("workspace refused");

    assert_eq!(
        parent.to_string(),
        git_refusal::git_dir_under_writable_root(&base, &base)
    );
    assert_eq!(itself.to_string(), git_refusal::git_dir_in_workspace(&ws));
}

/// A parent that is a git dir through a `commondir`, with no `objects`
/// or `refs` of its own, is found too: it holds the workspace, so a
/// writable workspace lies inside a git dir.
#[test]
fn an_enclosing_git_dir_with_a_commondir_is_found() {
    let (_dir, base, ws) = scratch();
    let common = base.join("common");
    for dir in [OBJECTS_DIR, REFS_DIR] {
        std::fs::create_dir_all(common.join(dir)).expect("common part");
    }
    std::fs::write(base.join(HEAD_FILE), LOOSE_HEAD_CONTENTS).expect("HEAD");
    std::fs::write(base.join(COMMONDIR_FILE), "common\n").expect("commondir");

    let err = check(&ws, std::slice::from_ref(&ws)).expect_err("refused");

    assert_eq!(err.to_string(), git_refusal::overlapping_root(&ws, &base));
}

/// A workspace with its own `.git` is not checked against a repository
/// above it: host git finds the workspace's first.
#[test]
fn a_repository_above_a_workspace_with_its_own_git_is_not_checked() {
    let (_dir, base, ws) = scratch();
    let common = base.join("main").join(GIT_DIR_NAME);
    std::fs::create_dir_all(&common).expect("common");
    real_git_dir(&common);
    gitfile(&base, &common);
    real_git_dir(&ws.join(GIT_DIR_NAME));

    let allowed = check(&ws, &[ws.clone(), base.join("main")]);

    assert!(allowed.is_ok(), "{allowed:?}");
}

/// The git dir is safe but the common dir it names is writable.
#[test]
fn a_common_dir_under_a_writable_root_is_refused() {
    let (_dir, base, ws) = scratch();
    let git_dir = base.join("private").join("ws");
    let common = base.join("shared").join(GIT_DIR_NAME);
    real_git_dir(&git_dir);
    real_git_dir(&common);
    std::fs::write(git_dir.join(COMMONDIR_FILE), "../../shared/.git\n").expect("commondir");
    gitfile(&ws, &git_dir);
    let shared = base.join("shared");

    let err = check(&ws, &[ws.clone(), shared.clone()]).expect_err("refused");

    assert_eq!(
        err.to_string(),
        git_refusal::git_dir_under_writable_root(&common, &shared)
    );
}

/// A submodule gitfile is checked like the root one; a git dir under
/// the root `.git/modules` is allowed.
#[test]
fn a_submodule_git_dir_outside_the_root_git_is_checked() {
    let (_dir, base, ws) = scratch();
    real_git_dir(&ws.join(GIT_DIR_NAME));
    let modules = ws.join(GIT_DIR_NAME).join(MODULES_DIR);
    std::fs::create_dir_all(modules.join("a")).expect("module a");
    gitfile(
        &ws.join("a"),
        &Path::new("..")
            .join(GIT_DIR_NAME)
            .join(MODULES_DIR)
            .join("a"),
    );
    let elsewhere = base.join("elsewhere");
    std::fs::create_dir_all(elsewhere.join("b")).expect("module b");
    gitfile(&ws.join("b"), &elsewhere.join("b"));
    std::fs::write(
        ws.join(GIT_DIR_NAME).join(INDEX_FILE),
        gitlinks_index(&["a", "b"]),
    )
    .expect("index");

    let refused = check(&ws, &[ws.clone(), elsewhere.clone()]).expect_err("refused");
    let allowed = check(&ws, std::slice::from_ref(&ws));

    assert_eq!(
        refused.to_string(),
        git_refusal::git_dir_under_writable_root(&elsewhere.join("b"), &elsewhere)
    );
    assert!(allowed.is_ok(), "{allowed:?}");
}

/// A gitfile naming a symlink outside the workspace that leads back
/// into it is refused by its resolved path.
#[cfg(unix)]
#[test]
fn a_git_dir_reached_through_a_symlink_is_checked_where_it_resolves() {
    let (_dir, base, ws) = scratch();
    real_git_dir(&ws.join("meta"));
    std::os::unix::fs::symlink(ws.join("meta"), base.join("link")).expect("symlink");
    gitfile(&ws, &base.join("link"));

    let err = check(&ws, std::slice::from_ref(&ws)).expect_err("refused");

    assert_eq!(
        err.to_string(),
        git_refusal::git_dir_in_workspace(&ws.join("meta"))
    );
}

/// A gitfile naming a symlink inside the workspace is refused by the
/// name, though it resolves outside: the sandbox can retarget it.
#[cfg(unix)]
#[test]
fn a_git_dir_named_through_a_workspace_symlink_is_checked_as_named() {
    let (_dir, base, ws) = scratch();
    real_git_dir(&base.join("safe"));
    std::os::unix::fs::symlink(base.join("safe"), ws.join("link")).expect("symlink");
    gitfile(&ws, Path::new("link"));

    let err = check(&ws, std::slice::from_ref(&ws)).expect_err("refused");

    assert_eq!(
        err.to_string(),
        git_refusal::git_dir_in_workspace(&ws.join("link"))
    );
}

/// A gitfile naming a path through a symlink under a writable root is
/// refused at the symlink: the sandbox could retarget it.
#[cfg(unix)]
#[test]
fn a_git_dir_through_a_symlink_under_a_writable_root_is_refused() {
    let (_dir, base, ws) = scratch();
    let elsewhere = base.join("elsewhere");
    real_git_dir(&elsewhere.join("gd"));
    std::os::unix::fs::symlink(&elsewhere, base.join("link")).expect("symlink");
    gitfile(&ws, &base.join("link").join("gd"));

    let err = check(&ws, &[ws.clone(), base.clone()]).expect_err("refused");

    assert_eq!(
        err.to_string(),
        git_refusal::git_dir_under_writable_root(&base.join("link"), &base)
    );
}

/// Every directory leading to a submodule is guarded, checked out or
/// not, nested ones included. A nested `.git` that the index does not list
/// is not: the profile's name rule denies writes to it.
#[test]
fn guarded_dirs_cover_every_submodule_path_and_no_other_nested_repo() {
    let dir = superproject("2", "sha1", false);
    let top = dir.path().join("top");
    std::fs::create_dir_all(top.join("vendor").join("dep").join(GIT_DIR_NAME)).expect("nested");

    let dirs = guarded_dirs(&top).expect("guarded");

    let expected: Vec<PathBuf> = ["libs", "libs/a", "libs/b"]
        .iter()
        .map(PathBuf::from)
        .collect();
    assert_eq!(dirs, expected);
}

/// The targets come from the indexes alone: the root `.git` and each
/// submodule's, checked out or not, and no nested repository the index
/// does not list.
#[test]
fn index_targets_read_only_the_indexes() {
    let dir = superproject("2", "sha1", false);
    let top = dir.path().join("top");

    let targets = index_targets(&top).expect("targets");

    let expected: Vec<PathBuf> = ["", "libs/a", "libs/b"]
        .iter()
        .map(|dir| Path::new(dir).join(GIT_DIR_NAME))
        .collect();
    assert_eq!(targets, expected);
}

/// A submodule's own submodule is guarded too, read from the module's
/// git dir when the outer submodule is not checked out.
#[test]
fn nested_submodules_are_guarded_checked_out_or_not() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path();
    for repo in ["inner", "mid", "top"] {
        std::fs::create_dir(root.join(repo)).expect("repo dir");
        git(&root.join(repo), &["init", "-q"]);
    }
    git(
        &root.join("inner"),
        &["commit", "-q", "--allow-empty", "-m", "init"],
    );
    git(
        &root.join("mid"),
        &["submodule", "add", "-q", "../inner", "deps/inner"],
    );
    git(&root.join("mid"), &["commit", "-q", "-m", "add"]);
    let top = root.join("top");
    git(&top, &["submodule", "add", "-q", "../mid", "m"]);
    git(
        &top,
        &["submodule", "update", "-q", "--init", "--recursive"],
    );
    let checked_out = guarded_dirs(&top).expect("checked out");
    git(&top, &["submodule", "deinit", "-q", "-f", "m"]);

    let not_checked_out = guarded_dirs(&top).expect("not checked out");

    let expected: Vec<PathBuf> = ["m", "m/deps", "m/deps/inner"]
        .iter()
        .map(PathBuf::from)
        .collect();
    assert_eq!(checked_out, expected);
    assert_eq!(not_checked_out, expected);
}

/// A symlinked directory on the path to a submodule is refused.
#[cfg(unix)]
#[test]
fn a_symlink_on_the_path_to_a_submodule_is_refused() {
    let dir = superproject("2", "sha1", false);
    let top = dir.path().join("top");
    std::fs::rename(top.join("libs"), dir.path().join("real-libs")).expect("move libs");
    std::os::unix::fs::symlink(dir.path().join("real-libs"), top.join("libs")).expect("symlink");

    let err = guarded_dirs(&top).expect_err("refused");

    assert_eq!(
        err.to_string(),
        git_refusal::symlink_leads_to_git(Path::new("libs"))
    );
}

/// The targets come from the indexes alone, so a directory nobody can
/// list does not stop them from being read.
#[cfg(unix)]
#[test]
fn a_workspace_with_an_unlistable_directory_has_index_targets() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::TempDir::new().expect("tempdir");
    let locked = dir.path().join("locked");
    std::fs::create_dir(&locked).expect("locked");
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).expect("chmod 000");
    std::fs::read_dir(&locked).expect_err("mode 000 is unlistable");

    let targets = index_targets(dir.path()).map_err(|err| err.to_string());

    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).expect("chmod back");
    assert_eq!(targets, Ok(vec![PathBuf::from(GIT_DIR_NAME)]));
}

/// An index with one gitlink entry, version 2, SHA-1, at `path`.
pub(crate) fn gitlink_index(path: &str) -> Vec<u8> {
    gitlinks_index(&[path])
}

/// An index with a gitlink entry at each of `paths` (sorted), version 2,
/// SHA-1.
pub(crate) fn gitlinks_index(paths: &[&str]) -> Vec<u8> {
    let mut data = b"DIRC".to_vec();
    data.extend_from_slice(&2u32.to_be_bytes());
    data.extend_from_slice(&u32::try_from(paths.len()).expect("few paths").to_be_bytes());
    for path in paths {
        let start = data.len();
        data.extend_from_slice(&[0; 24]);
        data.extend_from_slice(&GITLINK_MODE.to_be_bytes());
        data.extend_from_slice(&[0; 12]);
        data.extend_from_slice(&[0; 20]);
        let name_len = u16::try_from(path.len()).expect("short path");
        data.extend_from_slice(&name_len.to_be_bytes());
        data.extend_from_slice(path.as_bytes());
        let padded = (data.len() - start + 8) & !7;
        data.resize(start + padded, 0);
    }
    data.extend_from_slice(&[0; 20]);
    data
}

/// Submodule git dirs nested deeper than the limit, each one's index
/// naming the next under its `modules`, are refused rather than
/// followed without end.
#[test]
fn submodules_nested_past_the_limit_are_refused() {
    let (_dir, _base, ws) = scratch();
    let mut git_dir = ws.join(GIT_DIR_NAME);
    for _ in 0..=MAX_SUBMODULE_DEPTH + 1 {
        std::fs::create_dir_all(&git_dir).expect("git dir");
        std::fs::write(git_dir.join(INDEX_FILE), gitlink_index("m")).expect("index");
        git_dir = git_dir.join(MODULES_DIR).join("m");
    }
    let deepest: PathBuf = std::iter::repeat_n("m", MAX_SUBMODULE_DEPTH + 1).collect();

    let err = guarded_dirs(&ws).expect_err("too deep");

    assert_eq!(err.to_string(), git_refusal::submodules_too_deep(&deepest));
}

/// A submodule path in the index that is not plain names a directory
/// outside the one it claims, so it is refused.
#[test]
fn a_submodule_path_that_is_not_plain_is_refused() {
    let (_dir, _base, ws) = scratch();
    std::fs::create_dir(ws.join(GIT_DIR_NAME)).expect("git dir");
    std::fs::write(
        ws.join(GIT_DIR_NAME).join(INDEX_FILE),
        gitlink_index("a/../b"),
    )
    .expect("index");

    let err = guarded_dirs(&ws).expect_err("not plain");

    assert_eq!(
        err.to_string(),
        git_refusal::path_not_plain(Path::new("a/../b"))
    );
}

/// git writes the index entry offset table before `link`, so every
/// extension is read, not only the first.
#[test]
fn a_split_index_after_another_extension_is_refused() {
    let mut data = b"DIRC".to_vec();
    data.extend_from_slice(&2u32.to_be_bytes());
    data.extend_from_slice(&0u32.to_be_bytes());
    data.extend_from_slice(b"IEOT");
    data.extend_from_slice(&4u32.to_be_bytes());
    data.extend_from_slice(&[0; 4]);
    data.extend_from_slice(SPLIT_INDEX_EXTENSION);
    data.extend_from_slice(&0u32.to_be_bytes());
    data.extend_from_slice(&[0; 20]);

    let err = gitlinks(&data, 20).expect_err("split index");

    assert_eq!(err.to_string(), git_refusal::SPLIT_INDEX);
}

/// `gitdir:  /usr` names `<ws>/ /usr`, not `/usr`.
#[test]
fn a_gitfile_with_two_spaces_is_checked_at_the_spaced_path() {
    let (_dir, _base, ws) = scratch();
    let hidden = ws.join(" ").join("usr");
    real_git_dir(&hidden);
    std::fs::write(ws.join(GIT_DIR_NAME), "gitdir:  /usr\n").expect("gitfile");

    let err = check(&ws, std::slice::from_ref(&ws)).expect_err("refused");

    assert_eq!(err.to_string(), git_refusal::git_dir_in_workspace(&hidden));
}

/// `gitdir: x ` names `x `, including the trailing space.
#[test]
fn a_gitfile_with_a_trailing_space_is_checked_at_that_path() {
    let (_dir, _base, ws) = scratch();
    let hidden = ws.join("x ");
    real_git_dir(&hidden);
    std::fs::write(ws.join(GIT_DIR_NAME), "gitdir: x \n").expect("gitfile");

    let err = check(&ws, std::slice::from_ref(&ws)).expect_err("refused");

    assert_eq!(err.to_string(), git_refusal::git_dir_in_workspace(&hidden));
}

/// A zero-length `commondir` is not a git dir. Git dies when it reads
/// one. A file of only a newline still names the git dir, as git does.
#[test]
fn an_empty_commondir_is_not_a_git_dir() {
    let (_dir, _base, ws) = scratch();
    let git = ws.join(GIT_DIR_NAME);
    real_git_dir(&git);
    std::fs::write(git.join(COMMONDIR_FILE), "").expect("commondir");

    assert_eq!(discover(&ws), Discovery::InvalidGitEntry(git.clone()));

    std::fs::write(git.join(COMMONDIR_FILE), "\n").expect("newline");
    assert_eq!(
        discover(&ws),
        Discovery::Repository {
            dir: ws,
            git_dir: git,
        }
    );
}

/// A trailing space in `commondir` is part of the path.
#[test]
fn a_commondir_with_a_trailing_space_is_checked_at_that_path() {
    let (_dir, base, ws) = scratch();
    let git_dir = base.join("private").join("gd");
    let common = ws.join("store ");
    std::fs::create_dir_all(&git_dir).expect("git dir");
    real_git_dir(&common);
    std::fs::write(git_dir.join(HEAD_FILE), LOOSE_HEAD_CONTENTS).expect("HEAD");
    std::fs::write(git_dir.join(COMMONDIR_FILE), "../../ws/store \n").expect("commondir");
    gitfile(&ws, &git_dir);

    let err = check(&ws, std::slice::from_ref(&ws)).expect_err("refused");

    assert_eq!(err.to_string(), git_refusal::git_dir_in_workspace(&common));
}

/// A `HEAD`, `objects`, `refs`, or `commondir` symlink inside `.git` is
/// in the workspace, so the sandbox could retarget it.
#[cfg(unix)]
#[test]
fn an_objects_symlink_into_the_workspace_is_refused() {
    // These four names are the whole of `GIT_DIR_LINKS`.
    for name in [HEAD_FILE, OBJECTS_DIR, REFS_DIR, COMMONDIR_FILE] {
        let (_dir, _base, ws) = scratch();
        let git = ws.join(GIT_DIR_NAME);
        real_git_dir(&git);
        plant_git_dir_link(&git, &ws, name);

        let err = check(&ws, std::slice::from_ref(&ws)).expect_err(name);

        assert_eq!(
            err.to_string(),
            git_refusal::git_dir_in_workspace(&git.join(name)),
            "{name}"
        );
    }
    assert_eq!(
        GIT_DIR_LINKS,
        [HEAD_FILE, OBJECTS_DIR, REFS_DIR, COMMONDIR_FILE]
    );
}

/// A linked worktree's common dir `objects` points into the workspace.
/// Only the common-dir link check sees that symlink.
#[cfg(unix)]
#[test]
fn a_common_dir_objects_symlink_into_the_workspace_is_refused() {
    let (_dir, base, ws) = scratch();
    let common = base.join("main").join(GIT_DIR_NAME);
    let git_dir = plant_linked_worktree(&common, "ws").expect("linked worktree");
    gitfile(&ws, &git_dir);
    let store = ws.join("store");
    std::fs::create_dir(&store).expect("store");
    std::fs::remove_dir(common.join(OBJECTS_DIR)).expect("objects");
    std::os::unix::fs::symlink(&store, common.join(OBJECTS_DIR)).expect("link");

    let err = check(&ws, std::slice::from_ref(&ws)).expect_err("refused");

    assert_eq!(err.to_string(), git_refusal::git_dir_in_workspace(&store));
}

/// A linked worktree's own `HEAD` resolves into the workspace. The
/// common dir's `HEAD` is a normal file, so only the git-dir link
/// check sees this symlink.
#[cfg(unix)]
#[test]
fn a_worktree_head_symlink_into_the_workspace_is_refused() {
    let (_dir, base, ws) = scratch();
    let common = base.join("main").join(GIT_DIR_NAME);
    let git_dir = plant_linked_worktree(&common, "ws").expect("linked worktree");
    gitfile(&ws, &git_dir);
    let planted = ws.join("planted");
    std::fs::create_dir(&planted).expect("planted");
    let text = head_symlink_text(&git_dir, &planted);
    std::fs::remove_file(git_dir.join(HEAD_FILE)).expect("head");
    std::os::unix::fs::symlink(&text, git_dir.join(HEAD_FILE)).expect("link");

    let err = check(&ws, std::slice::from_ref(&ws)).expect_err("refused");

    assert_eq!(err.to_string(), git_refusal::git_dir_in_workspace(&planted));
}

/// A git dir outside the workspace whose `HEAD` is a symlink to a
/// missing `refs/heads/…` names that link. Git accepts the unborn branch.
#[cfg(unix)]
#[test]
fn a_dangling_head_symlink_to_an_unborn_branch_names_the_link() {
    let (_dir, base, ws) = scratch();
    let git = base.join("main").join(GIT_DIR_NAME);
    real_git_dir(&git);
    gitfile(&ws, &git);
    std::fs::remove_file(git.join(HEAD_FILE)).expect("head");
    std::os::unix::fs::symlink("refs/heads/unborn", git.join(HEAD_FILE)).expect("link");

    let err = check(&ws, std::slice::from_ref(&ws)).expect_err("refused");

    assert_eq!(
        err.to_string(),
        git_refusal::dangling_head(&git.join(HEAD_FILE))
    );
}

/// Replace `git`'s `name` entry with a symlink the workspace can delete.
/// The git dir stays one that discovery accepts.
#[cfg(unix)]
fn plant_git_dir_link(git: &Path, ws: &Path, name: &str) {
    let link = git.join(name);
    match name {
        HEAD_FILE => {
            let store = ws.join("store");
            std::fs::create_dir(&store).expect("store");
            std::fs::remove_file(&link).expect("head");
            std::os::unix::fs::symlink(head_symlink_text(git, &store), &link).expect("link");
        }
        OBJECTS_DIR | REFS_DIR => {
            std::fs::create_dir(ws.join("store")).expect("store");
            std::fs::remove_dir(&link).expect("dir");
            std::os::unix::fs::symlink("../store", &link).expect("link");
        }
        COMMONDIR_FILE => {
            std::fs::write(ws.join("cd"), ".\n").expect("cd");
            std::os::unix::fs::symlink("../cd", &link).expect("link");
        }
        other => panic!("unknown git dir link {other}"),
    }
}

/// Symlink text that starts with `refs/` and, from `git_dir`, names `target`.
#[cfg(unix)]
fn head_symlink_text(git_dir: &Path, target: &Path) -> PathBuf {
    let mut cursor = git_dir.join(REFS_DIR);
    let mut up = PathBuf::new();
    while !target.starts_with(&cursor) {
        cursor = cursor
            .parent()
            .expect("git dir and target share a root")
            .to_path_buf();
        up.push("..");
    }
    let rest = target.strip_prefix(&cursor).expect("prefix");
    let mut text = PathBuf::from(REFS_DIR);
    if !up.as_os_str().is_empty() {
        text.push(up);
    }
    if !rest.as_os_str().is_empty() {
        text.push(rest);
    }
    text
}

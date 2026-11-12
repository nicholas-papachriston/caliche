//! Tool and handler code reaches the file system only through the tool
//! file-access module, and opens a repository or runs `git` only through
//! the checked repository.
//!
//! The file-access module (`src/agent/tools/common.rs`, `common::path`)
//! resolves a path against the tool workspace and opens it without following
//! a link. A raw `std::fs` or `tokio::fs` call, or a `File::open`, in a tool
//! or handler would read or write wherever a link swapped in after the check
//! points. The checked repository (`caliche_sandbox::git_access`) refuses a
//! `.git` that names a git dir outside the workspace before git reads it.
//!
//! A use that reaches no workspace path is listed below with the reason it
//! is safe. An entry that no longer matches a line fails the scan, so the
//! lists stay current.

use std::path::{Path, PathBuf};

/// Folders holding tool and handler code.
const TOOL_AND_HANDLER_DIRS: [&str; 3] = [
    "src/agent/tools",
    "caliche_server/src/handlers",
    "caliche_cli/src/handlers",
];

/// The tool file-access module.
const FILE_ACCESS_MODULE: &str = "src/agent/tools/common.rs";

/// Text that reaches the file system without the module.
const RAW_FILE_ACCESS: [&str; 6] = [
    "std::fs",
    "tokio::fs",
    "File::open",
    "File::create",
    "OpenOptions",
    "DirBuilder",
];

/// Folders whose first-party code may open a repository or run `git`.
const FIRST_PARTY_DIRS: [&str; 5] = [
    "src",
    "crates",
    "caliche_cli/src",
    "caliche_server/src",
    "caliche_sandbox/src",
];

/// The modules that check a repository before git reads it.
const GIT_ACCESS_MODULES: [&str; 2] = [
    "caliche_sandbox/src/git_access.rs",
    "src/agent/tools/git/mod.rs",
];

/// Text that opens a repository or starts `git`.
const RAW_GIT_ACCESS: [&str; 5] = [
    "Repository::open",
    "Repository::discover",
    "Repository::open_ext",
    "Repository::open_bare",
    "Command::new(\"git\")",
];

/// A reviewed use: the file, text on the line, and why it is safe.
type Allowed = (&'static str, &'static str, &'static str);

/// Reviewed raw file access in tool and handler code. None of these paths
/// comes from the workspace or the model.
const ALLOWED_FILE_ACCESS: [Allowed; 16] = [
    (
        "src/agent/tools/bash.rs",
        "OpenOptions",
        "spill file in this process's own 0700 temp dir, created new",
    ),
    (
        "src/agent/tools/bash.rs",
        "DirBuilder",
        "this process's own spill dir under the temp dir, random name",
    ),
    (
        "caliche_server/src/handlers/documents.rs",
        "std::fs::",
        "the server's own upload temp file, random name",
    ),
    (
        "caliche_server/src/handlers/gateway.rs",
        "std::fs::create_dir_all(parent)",
        "parent of the server's own ~/.caliche cluster database",
    ),
    (
        "caliche_server/src/handlers/endpoints.rs",
        "tokio::fs::",
        "the server's own endpoint persistence file",
    ),
    (
        "caliche_server/src/handlers/mcp_config.rs",
        "tokio::fs::",
        "the server's own MCP config file",
    ),
    (
        "caliche_cli/src/handlers/shell.rs",
        "std::fs::",
        "completion scripts written where the user's own command names",
    ),
    (
        "caliche_cli/src/handlers/config.rs",
        "std::fs::",
        "the user's own config file in the config dir",
    ),
    (
        "caliche_cli/src/handlers/config.rs",
        "OpenOptionsExt",
        "the owner-only mode for that config file",
    ),
    (
        "caliche_cli/src/handlers/skills.rs",
        "std::fs::read_to_string(&lock_path)",
        "the skills lock in the user's own skill root",
    ),
    (
        "caliche_cli/src/handlers/git.rs",
        "tokio::fs::",
        "the commit message temp file this command created",
    ),
    (
        "caliche_cli/src/handlers/editor.rs",
        "tokio::fs::",
        "the editor temp file this command created",
    ),
    (
        "caliche_cli/src/handlers/session_actions.rs",
        "tokio::fs::write(&output_path",
        "a session export the user's own command names",
    ),
    (
        "caliche_cli/src/handlers/host_git.rs",
        "std::fs::symlink_metadata(&dot_git)",
        "reads the link itself, never its target",
    ),
    (
        "caliche_cli/src/handlers/init.rs",
        "fs::",
        "only a path the user named outside the working directory; inside it the no-link walk reads and writes",
    ),
    (
        "caliche_cli/src/handlers/interactive/context.rs",
        "std::fs::read_dir(",
        "names for completing the path the user is typing; nothing is read",
    ),
];

/// Reviewed git access outside the checked repository.
const ALLOWED_GIT_ACCESS: [Allowed; 4] = [
    (
        "caliche_cli/src/handlers/host_git.rs",
        "Command::new(\"git\")",
        "the user's own /commit, /pr and /review; HostGit::open refuses a linked or foreign .git and a submodule outside modules",
    ),
    (
        "caliche_cli/src/config/sources.rs",
        "Command::new(\"git\")",
        "startup reads one config key; the URL it makes is fetched through HttpPolicy",
    ),
    (
        "src/agent/skills/remote/remote_files.rs",
        "Command::new(\"git\")",
        "clones into and reads a temp checkout this installer created",
    ),
    (
        "src/workspace/declaration.rs",
        "Command::new(\"git\")",
        "clones a declared repository into a new directory",
    ),
];

/// Folders that hold no first-party Rust source.
const SKIPPED_DIRS: [&str; 6] = [
    "target",
    "node_modules",
    "shotput",
    ".git",
    "third_party",
    "tests",
];

fn workspace_root() -> PathBuf {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    crate_dir
        .ancestors()
        .nth(2)
        .map_or_else(|| crate_dir.to_path_buf(), Path::to_path_buf)
}

fn rust_files(dir: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if kind.is_dir() {
            if !SKIPPED_DIRS.iter().any(|skip| name == *skip) {
                rust_files(&path, found);
            }
        } else if kind.is_file() && path.extension().is_some_and(|ext| ext == "rs") {
            found.push(path);
        }
    }
}

/// Whether `line` opens a test-only `cfg`: `test`, or `all(test, ...)`.
fn is_test_cfg(line: &str) -> bool {
    line.starts_with("#[cfg(test)]") || line.starts_with("#[cfg(all(test")
}

/// Whether the module file `file` is declared test-only in its parent.
fn is_test_module_file(file: &Path) -> bool {
    let Some(stem) = file.file_stem().and_then(|stem| stem.to_str()) else {
        return false;
    };
    if stem.ends_with("_tests") || stem == "tests" {
        return true;
    }
    let Some(dir) = file.parent() else {
        return false;
    };
    let declaration = format!("mod {stem};");
    [
        dir.join("mod.rs"),
        dir.join("lib.rs"),
        dir.with_extension("rs"),
    ]
    .iter()
    .filter_map(|parent| std::fs::read_to_string(parent).ok())
    .any(|text| {
        let mut test_cfg = false;
        for line in text.lines().map(str::trim) {
            if line.ends_with(&declaration) {
                return test_cfg;
            }
            if is_test_cfg(line) {
                test_cfg = true;
            } else if !line.starts_with("#[") && !line.starts_with("//") {
                test_cfg = false;
            }
        }
        false
    })
}

/// The lines of `text` outside inline test modules, with their numbers.
fn non_test_lines(text: &str) -> Vec<(usize, &str)> {
    let mut kept = Vec::new();
    let mut depth: i64 = 0;
    let mut skip_to: Option<i64> = None;
    let mut test_cfg = false;
    for (index, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        let opened = i64::try_from(line.matches('{').count()).unwrap_or(0);
        let closed = i64::try_from(line.matches('}').count()).unwrap_or(0);
        if let Some(level) = skip_to {
            depth += opened - closed;
            if depth <= level {
                skip_to = None;
            }
            continue;
        }
        if is_test_cfg(trimmed) {
            test_cfg = true;
            continue;
        }
        let module = trimmed
            .trim_start_matches("pub(crate) ")
            .trim_start_matches("pub ");
        if test_cfg && module.starts_with("mod ") && trimmed.ends_with('{') {
            skip_to = Some(depth);
            depth += opened - closed;
            test_cfg = false;
            continue;
        }
        if test_cfg && (trimmed.starts_with("#[") || trimmed.starts_with("//")) {
            continue;
        }
        test_cfg = false;
        depth += opened - closed;
        kept.push((index + 1, line));
    }
    kept
}

/// One scan: where to look, what to find, and what is allowed.
struct Scan<'a> {
    dirs: &'a [&'a str],
    exempt: &'a [&'a str],
    patterns: &'a [&'a str],
    allowed: &'a [Allowed],
}

/// The `path:line` hits no entry allows, then the entries that allow none.
fn scan(root: &Path, rules: &Scan<'_>) -> (Vec<String>, Vec<String>) {
    let mut files = Vec::new();
    for dir in rules.dirs {
        rust_files(&root.join(dir), &mut files);
    }
    let mut used = vec![false; rules.allowed.len()];
    let mut hits = Vec::new();
    for file in files {
        let shown = file
            .strip_prefix(root)
            .unwrap_or(&file)
            .to_string_lossy()
            .replace('\\', "/");
        if rules.exempt.contains(&shown.as_str()) || is_test_module_file(&file) {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        for (number, line) in non_test_lines(&text) {
            let code = line.split("//").next().unwrap_or_default();
            let compact: String = code.chars().filter(|c| !c.is_whitespace()).collect();
            if !rules
                .patterns
                .iter()
                .any(|pattern| compact.contains(pattern))
            {
                continue;
            }
            let allowed = rules
                .allowed
                .iter()
                .position(|(path, text, _)| shown == *path && code.contains(text));
            match allowed {
                Some(entry) => used[entry] = true,
                None => hits.push(format!("{shown}:{number}")),
            }
        }
    }
    hits.sort();
    let stale = rules
        .allowed
        .iter()
        .zip(used)
        .filter(|(_, used)| !used)
        .map(|((path, text, _), _)| format!("{path}: {text}"))
        .collect();
    (hits, stale)
}

const FILE_SCAN: Scan<'static> = Scan {
    dirs: &TOOL_AND_HANDLER_DIRS,
    exempt: &[FILE_ACCESS_MODULE],
    patterns: &RAW_FILE_ACCESS,
    allowed: &ALLOWED_FILE_ACCESS,
};

const GIT_SCAN: Scan<'static> = Scan {
    dirs: &FIRST_PARTY_DIRS,
    exempt: &GIT_ACCESS_MODULES,
    patterns: &RAW_GIT_ACCESS,
    allowed: &ALLOWED_GIT_ACCESS,
};

fn checked_root() -> PathBuf {
    let root = workspace_root();
    assert!(
        root.join("Cargo.toml").is_file() && root.join("src/agent/tools").is_dir(),
        "workspace root not found from {}",
        root.display()
    );
    root
}

#[test]
fn tool_and_handler_code_reaches_files_only_through_the_module() {
    assert_eq!(
        scan(&checked_root(), &FILE_SCAN),
        (Vec::new(), Vec::new()),
        "use agent::tools::common::path, or review the use and list it with its reason"
    );
}

#[test]
fn git_is_opened_or_run_only_through_the_checked_repository() {
    assert_eq!(
        scan(&checked_root(), &GIT_SCAN),
        (Vec::new(), Vec::new()),
        "use caliche_sandbox::git_access, or review the use and list it with its reason"
    );
}

#[test]
fn the_scan_finds_raw_access_outside_tests_and_reports_stale_entries() {
    let dir = tempfile_dir("caliche-tool-file-scan");
    let tools = dir.join("tools");
    std::fs::create_dir_all(&tools).ok();
    std::fs::write(
        tools.join("a.rs"),
        "fn a(p: &str) {\n    let _ = std::fs::read( p );\n}\n#[cfg(test)]\n/// doc\nmod tests {\n    fn t() { std::fs::read(\"x\").ok(); }\n}\nfn b() { tokio::fs::write(p, \"\"); }\n",
    )
    .ok();
    std::fs::write(tools.join("mod.rs"), "#[cfg(test)]\nmod fixture;\nmod c;\n").ok();
    std::fs::write(
        tools.join("fixture.rs"),
        "fn f() { std::fs::read(\"x\").ok(); }\n",
    )
    .ok();
    std::fs::write(tools.join("c.rs"), "fn c() { File::open(p); }\n").ok();
    std::fs::write(tools.join("module.rs"), "fn m() { std::fs::read(p); }\n").ok();
    let rules = Scan {
        dirs: &["tools"],
        exempt: &["tools/module.rs"],
        patterns: &RAW_FILE_ACCESS,
        allowed: &[
            ("tools/c.rs", "File::open(p)", "fixture"),
            ("tools/c.rs", "File::create", "fixture"),
        ],
    };
    let found = scan(&dir, &rules);
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(
        found,
        (
            vec!["tools/a.rs:2".to_string(), "tools/a.rs:9".to_string()],
            vec!["tools/c.rs: File::create".to_string()],
        )
    );
}

fn tempfile_dir(prefix: &str) -> PathBuf {
    std::env::temp_dir().join(format!("{prefix}-{}", std::process::id()))
}

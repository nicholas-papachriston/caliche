//! Edit and search hotpath timing plus instrumentation tests.

#![expect(
    clippy::await_holding_lock,
    reason = "each test holds the global perf lock while the tool under test runs"
)]

use super::apply_patch::{ApplyPatchTool, apply_hunks_to_cwd};
use super::edit_file::EditFileTool;
use super::fs_tool::FsTool;
use super::tool::Tool;
use crate::observability::perf;
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

fn fixture_root(name: &str) -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("edit-search-hotpath")
        .join(name);
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("fixture dir");
    root
}

fn write_search_corpus(root: &Path, files: usize, lines: usize) {
    for file_idx in 0..files {
        let mut body = String::new();
        for line_idx in 0..lines {
            if file_idx % 8 == 0 && line_idx == 3 {
                body.push_str("hotpath_token unique_marker\n");
            } else {
                body.push_str("plain line of searchable text\n");
            }
        }
        fs::write(root.join(format!("file_{file_idx:03}.rs")), body).expect("write corpus file");
    }
}

#[tokio::test]
async fn grep_directory_stays_under_budget_when_not_measuring() {
    let _lock = perf::lock_for_test();
    perf::configure(false);
    perf::clear();
    let root = fixture_root("grep-budget");
    write_search_corpus(&root, 48, 80);
    let before = Instant::now();
    let result = FsTool::new()
        .execute(json!({
            "op": "grep",
            "path": root.to_str().expect("utf8 path"),
            "pattern": "hotpath_token",
            "context_lines": 0,
            "max_matches": 50
        }))
        .await;
    let elapsed = before.elapsed();
    assert!(result.success, "{}", result.error.unwrap_or_default());
    let match_count = result
        .typed_payload
        .as_ref()
        .and_then(|payload| payload.get("match_count"))
        .and_then(serde_json::Value::as_u64)
        .expect("match_count");
    assert!(match_count >= 6, "expected several hits, got {match_count}");
    assert!(
        elapsed < Duration::from_millis(750),
        "fs grep took {elapsed:?}"
    );
    assert_eq!(
        perf::event_count(),
        0,
        "disabled gate must not record events"
    );
}

#[tokio::test]
async fn read_skips_stage_timing_payload_when_not_measuring() {
    let _lock = perf::lock_for_test();
    perf::configure(false);
    let root = fixture_root("read-budget");
    let path = root.join("note.rs");
    fs::write(&path, "fn main() {}\n").expect("write");
    let result = super::read_file::ReadFileTool
        .execute(json!({
            "path": path.to_str().expect("utf8 path")
        }))
        .await;
    assert!(result.success, "{}", result.error.unwrap_or_default());
    let payload = result.typed_payload.expect("payload");
    assert!(
        payload.get("stage_timings_ms").is_none(),
        "measurement-off reads must not attach stage timings: {payload}"
    );
}

#[tokio::test]
async fn grep_records_granular_spans_when_measuring() {
    let _lock = perf::lock_for_test();
    perf::configure(true);
    perf::clear();
    let root = fixture_root("grep-spans");
    write_search_corpus(&root, 16, 40);
    let result = FsTool::new()
        .execute(json!({
            "op": "grep",
            "path": root.to_str().expect("utf8 path"),
            "pattern": "hotpath_token",
            "context_lines": 0
        }))
        .await;
    assert!(result.success, "{}", result.error.unwrap_or_default());
    let names: Vec<&'static str> = perf::snapshot()
        .into_iter()
        .map(|event| event.name)
        .collect();
    assert!(
        names.iter().any(|name| *name == "grep.search"),
        "missing grep.search in {names:?}"
    );
    assert!(
        names.iter().any(|name| *name == "fs.grep"),
        "missing fs.grep in {names:?}"
    );
    perf::configure(false);
    perf::clear();
}

#[tokio::test]
async fn edit_str_replace_stays_under_budget_when_not_measuring() {
    let _lock = perf::lock_for_test();
    perf::configure(false);
    perf::clear();
    let root = fixture_root("edit-budget");
    let path = root.join("large.rs");
    let mut body = String::new();
    for idx in 0..2_000 {
        body.push_str(&format!("let value_{idx} = {idx};\n"));
    }
    body.push_str("let target = REPLACE_ME;\n");
    fs::write(&path, &body).expect("write edit fixture");

    let before = Instant::now();
    let result = EditFileTool
        .execute(json!({
            "path": path.to_str().expect("utf8 path"),
            "operations": [{
                "type": "str_replace",
                "old_string": "REPLACE_ME",
                "new_string": "REPLACED"
            }]
        }))
        .await;
    let elapsed = before.elapsed();
    assert!(result.success, "{}", result.error.unwrap_or_default());
    let updated = fs::read_to_string(&path).expect("read edited file");
    assert!(updated.contains("REPLACED"));
    assert!(
        elapsed < Duration::from_millis(500),
        "edit.apply took {elapsed:?}"
    );
    assert_eq!(perf::event_count(), 0);
}

#[tokio::test]
async fn apply_patch_update_records_when_measuring() {
    let _lock = perf::lock_for_test();
    perf::configure(true);
    perf::clear();
    let root = fixture_root("apply-patch");
    let path = root.join("note.txt");
    fs::write(&path, "hello\nworld\n").expect("write patch fixture");
    let patch = format!(
        "*** Begin Patch\n*** Update File: {}\n@@\n-hello\n+HELLO\n*** End Patch\n",
        path.display()
    );
    let before = Instant::now();
    let summary = apply_hunks_to_cwd(&patch, None).await.expect("apply patch");
    let elapsed = before.elapsed();
    assert!(summary.contains("Updated"));
    assert!(
        elapsed < Duration::from_millis(500),
        "apply_patch took {elapsed:?}"
    );

    let result = ApplyPatchTool::new()
        .execute(json!({
            "patch": format!(
                "*** Begin Patch\n*** Update File: {}\n@@\n-world\n+WORLD\n*** End Patch\n",
                path.display()
            )
        }))
        .await;
    assert!(result.success, "{}", result.error.unwrap_or_default());
    let names: Vec<&'static str> = perf::snapshot()
        .into_iter()
        .map(|event| event.name)
        .collect();
    assert!(
        names.iter().any(|name| *name == "apply_patch"),
        "missing apply_patch in {names:?}"
    );
    perf::configure(false);
    perf::clear();
}

#[tokio::test]
async fn glob_search_stays_under_budget_when_not_measuring() {
    let _lock = perf::lock_for_test();
    perf::configure(false);
    let root = fixture_root("glob-budget");
    write_search_corpus(&root, 32, 4);
    let before = Instant::now();
    let result = FsTool::new()
        .execute(json!({
            "op": "glob",
            "path": root.to_str().expect("utf8 path"),
            "pattern": "*.rs",
            "max_matches": 100
        }))
        .await;
    let elapsed = before.elapsed();
    assert!(result.success, "{}", result.error.unwrap_or_default());
    assert!(
        elapsed < Duration::from_millis(500),
        "fs glob took {elapsed:?}"
    );
}

#[tokio::test]
async fn grep_skips_vcs_metadata_trees() {
    let _lock = perf::lock_for_test();
    perf::configure(false);
    let root = fixture_root("grep-skip-git");
    fs::write(root.join("keep.rs"), "hotpath_token in source\n").expect("write keep");
    let git_dir = root.join(".git");
    fs::create_dir_all(&git_dir).expect("mkdir .git");
    fs::write(git_dir.join("HEAD"), "hotpath_token in git metadata\n").expect("write HEAD");
    let result = FsTool::new()
        .execute(json!({
            "op": "grep",
            "path": root.to_str().expect("utf8 path"),
            "pattern": "hotpath_token",
            "context_lines": 0
        }))
        .await;
    assert!(result.success, "{}", result.error.unwrap_or_default());
    assert!(
        result.content.contains("keep.rs"),
        "expected source hit: {}",
        result.content
    );
    assert!(
        !result.content.contains(".git"),
        "grep must not scan .git: {}",
        result.content
    );
}

#[tokio::test]
async fn fs_omits_duration_payload_when_not_measuring() {
    let _lock = perf::lock_for_test();
    perf::configure(false);
    let root = fixture_root("fs-no-duration");
    fs::write(root.join("note.rs"), "fn main() {}\n").expect("write");
    let result = FsTool::new()
        .execute(json!({
            "op": "read",
            "path": root.join("note.rs").to_str().expect("utf8 path")
        }))
        .await;
    assert!(result.success, "{}", result.error.unwrap_or_default());
    let payload = result.typed_payload.expect("payload");
    assert!(
        payload.get("duration_ms").is_none(),
        "measurement-off fs payloads must not attach duration_ms: {payload}"
    );
}

#[tokio::test]
async fn grep_streams_context_lines_without_slurping() {
    let _lock = perf::lock_for_test();
    perf::configure(false);
    let root = fixture_root("grep-context");
    fs::write(
        root.join("ctx.rs"),
        "alpha\nbeta\nhotpath_token here\ngamma\ndelta\n",
    )
    .expect("write");
    let result = FsTool::new()
        .execute(json!({
            "op": "grep",
            "path": root.to_str().expect("utf8 path"),
            "pattern": "hotpath_token",
            "context_lines": 2
        }))
        .await;
    assert!(result.success, "{}", result.error.unwrap_or_default());
    assert!(
        result.content.contains("hotpath_token here"),
        "missing match: {}",
        result.content
    );
    assert!(
        result.content.contains("- ") && result.content.contains("+ "),
        "expected before/after context: {}",
        result.content
    );
    assert!(
        result.content.contains("beta") && result.content.contains("gamma"),
        "expected neighboring lines: {}",
        result.content
    );
    let match_at = result
        .content
        .find("hotpath_token here")
        .expect("match line");
    let before_at = result.content.find("beta").expect("before context");
    let after_at = result.content.find("gamma").expect("after context");
    assert!(
        before_at < match_at && match_at < after_at,
        "context must wrap the match (before, match, after): {}",
        result.content
    );
}

#[tokio::test]
async fn list_directory_stays_under_budget_when_not_measuring() {
    let _lock = perf::lock_for_test();
    perf::configure(false);
    let root = fixture_root("list-budget");
    for idx in 0..32 {
        fs::write(root.join(format!("file_{idx:02}.rs")), "fn x() {}\n").expect("write");
    }
    fs::create_dir_all(root.join("subdir")).expect("mkdir");
    fs::write(root.join("subdir").join("nested.rs"), "fn y() {}\n").expect("write nested");
    let before = Instant::now();
    let result = FsTool::new()
        .execute(json!({
            "op": "list",
            "path": root.to_str().expect("utf8 path"),
            "recursive": true,
            "max_depth": 2
        }))
        .await;
    let elapsed = before.elapsed();
    assert!(result.success, "{}", result.error.unwrap_or_default());
    assert!(
        result.content.contains("file_00.rs"),
        "missing listing: {}",
        result.content
    );
    assert!(
        elapsed < Duration::from_millis(400),
        "fs list took {elapsed:?}"
    );
}

#[tokio::test]
async fn list_hidden_files_do_not_consume_max_entries() {
    let _lock = perf::lock_for_test();
    perf::configure(false);
    let root = fixture_root("list-hidden-quota");
    for idx in 0..8 {
        fs::write(root.join(format!(".secret_{idx}")), "x\n").expect("write hidden");
    }
    let result = FsTool::new()
        .execute(json!({
            "op": "list",
            "path": root.to_str().expect("utf8 path"),
            "max_entries": 1,
            "include_hidden": false
        }))
        .await;
    assert!(result.success, "{}", result.error.unwrap_or_default());
    let payload = result.typed_payload.expect("payload");
    assert_eq!(payload["entry_count"], 0);
    assert_eq!(payload["truncated"], false);
    assert!(
        !result.content.contains(".secret_"),
        "hidden files must be omitted: {}",
        result.content
    );
}

#[tokio::test]
async fn grep_file_pattern_rs_does_not_match_yours() {
    let _lock = perf::lock_for_test();
    perf::configure(false);
    let root = fixture_root("grep-ext-filter");
    fs::write(root.join("lib.rs"), "unique_token in rust\n").expect("write rust");
    fs::write(root.join("yours"), "unique_token in name\n").expect("write yours");
    let result = FsTool::new()
        .execute(json!({
            "op": "grep",
            "path": root.to_str().expect("utf8 path"),
            "pattern": "unique_token",
            "file_pattern": "rs",
            "context_lines": 0
        }))
        .await;
    assert!(result.success, "{}", result.error.unwrap_or_default());
    assert!(
        result.content.contains("lib.rs"),
        "expected rust hit: {}",
        result.content
    );
    assert!(
        !result.content.contains("yours"),
        "bare rs must not match a file named yours: {}",
        result.content
    );
}

#[tokio::test]
async fn grep_files_with_matches_reports_paths_not_line_anchors() {
    let _lock = perf::lock_for_test();
    perf::configure(false);
    let root = fixture_root("grep-files-mode");
    fs::write(root.join("hit.rs"), "shared_token one\nshared_token two\n").expect("write hit");
    fs::write(root.join("other.rs"), "shared_token three\n").expect("write other");
    let result = FsTool::new()
        .execute(json!({
            "op": "grep",
            "path": root.to_str().expect("utf8 path"),
            "pattern": "shared_token",
            "mode": "files_with_matches",
            "context_lines": 0
        }))
        .await;
    assert!(result.success, "{}", result.error.unwrap_or_default());
    let payload = result.typed_payload.expect("payload");
    assert_eq!(payload["file_count"], 2);
    let files = payload["files"]
        .as_array()
        .expect("files array")
        .iter()
        .map(|value| value.as_str().expect("file string").to_string())
        .collect::<Vec<_>>();
    assert_eq!(files.len(), 2);
    for file in &files {
        assert!(
            file.ends_with("hit.rs") || file.ends_with("other.rs"),
            "expected a file path, got {file}"
        );
        assert!(
            !file.contains(".rs:"),
            "file path must not include a line number: {file}"
        );
    }
}

#[tokio::test]
async fn grep_and_read_agree_on_line_hashes() {
    let _lock = perf::lock_for_test();
    perf::configure(false);
    let root = fixture_root("hash-agree");
    let path = root.join("note.rs");
    let line = "fn hashed_line() {}";
    fs::write(&path, format!("{line}\n")).expect("write");
    let read = super::read_file::ReadFileTool
        .execute(json!({
            "path": path.to_str().expect("utf8 path")
        }))
        .await;
    assert!(read.success, "{}", read.error.unwrap_or_default());
    let grep = FsTool::new()
        .execute(json!({
            "op": "grep",
            "path": path.to_str().expect("utf8 path"),
            "pattern": "hashed_line",
            "context_lines": 0
        }))
        .await;
    assert!(grep.success, "{}", grep.error.unwrap_or_default());
    let hash = crate::agent::tools::hashline::compute_line_hash(line);
    assert!(
        read.content.contains(&format!("1:{hash}|")),
        "read missing hash {hash}: {}",
        read.content
    );
    assert!(
        grep.content.contains(&format!(":{hash}:")),
        "grep missing hash {hash}: {}",
        grep.content
    );
}

#[tokio::test]
async fn grep_skips_hidden_dirs_and_files_by_default() {
    let _lock = perf::lock_for_test();
    perf::configure(false);
    let root = fixture_root("grep-skip-hidden");
    fs::write(root.join("visible.rs"), "hidden_token in source\n").expect("write visible");
    fs::write(root.join(".secret"), "hidden_token in secret file\n").expect("write secret");
    let hidden_dir = root.join(".cursor");
    fs::create_dir_all(&hidden_dir).expect("mkdir hidden");
    fs::write(hidden_dir.join("notes.rs"), "hidden_token in hidden dir\n").expect("write nested");
    let result = FsTool::new()
        .execute(json!({
            "op": "grep",
            "path": root.to_str().expect("utf8 path"),
            "pattern": "hidden_token",
            "context_lines": 0
        }))
        .await;
    assert!(result.success, "{}", result.error.unwrap_or_default());
    assert!(
        result.content.contains("visible.rs"),
        "expected visible hit: {}",
        result.content
    );
    assert!(
        !result.content.contains(".secret") && !result.content.contains(".cursor"),
        "grep must skip hidden paths by default: {}",
        result.content
    );
}

#[tokio::test]
async fn grep_include_hidden_still_skips_vcs() {
    let _lock = perf::lock_for_test();
    perf::configure(false);
    let root = fixture_root("grep-hidden-keep-git");
    fs::write(root.join(".envrc"), "hidden_token in envrc\n").expect("write envrc");
    let git_dir = root.join(".git");
    fs::create_dir_all(&git_dir).expect("mkdir .git");
    fs::write(git_dir.join("HEAD"), "hidden_token in git metadata\n").expect("write HEAD");
    let result = FsTool::new()
        .execute(json!({
            "op": "grep",
            "path": root.to_str().expect("utf8 path"),
            "pattern": "hidden_token",
            "include_hidden": true,
            "context_lines": 0
        }))
        .await;
    assert!(result.success, "{}", result.error.unwrap_or_default());
    assert!(
        result.content.contains(".envrc"),
        "include_hidden must search hidden files: {}",
        result.content
    );
    assert!(
        !result.content.contains(".git"),
        "include_hidden must still skip VCS trees: {}",
        result.content
    );
}

#[tokio::test]
async fn hashline_replace_relocates_after_concurrent_insert() {
    let _lock = perf::lock_for_test();
    perf::configure(false);
    let root = fixture_root("hashline-concurrent");
    let path = root.join("note.rs");
    fs::write(&path, "alpha_unique\nmiddle\nbeta_unique\n").expect("write");
    let alpha_hash = crate::agent::tools::hashline::compute_line_hash("alpha_unique");
    let beta_hash = crate::agent::tools::hashline::compute_line_hash("beta_unique");
    let path_str = path.to_str().expect("utf8 path").to_string();

    let first = tokio::spawn({
        let path_str = path_str.clone();
        async move {
            EditFileTool
                .execute(json!({
                    "path": path_str,
                    "operations": [{
                        "type": "hashline_replace",
                        "start_line": 1,
                        "start_hash": alpha_hash,
                        "end_line": 1,
                        "end_hash": alpha_hash,
                        "content": "alpha_unique\ninserted_one\ninserted_two"
                    }]
                }))
                .await
        }
    });
    let second = tokio::spawn({
        let path_str = path_str.clone();
        async move {
            EditFileTool
                .execute(json!({
                    "path": path_str,
                    "operations": [{
                        "type": "hashline_replace",
                        "start_line": 3,
                        "start_hash": beta_hash,
                        "end_line": 3,
                        "end_hash": beta_hash,
                        "content": "beta_replaced"
                    }]
                }))
                .await
        }
    });

    let first = first.await.expect("join first");
    let second = second.await.expect("join second");
    assert!(
        first.success,
        "first edit: {}",
        first.error.unwrap_or_default()
    );
    assert!(
        second.success,
        "second edit after concurrent mutation: {}",
        second.error.unwrap_or_default()
    );
    let body = fs::read_to_string(&path).expect("read");
    assert!(body.contains("inserted_one"), "missing insert: {body}");
    assert!(
        body.contains("beta_replaced"),
        "missing relocated replace: {body}"
    );
    assert!(!body.contains("beta_unique"), "stale beta remained: {body}");
}

//! Codex-style freeform `apply_patch` tool.
//!
//! Mutually exclusive with `fs` write/edit when [`MutationStyle::ApplyPatch`] is selected.

mod parser;
mod placement;
mod seek_sequence;
mod tool;

pub use parser::{Hunk, ParseError, UpdateFileChunk, parse_patch};
pub use tool::{ApplyPatchTool, MutationStyle, apply_hunks_to_cwd};

#[cfg(test)]
mod placement_fuzz;

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn parses_add_update_delete_hunks() {
        let patch = r#"*** Begin Patch
*** Add File: hello.txt
+hello
+world
*** Update File: hello.txt
@@
-hello
+HELLO
*** Delete File: gone.txt
*** End Patch
"#;
        let args = parse_patch(patch).expect("parse");
        assert_eq!(args.hunks.len(), 3);
        assert!(matches!(&args.hunks[0], Hunk::AddFile { .. }));
        assert!(matches!(&args.hunks[1], Hunk::UpdateFile { .. }));
        assert!(matches!(&args.hunks[2], Hunk::DeleteFile { .. }));
    }

    #[test]
    fn rejects_missing_boundaries() {
        let err = parse_patch("*** Add File: x\n+y\n").unwrap_err();
        assert!(err.to_string().contains("Begin Patch"));
    }

    #[tokio::test]
    async fn applies_add_update_and_delete() {
        let dir = tempdir().expect("tempdir");
        crate::agent::tools::common::set_tool_workspace_root(Some(dir.path().to_path_buf()));
        let gone = dir.path().join("gone.txt");
        fs::write(&gone, "bye").expect("write gone");

        let patch = format!(
            r#"*** Begin Patch
*** Add File: {add}
+line1
+line2
*** Update File: {add}
@@
-line1
+LINE1
*** Delete File: {del}
*** End Patch
"#,
            add = dir.path().join("hello.txt").display(),
            del = gone.display(),
        );

        let summary = apply_hunks_to_cwd(&patch, None).await.expect("apply");
        assert!(summary.contains("Added"));
        assert!(summary.contains("Updated"));
        assert!(summary.contains("Deleted"));

        let content = fs::read_to_string(dir.path().join("hello.txt")).expect("read");
        assert_eq!(content, "LINE1\nline2\n");
        assert!(!gone.exists());
    }

    #[tokio::test]
    async fn update_requires_unique_context_match() {
        let dir = tempdir().expect("tempdir");
        crate::agent::tools::common::set_tool_workspace_root(Some(dir.path().to_path_buf()));
        let path = dir.path().join("dup.txt");
        fs::write(&path, "same\nsame\n").expect("write");

        let patch = format!(
            r#"*** Begin Patch
*** Update File: {}
@@
-same
+changed
*** End Patch
"#,
            path.display()
        );

        // First occurrence is updated (Codex seeks from start); second remains.
        let _ = apply_hunks_to_cwd(&patch, None).await.expect("apply");
        let content = fs::read_to_string(&path).expect("read");
        assert_eq!(content, "changed\nsame\n");
    }

    #[tokio::test]
    async fn plus_only_hunk_inserts_after_context_not_eof() {
        let dir = tempdir().expect("tempdir");
        crate::agent::tools::common::set_tool_workspace_root(Some(dir.path().to_path_buf()));
        let path = dir.path().join("lib.rs");
        fs::write(&path, "fn alpha() {}\nfn beta() {}\nfn omega() {}\n").expect("write");

        let patch = format!(
            r#"*** Begin Patch
*** Update File: {}
@@ fn beta() {{}}
+fn gamma() {{}}
*** End Patch
"#,
            path.display()
        );

        apply_hunks_to_cwd(&patch, None).await.expect("apply");
        let content = fs::read_to_string(&path).expect("read");
        assert_eq!(
            content,
            "fn alpha() {}\nfn beta() {}\nfn gamma() {}\nfn omega() {}\n"
        );
    }

    #[tokio::test]
    async fn underindented_patch_adopts_file_indent() {
        let dir = tempdir().expect("tempdir");
        crate::agent::tools::common::set_tool_workspace_root(Some(dir.path().to_path_buf()));
        let path = dir.path().join("lib.rs");
        fs::write(&path, "impl Foo {\n    fn a() {}\n}\n").expect("write");

        let patch = format!(
            r#"*** Begin Patch
*** Update File: {}
@@
-    fn a() {{}}
+    fn a() {{}}
+fn b() {{}}
*** End Patch
"#,
            path.display()
        );

        apply_hunks_to_cwd(&patch, None).await.expect("apply");
        let content = fs::read_to_string(&path).expect("read");
        assert_eq!(content, "impl Foo {\n    fn a() {}\n    fn b() {}\n}\n");
    }

    #[tokio::test]
    async fn concurrent_updates_to_the_same_file_both_apply() {
        let dir = tempdir().expect("tempdir");
        crate::agent::tools::common::set_tool_workspace_root(Some(dir.path().to_path_buf()));
        let path = dir.path().join("lib.rs");
        fs::write(&path, "fn alpha() {}\nfn beta() {}\n").expect("write");
        let first = format!(
            r#"*** Begin Patch
*** Update File: {}
@@
-fn alpha() {{}}
+fn alpha_new() {{}}
*** End Patch
"#,
            path.display()
        );
        let second = format!(
            r#"*** Begin Patch
*** Update File: {}
@@
-fn beta() {{}}
+fn beta_new() {{}}
*** End Patch
"#,
            path.display()
        );
        let t1 = tokio::spawn(async move { apply_hunks_to_cwd(&first, None).await });
        let t2 = tokio::spawn(async move { apply_hunks_to_cwd(&second, None).await });
        t1.await.expect("join first").expect("first patch");
        t2.await.expect("join second").expect("second patch");
        let content = fs::read_to_string(&path).expect("read");
        assert!(content.contains("alpha_new"), "{content}");
        assert!(content.contains("beta_new"), "{content}");
    }
}

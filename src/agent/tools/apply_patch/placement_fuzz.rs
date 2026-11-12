//! Property-style fuzz for patch placement, indent matching, and large edits.

use super::apply_hunks_to_cwd;
use std::fs;
use tempfile::tempdir;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        self.0
    }

    fn usize(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() as usize) % n
        }
    }
}

fn indent_unit(kind: usize) -> &'static str {
    match kind % 3 {
        0 => "  ",
        1 => "    ",
        _ => "\t",
    }
}

fn function_block(indent: &str, name: &str, body: u32) -> String {
    format!("{indent}fn {name}() {{\n{indent}{indent}let x = {body};\n{indent}}}\n")
}

fn join_functions(indent: &str, names: &[String], bodies: &[u32]) -> String {
    names
        .iter()
        .zip(bodies.iter())
        .map(|(name, body)| function_block(indent, name, *body))
        .collect()
}

#[tokio::test]
async fn fuzz_context_insert_stays_next_to_anchor() {
    let mut rng = Lcg(0xC0DE_F11E);
    for trial in 0..48 {
        let indent = indent_unit(rng.usize(3));
        let count = 4 + rng.usize(8);
        let names: Vec<String> = (0..count).map(|i| format!("fn_{trial}_{i}")).collect();
        let bodies: Vec<u32> = (0..count).map(|i| u32::try_from(i).unwrap_or(0)).collect();
        let original = join_functions(indent, &names, &bodies);

        let anchor = rng.usize(count);
        let new_name = format!("inserted_{trial}");
        let dir = tempdir().expect("tempdir");
        crate::agent::tools::common::set_tool_workspace_root(Some(dir.path().to_path_buf()));
        let path = dir.path().join("lib.rs");
        fs::write(&path, &original).expect("write");

        let patch = format!(
            "*** Begin Patch\n*** Update File: {}\n@@ {indent}fn {}() {{\n+{indent}fn {new_name}() {{}}\n*** End Patch\n",
            path.display(),
            names[anchor]
        );
        apply_hunks_to_cwd(&patch, None)
            .await
            .unwrap_or_else(|err| panic!("trial {trial} apply failed: {err}"));
        let updated = fs::read_to_string(&path).expect("read");

        let anchor_open = format!("{indent}fn {}() {{", names[anchor]);
        let inserted = format!("{indent}fn {new_name}() {{}}");
        let Some(anchor_at) = updated.find(&anchor_open) else {
            panic!("trial {trial}: lost anchor {}", names[anchor]);
        };
        let Some(inserted_at) = updated.find(&inserted) else {
            panic!("trial {trial}: missing inserted function");
        };
        assert!(
            inserted_at > anchor_at,
            "trial {trial}: insert must follow anchor"
        );
        let after_anchor = &updated[anchor_at..];
        let next_fn = after_anchor.find(&inserted);
        assert_eq!(
            next_fn.map(|_| true),
            Some(true),
            "trial {trial}: inserted block should appear after the anchor snippet"
        );
        let eof = updated.rfind(&inserted).unwrap_or(0);
        let tail_after_insert = &updated[eof + inserted.len()..];
        if anchor + 1 < names.len() {
            assert!(
                tail_after_insert.contains(&format!("{indent}fn {}() {{", names[anchor + 1])),
                "trial {trial}: later functions must remain after the insert, not be skipped to EOF. got:\n{updated}"
            );
        }
        let last_original = format!("{indent}fn {}() {{", names[count - 1]);
        if anchor + 1 < count {
            assert!(
                updated.find(&inserted).expect("inserted")
                    < updated.find(&last_original).expect("last"),
                "trial {trial}: insert landed at EOF instead of beside the anchor"
            );
        }
    }
}

#[tokio::test]
async fn fuzz_large_independent_body_edits() {
    let mut rng = Lcg(0xA11E_D175);
    for trial in 0..24 {
        let indent = indent_unit(rng.usize(3));
        let count = 6 + rng.usize(10);
        let names: Vec<String> = (0..count).map(|i| format!("item_{trial}_{i}")).collect();
        let bodies: Vec<u32> = (0..count)
            .map(|i| u32::try_from(trial * 100 + i).unwrap_or(0))
            .collect();
        let original = join_functions(indent, &names, &bodies);
        let dir = tempdir().expect("tempdir");
        crate::agent::tools::common::set_tool_workspace_root(Some(dir.path().to_path_buf()));
        let path = dir.path().join("mod.rs");
        fs::write(&path, &original).expect("write");

        let mut hunks = String::new();
        let mut expected_bodies = bodies.clone();
        for (idx, name) in names.iter().enumerate() {
            if rng.usize(3) == 0 {
                continue;
            }
            let new_body = expected_bodies[idx].saturating_add(100);
            expected_bodies[idx] = new_body;
            hunks.push_str(&format!(
                "@@\n-{indent}{indent}let x = {};\n+{indent}{indent}let x = {new_body};\n",
                bodies[idx]
            ));
            let _ = name;
        }
        if hunks.is_empty() {
            continue;
        }
        let patch = format!(
            "*** Begin Patch\n*** Update File: {}\n{hunks}*** End Patch\n",
            path.display()
        );
        apply_hunks_to_cwd(&patch, None)
            .await
            .unwrap_or_else(|err| panic!("trial {trial} large edit failed: {err}\n{patch}"));
        let updated = fs::read_to_string(&path).expect("read");
        for (name, body) in names.iter().zip(expected_bodies.iter()) {
            assert!(
                updated.contains(&format!("{indent}fn {name}() {{")),
                "trial {trial}: lost {name}"
            );
            assert!(
                updated.contains(&format!("{indent}{indent}let x = {body};")),
                "trial {trial}: {name} body not updated to {body}. got:\n{updated}"
            );
        }
    }
}

#[tokio::test]
async fn fuzz_underindented_replacement_adopts_file_style() {
    let mut rng = Lcg(0x51E_571E);
    for trial in 0..24 {
        let indent = indent_unit(1); // 4 spaces — patch will use none
        let original = format!("{indent}fn keep() {{}}\n{indent}fn target() {{}}\n");
        let dir = tempdir().expect("tempdir");
        crate::agent::tools::common::set_tool_workspace_root(Some(dir.path().to_path_buf()));
        let path = dir.path().join(format!("t{trial}.rs"));
        fs::write(&path, &original).expect("write");
        let _ = rng.next();
        let patch = format!(
            "*** Begin Patch\n*** Update File: {}\n@@\n-fn target() {{}}\n+fn target() {{}}\n+fn extra() {{}}\n*** End Patch\n",
            path.display()
        );
        apply_hunks_to_cwd(&patch, None)
            .await
            .unwrap_or_else(|err| panic!("trial {trial}: {err}"));
        let updated = fs::read_to_string(&path).expect("read");
        assert!(
            updated.contains(&format!("{indent}fn extra() {{}}")),
            "trial {trial}: extra() should match file indent. got:\n{updated}"
        );
        assert!(
            !updated.contains("\nfn extra()"),
            "trial {trial}: extra() must not be flush-left. got:\n{updated}"
        );
    }
}

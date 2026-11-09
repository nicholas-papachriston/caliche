use crate::workspace::config::workspace_memory_markdown_dir;
use anyhow::Context;
use std::path::{Path, PathBuf};

pub async fn ensure_workspace_memory_dir(root_path: &Path) -> anyhow::Result<PathBuf> {
    let dir = workspace_memory_markdown_dir(root_path);
    tokio::fs::create_dir_all(&dir).await?;
    Ok(dir)
}

pub async fn write_workspace_memory_markdown(
    root_path: &Path,
    title: &str,
    body: &str,
) -> anyhow::Result<PathBuf> {
    let dir = ensure_workspace_memory_dir(root_path).await?;
    let slug = sanitize_slug(title);
    let ts = chrono::Utc::now().format("%Y%m%d%H%M%S").to_string();
    let file_name = format!("{}-{}.md", ts, slug);
    let path = dir.join(file_name);
    let tmp_path = dir.join(format!(".{}.tmp", ts));

    let content = format!("# {}\n\n{}\n", title.trim(), body.trim());
    tokio::fs::write(&tmp_path, content)
        .await
        .with_context(|| {
            format!(
                "Failed to write temporary memory file {}",
                tmp_path.display()
            )
        })?;
    tokio::fs::rename(&tmp_path, &path)
        .await
        .with_context(|| format!("Failed to finalize memory file {}", path.display()))?;
    Ok(path)
}

pub async fn list_workspace_memory_markdown_files(
    root_path: &Path,
) -> anyhow::Result<Vec<PathBuf>> {
    let dir = workspace_memory_markdown_dir(root_path);
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut out = Vec::new();
    let mut entries = tokio::fs::read_dir(dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        let is_md = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("md"));
        if is_md && path.is_file() {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

fn sanitize_slug(input: &str) -> String {
    let mut out = String::new();
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if (ch == ' ' || ch == '-' || ch == '_') && !out.ends_with('-') {
            out.push('-');
        }
    }
    let out = out.trim_matches('-').to_string();
    if out.is_empty() {
        "memory".to_string()
    } else {
        out
    }
}

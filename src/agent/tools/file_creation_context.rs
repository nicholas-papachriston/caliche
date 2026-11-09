//! Optional context for tracking file creation (e.g. CLI session).
//! Used to warn when creating many new documentation files.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

static CTX: std::sync::OnceLock<Mutex<FileCreationContext>> = std::sync::OnceLock::new();

#[derive(Default)]
struct FileCreationContext {
    files_created: Vec<PathBuf>,
    max_new_files: usize,
}

/// Set context from CLI (call before agent turn). Use `max_new_files` from config, default 10.
pub fn set_context(files_created: Vec<PathBuf>, max_new_files: usize) {
    let ctx = CTX.get_or_init(|| Mutex::new(FileCreationContext::default()));
    if let Ok(mut g) = ctx.lock() {
        g.files_created = files_created;
        g.max_new_files = max_new_files;
    }
}

/// Clear context (call when ending session).
pub fn clear_context() {
    if let Some(c) = CTX.get()
        && let Ok(mut g) = c.lock()
    {
        g.files_created.clear();
        g.max_new_files = 0;
    }
}

/// Record that a new file was created. Returns current count.
pub fn record_created(path: PathBuf) -> usize {
    if let Some(c) = CTX.get()
        && let Ok(mut g) = c.lock()
    {
        g.files_created.push(path);
        return g.files_created.len();
    }
    0
}

/// Check whether we may create a new doc file. Returns warning message if over limit.
pub fn check_may_create(path: &Path, is_new: bool) -> Option<String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let is_doc = matches!(ext, "md" | "txt" | "rst" | "adoc");
    if !is_new || !is_doc {
        return None;
    }
    if let Some(c) = CTX.get()
        && let Ok(g) = c.lock()
        && g.max_new_files > 0
        && g.files_created.len() >= g.max_new_files
    {
        return Some(format!(
            "Already created {} files this session (limit {}). \
                     Consider updating existing files instead of creating new ones. \
                     Created: {:?}",
            g.files_created.len(),
            g.max_new_files,
            g.files_created
        ));
    }
    None
}

/// Take current list of created paths (e.g. to sync back to CLI session).
pub fn take_files_created() -> Vec<PathBuf> {
    if let Some(c) = CTX.get()
        && let Ok(mut g) = c.lock()
    {
        return std::mem::take(&mut g.files_created);
    }
    Vec::new()
}

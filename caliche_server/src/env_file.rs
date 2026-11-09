//! Loading an explicit `--env-file` into the process environment.
//!
//! `caliche` and `caliche_server` share this rule. Only the file named by
//! `--env-file` is read. A `.env` in the working directory is never read: it
//! may come from a cloned repo and could pick the provider, the model or a
//! `*_BASE_URL`, and so send the user's keys to another host.

/// Load the file at `path`, if one is given, into the process environment.
///
/// A name that is already set, for example by the shell, keeps its value.
///
/// This writes the process environment. Call it from synchronous `main`
/// before the async runtime is built, while no other thread exists. It runs
/// after the command line is parsed, so the file does not supply values for
/// flags that read the environment (`PORT`, `DEFAULT_MODEL` and so on).
///
/// # Errors
///
/// Returns an error when the file cannot be read or parsed.
pub fn load_before_runtime(path: Option<&str>) -> anyhow::Result<()> {
    if let Some(path) = path {
        dotenvy::from_filename(path)
            .map_err(|e| anyhow::anyhow!("Failed to load --env-file '{path}': {e}"))?;
    }
    Ok(())
}

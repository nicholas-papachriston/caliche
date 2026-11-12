//! Install and update skills from a git checkout.
//!
//! The lock file stores a git commit and a content hash. An update is due when
//! either value changes. Network fetch goes through the egress policy. An
//! optional source allowlist limits which repositories a command may clone.

use std::fs;
use std::path::{Path, PathBuf};

use std::collections::HashSet;

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

mod remote_files;
use remote_files::{
    TempCheckout, clone_repo, collect_hash_files, git_head, replace_tree, skill_tree_diff,
};

#[cfg(test)]
use std::process::Command;

use crate::gateway::egress::{EgressClass, current_policy};

const LOCK_FILE: &str = "skills-lock.json";
const SKILL_FILE: &str = "SKILL.md";

/// Parsed skill source from a CLI argument.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteSkillSource {
    /// `owner/repo` key used by the allowlist.
    pub owner_repo: String,
    /// Clone URL or local checkout path.
    pub locator: String,
    /// Path inside the repository that contains `SKILL.md`.
    pub skill_path: Option<String>,
    /// True when `locator` is an https URL that must pass egress policy.
    pub remote: bool,
}

/// One installed skill in `skills-lock.json`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkillLockEntry {
    pub source: String,
    pub source_type: String,
    pub skill_path: String,
    pub computed_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_sha: Option<String>,
}

/// Lock file written beside installed skill directories.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SkillLock {
    pub version: u32,
    pub skills: std::collections::BTreeMap<String, SkillLockEntry>,
}

/// Result of copying one skill into the install root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallReport {
    pub name: String,
    pub computed_hash: String,
    pub git_sha: Option<String>,
}

/// A fetched skill compared with the installed copy. Nothing is written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillUpdatePreview {
    pub name: String,
    pub source: String,
    pub changed: bool,
    pub locked_hash: String,
    pub remote_hash: String,
    pub locked_git_sha: Option<String>,
    pub remote_git_sha: Option<String>,
    pub diff: String,
}

/// Result of accepting one skill update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkillUpdateAction {
    Updated(InstallReport),
    Current { name: String },
}

/// Why an installed skill differs from its lock entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkillFreshness {
    Current,
    Missing,
    ContentChanged { locked: String, actual: String },
    GitChanged { locked: String, actual: String },
}

impl Default for SkillLock {
    fn default() -> Self {
        Self {
            version: 1,
            skills: std::collections::BTreeMap::new(),
        }
    }
}

/// Parse `owner/repo`, a GitHub tree URL, or a local directory.
pub fn parse_remote_source(input: &str) -> Result<RemoteSkillSource> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        bail!("skill source is empty");
    }
    if trimmed.contains("://") {
        return parse_url_source(trimmed);
    }
    if looks_like_local_path(trimmed) {
        return Ok(RemoteSkillSource {
            owner_repo: trimmed.to_string(),
            locator: trimmed.to_string(),
            skill_path: None,
            remote: false,
        });
    }
    parse_owner_repo(trimmed)
}

/// True when `owner/repo` matches the allowlist.
///
/// `None` means the operator did not set a list, so the explicit source is
/// allowed. An empty list denies every remote source.
pub fn source_allowed(owner_repo: &str, allowlist: Option<&[String]>) -> bool {
    let Some(allowlist) = allowlist else {
        return true;
    };
    if allowlist.is_empty() {
        return false;
    }
    allowlist
        .iter()
        .any(|entry| source_entry_matches(owner_repo, entry))
}

/// Read the skill-source allowlist from `CALICHE_SKILL_SOURCES`.
#[must_use]
pub fn skill_source_allowlist() -> Option<Vec<String>> {
    let Ok(value) = std::env::var("CALICHE_SKILL_SOURCES") else {
        return None;
    };
    Some(
        value
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(str::to_string)
            .collect(),
    )
}

/// SHA-256 over the skill directory's relative paths and file bytes.
pub fn content_hash(dir: &Path) -> Result<String> {
    let mut files = Vec::new();
    collect_hash_files(dir, dir, &mut files)?;
    files.sort();
    let mut hasher = Sha256::new();
    for relative in &files {
        hasher.update(relative.as_bytes());
        hasher.update([0]);
        let bytes = fs::read(dir.join(relative)).with_context(|| format!("read {}", relative))?;
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Compare an installed directory with its lock entry and optional remote git SHA.
pub fn skill_freshness(
    entry: &SkillLockEntry,
    dir: &Path,
    remote_git_sha: Option<&str>,
) -> Result<SkillFreshness> {
    if !dir.join(SKILL_FILE).is_file() {
        return Ok(SkillFreshness::Missing);
    }
    let actual = content_hash(dir)?;
    if actual != entry.computed_hash {
        return Ok(SkillFreshness::ContentChanged {
            locked: entry.computed_hash.clone(),
            actual,
        });
    }
    if let (Some(locked), Some(actual_sha)) = (entry.git_sha.as_deref(), remote_git_sha)
        && locked != actual_sha
    {
        return Ok(SkillFreshness::GitChanged {
            locked: locked.to_string(),
            actual: actual_sha.to_string(),
        });
    }
    Ok(SkillFreshness::Current)
}

/// Copy skills from an already-fetched checkout into `dest_root`.
pub fn install_checkout(
    checkout: &Path,
    source: &RemoteSkillSource,
    dest_root: &Path,
    git_sha: Option<&str>,
) -> Result<Vec<InstallReport>> {
    let dirs = find_skill_dirs(checkout, source.skill_path.as_deref())?;
    let mut lock = read_lock(&lock_path(dest_root))?;
    let mut reports = Vec::with_capacity(dirs.len());
    for dir in dirs {
        let name = skill_name(&dir)?;
        let hash = content_hash(&dir)?;
        let dest = dest_root.join(&name);
        replace_tree(&dir, &dest)?;
        lock.skills.insert(
            name.clone(),
            SkillLockEntry {
                source: source.owner_repo.clone(),
                source_type: if source.remote { "github" } else { "local" }.to_string(),
                skill_path: source
                    .skill_path
                    .clone()
                    .unwrap_or_else(|| relative_skill_path(checkout, &dir)),
                computed_hash: hash.clone(),
                git_sha: git_sha.map(str::to_string),
            },
        );
        reports.push(InstallReport {
            name,
            computed_hash: hash,
            git_sha: git_sha.map(str::to_string),
        });
    }
    write_lock(&lock_path(dest_root), &lock)?;
    Ok(reports)
}

/// Clone or reuse a local checkout, then install.
///
/// Remote URLs must pass the current egress policy before git runs.
pub fn install_source(source: &RemoteSkillSource, dest_root: &Path) -> Result<Vec<InstallReport>> {
    if !source_allowed(&source.owner_repo, skill_source_allowlist().as_deref()) {
        bail!(
            "skill source {} is not in CALICHE_SKILL_SOURCES",
            source.owner_repo
        );
    }
    if source.remote {
        authorize_remote(&source.locator)?;
        let checkout = TempCheckout::create()?;
        let sha = clone_repo(&source.locator, &checkout.0)?;
        return install_checkout(&checkout.0, source, dest_root, Some(&sha));
    }
    let path = PathBuf::from(&source.locator);
    let sha = git_head(&path).ok();
    install_checkout(&path, source, dest_root, sha.as_deref())
}

/// Fetch the named skills and return a diff. This function does not write files.
///
/// An empty `names` list reviews every installed skill. Each name must exist.
pub fn preview_updates(dest_root: &Path, names: &[String]) -> Result<Vec<SkillUpdatePreview>> {
    let lock = read_lock(&lock_path(dest_root))?;
    let selected = selected_names(&lock, names)?;
    let mut previews = Vec::with_capacity(selected.len());
    for name in selected {
        let entry = lock
            .skills
            .get(&name)
            .context("selected skill left the lock")?;
        previews.push(preview_one(dest_root, &name, entry)?);
    }
    Ok(previews)
}

/// Fetch the named skills and write the ones that differ.
///
/// An empty `names` list accepts every change. Each name must exist.
pub fn apply_updates(dest_root: &Path, names: &[String]) -> Result<Vec<SkillUpdateAction>> {
    let mut lock = read_lock(&lock_path(dest_root))?;
    let selected = selected_names(&lock, names)?;
    let mut actions = Vec::with_capacity(selected.len());
    for name in selected {
        let entry = lock
            .skills
            .get(&name)
            .cloned()
            .context("selected skill left the lock")?;
        let action = apply_one(dest_root, &name, &entry)?;
        if let SkillUpdateAction::Updated(report) = &action
            && let Some(slot) = lock.skills.get_mut(&name)
        {
            slot.computed_hash.clone_from(&report.computed_hash);
            slot.git_sha.clone_from(&report.git_sha);
            write_lock(&lock_path(dest_root), &lock)?;
        }
        actions.push(action);
    }
    Ok(actions)
}

fn selected_names(lock: &SkillLock, names: &[String]) -> Result<Vec<String>> {
    if names.is_empty() {
        return Ok(lock.skills.keys().cloned().collect());
    }
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for name in names {
        if !seen.insert(name.clone()) {
            continue;
        }
        if !lock.skills.contains_key(name) {
            bail!("skill {name} is not installed");
        }
        out.push(name.clone());
    }
    Ok(out)
}

fn preview_one(dest_root: &Path, name: &str, entry: &SkillLockEntry) -> Result<SkillUpdatePreview> {
    let fetched = fetch_locked_skill(entry)?;
    let remote_hash = content_hash(&fetched.skill_dir)?;
    let installed = dest_root.join(name);
    let diff = skill_tree_diff(&installed, &fetched.skill_dir)?;
    let changed = skill_changed(entry, &remote_hash, fetched.git_sha.as_deref(), &diff);
    Ok(SkillUpdatePreview {
        name: name.to_string(),
        source: entry.source.clone(),
        changed,
        locked_hash: entry.computed_hash.clone(),
        remote_hash,
        locked_git_sha: entry.git_sha.clone(),
        remote_git_sha: fetched.git_sha,
        diff,
    })
}

fn apply_one(dest_root: &Path, name: &str, entry: &SkillLockEntry) -> Result<SkillUpdateAction> {
    let fetched = fetch_locked_skill(entry)?;
    let remote_hash = content_hash(&fetched.skill_dir)?;
    let installed = dest_root.join(name);
    let diff = skill_tree_diff(&installed, &fetched.skill_dir)?;
    if !skill_changed(entry, &remote_hash, fetched.git_sha.as_deref(), &diff) {
        return Ok(SkillUpdateAction::Current {
            name: name.to_string(),
        });
    }
    replace_tree(&fetched.skill_dir, &installed)?;
    Ok(SkillUpdateAction::Updated(InstallReport {
        name: name.to_string(),
        computed_hash: remote_hash,
        git_sha: fetched.git_sha,
    }))
}

fn skill_changed(
    entry: &SkillLockEntry,
    remote_hash: &str,
    remote_sha: Option<&str>,
    diff: &str,
) -> bool {
    if !diff.is_empty() || entry.computed_hash != remote_hash {
        return true;
    }
    match (entry.git_sha.as_deref(), remote_sha) {
        (Some(locked), Some(remote)) => locked != remote,
        _ => false,
    }
}

struct FetchedSkill {
    _hold: Option<TempCheckout>,
    skill_dir: PathBuf,
    git_sha: Option<String>,
}

fn fetch_locked_skill(entry: &SkillLockEntry) -> Result<FetchedSkill> {
    if entry.source_type == "github" {
        return fetch_remote_skill(entry);
    }
    fetch_local_skill(entry)
}

fn fetch_local_skill(entry: &SkillLockEntry) -> Result<FetchedSkill> {
    let root = PathBuf::from(&entry.source);
    if !root.is_dir() {
        bail!("local skill source {} is missing", entry.source);
    }
    let skill_dir = skill_dir_in(&root, &entry.skill_path)?;
    Ok(FetchedSkill {
        _hold: None,
        skill_dir,
        git_sha: git_head(&root).ok(),
    })
}

fn fetch_remote_skill(entry: &SkillLockEntry) -> Result<FetchedSkill> {
    let source = source_from_lock(entry)?;
    if !source_allowed(&source.owner_repo, skill_source_allowlist().as_deref()) {
        bail!(
            "skill source {} is not in CALICHE_SKILL_SOURCES",
            source.owner_repo
        );
    }
    authorize_remote(&source.locator)?;
    let checkout = TempCheckout::create()?;
    let sha = clone_repo(&source.locator, &checkout.0)?;
    let skill_dir = skill_dir_in(&checkout.0, &entry.skill_path)?;
    Ok(FetchedSkill {
        _hold: Some(checkout),
        skill_dir,
        git_sha: Some(sha),
    })
}

fn skill_dir_in(root: &Path, skill_path: &str) -> Result<PathBuf> {
    let dir = if skill_path.is_empty() || skill_path == "." {
        root.to_path_buf()
    } else {
        root.join(skill_path)
    };
    if !dir.join(SKILL_FILE).is_file() {
        bail!("no {SKILL_FILE} at {}", dir.display());
    }
    Ok(dir)
}

fn source_from_lock(entry: &SkillLockEntry) -> Result<RemoteSkillSource> {
    let mut source = parse_owner_repo(&entry.source)?;
    if !entry.skill_path.is_empty() && entry.skill_path != "." {
        source.skill_path = Some(entry.skill_path.clone());
    }
    Ok(source)
}

fn authorize_remote(url: &str) -> Result<()> {
    current_policy()
        .context("load egress policy")?
        .validate_static(EgressClass::Plugin, url)
        .with_context(|| format!("egress policy denied {url}"))?;
    Ok(())
}

fn parse_url_source(input: &str) -> Result<RemoteSkillSource> {
    if !input.starts_with("https://") {
        bail!("skill URLs must use https");
    }
    if input.contains('@') {
        bail!("skill URLs must not embed credentials");
    }
    let rest = input.trim_start_matches("https://");
    let mut parts = rest.split('/');
    let host = parts.next().unwrap_or_default();
    let owner = parts.next().unwrap_or_default();
    let repo_raw = parts.next().unwrap_or_default();
    if host.is_empty() || owner.is_empty() || repo_raw.is_empty() {
        bail!("skill URL needs a host, owner, and repository");
    }
    let repo = repo_raw.trim_end_matches(".git");
    let skill_path = skill_path_from_url_parts(parts);
    reject_parent_path(skill_path.as_deref())?;
    Ok(RemoteSkillSource {
        owner_repo: format!("{owner}/{repo}"),
        locator: format!("https://{host}/{owner}/{repo}.git"),
        skill_path,
        remote: true,
    })
}

fn skill_path_from_url_parts<'a>(parts: impl Iterator<Item = &'a str>) -> Option<String> {
    let mut rest: Vec<&str> = parts.filter(|part| !part.is_empty()).collect();
    if rest.first().copied() == Some("tree") {
        rest.drain(..2.min(rest.len()));
    }
    if rest.is_empty() {
        None
    } else {
        Some(rest.join("/"))
    }
}

fn parse_owner_repo(input: &str) -> Result<RemoteSkillSource> {
    let parts: Vec<&str> = input.split('/').filter(|part| !part.is_empty()).collect();
    if parts.len() < 2 {
        bail!("skill source must look like owner/repo");
    }
    let owner = parts[0];
    let repo = parts[1].trim_end_matches(".git");
    if !is_repo_token(owner) || !is_repo_token(repo) {
        bail!("skill source has an invalid owner or repository");
    }
    let skill_path = if parts.len() > 2 {
        Some(parts[2..].join("/"))
    } else {
        None
    };
    reject_parent_path(skill_path.as_deref())?;
    Ok(RemoteSkillSource {
        owner_repo: format!("{owner}/{repo}"),
        locator: format!("https://github.com/{owner}/{repo}.git"),
        skill_path,
        remote: true,
    })
}

fn looks_like_local_path(input: &str) -> bool {
    input.starts_with('.') || input.starts_with('/') || input.starts_with('~')
}

fn is_repo_token(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.')
}

fn reject_parent_path(path: Option<&str>) -> Result<()> {
    if path.is_some_and(|value| value.split('/').any(|part| part == "..")) {
        bail!("skill path must not contain ..");
    }
    Ok(())
}

fn source_entry_matches(owner_repo: &str, entry: &str) -> bool {
    let entry = entry.trim();
    entry == "*"
        || entry == owner_repo
        || entry
            .strip_suffix("/*")
            .is_some_and(|owner| owner_repo.starts_with(&format!("{owner}/")))
}

fn find_skill_dirs(root: &Path, skill_path: Option<&str>) -> Result<Vec<PathBuf>> {
    if let Some(path) = skill_path {
        let dir = root.join(path);
        if is_skill_dir(root, &dir) {
            return Ok(vec![dir]);
        }
        bail!("no {SKILL_FILE} at {}", dir.display());
    }
    let mut out = Vec::new();
    push_if_skill(root, root, &mut out);
    for relative in [
        "skills",
        ".agents/skills",
        ".claude/skills",
        ".codex/skills",
    ] {
        let base = root.join(relative);
        let Ok(entries) = fs::read_dir(&base) else {
            continue;
        };
        for entry in entries.flatten() {
            push_if_skill(root, &entry.path(), &mut out);
        }
    }
    if out.is_empty() {
        bail!("no {SKILL_FILE} files in {}", root.display());
    }
    Ok(out)
}

fn push_if_skill(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) {
    if is_skill_dir(root, dir) {
        out.push(dir.to_path_buf());
    }
}

/// Whether `dir`, inside the checkout `root`, holds a `SKILL.md` that is a
/// regular file reached from `root` without a link: a link in the checkout
/// would otherwise install a directory from outside it as a skill.
fn is_skill_dir(root: &Path, dir: &Path) -> bool {
    let Ok(below) = dir.strip_prefix(root) else {
        return false;
    };
    if below
        .components()
        .any(|part| !matches!(part, std::path::Component::Normal(_)))
    {
        return false;
    }
    let Ok(root) = root.canonicalize() else {
        return false;
    };
    caliche_sandbox::open_file_beneath(&root, &root.join(below).join(SKILL_FILE)).is_ok()
}

fn skill_name(dir: &Path) -> Result<String> {
    let text = fs::read_to_string(dir.join(SKILL_FILE))
        .with_context(|| format!("read {}", dir.join(SKILL_FILE).display()))?;
    if let Some(name) = front_matter_name(&text) {
        return Ok(name);
    }
    dir.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .context("skill directory has no name")
}

fn front_matter_name(text: &str) -> Option<String> {
    let rest = text.strip_prefix("---\n")?;
    let end = rest.find("\n---")?;
    for line in rest[..end].lines() {
        let Some(value) = line.strip_prefix("name:") else {
            continue;
        };
        let name = value.trim().trim_matches('"').trim_matches('\'');
        if !name.is_empty() && !name.contains('/') && !name.contains("..") {
            return Some(name.to_string());
        }
    }
    None
}

fn relative_skill_path(root: &Path, dir: &Path) -> String {
    dir.strip_prefix(root)
        .ok()
        .map(|path| path.to_string_lossy().replace('\\', "/"))
        .filter(|path| !path.is_empty())
        .unwrap_or_else(|| ".".to_string())
}

fn lock_path(dest_root: &Path) -> PathBuf {
    dest_root.join(LOCK_FILE)
}

fn read_lock(path: &Path) -> Result<SkillLock> {
    if !path.is_file() {
        return Ok(SkillLock::default());
    }
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

fn write_lock(path: &Path, lock: &SkillLock) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let text = serde_json::to_string_pretty(lock).context("encode skill lock")?;
    fs::write(path, text).with_context(|| format!("write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_owner_repo_and_rejects_unsafe_sources() {
        let source = parse_remote_source("anthropics/skills/skills/review").expect("parse");
        assert_eq!(source.owner_repo, "anthropics/skills");
        assert_eq!(source.locator, "https://github.com/anthropics/skills.git");
        assert_eq!(source.skill_path.as_deref(), Some("skills/review"));
        assert!(source.remote);

        let url =
            parse_remote_source("https://github.com/anthropics/skills/tree/main/skills/review")
                .expect("url");
        assert_eq!(url.skill_path.as_deref(), Some("skills/review"));
        assert!(parse_remote_source("http://github.com/a/b").is_err());
        assert!(parse_remote_source("https://user:token@github.com/a/b").is_err());
        assert!(parse_remote_source("owner/repo/../secret").is_err());
    }

    #[test]
    fn allowlist_matches_owner_and_wildcard() {
        assert!(source_allowed("a/b", None));
        assert!(!source_allowed("a/b", Some(&[])));
        let list = vec!["acme/*".to_string(), "other/repo".to_string()];
        assert!(source_allowed("acme/skills", Some(&list)));
        assert!(source_allowed("other/repo", Some(&list)));
        assert!(!source_allowed("acme-evil/skills", Some(&list)));
    }

    #[test]
    fn install_records_hash_and_detects_a_change() {
        let repo = tempfile::tempdir().expect("repo");
        let skill = repo.path().join("skills").join("demo");
        fs::create_dir_all(&skill).expect("skill dir");
        fs::write(
            skill.join(SKILL_FILE),
            "---\nname: demo\n---\nDo the documented task.\n",
        )
        .expect("skill file");
        git_commit(repo.path(), "add skill");
        let sha = git_head(repo.path()).expect("sha");

        let dest = tempfile::tempdir().expect("dest");
        let source = RemoteSkillSource {
            owner_repo: repo.path().display().to_string(),
            locator: repo.path().display().to_string(),
            skill_path: None,
            remote: false,
        };
        let reports =
            install_checkout(repo.path(), &source, dest.path(), Some(&sha)).expect("install");
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].name, "demo");
        let installed = dest.path().join("demo");
        let lock = read_lock(&lock_path(dest.path())).expect("lock");
        let entry = lock.skills.get("demo").expect("entry");
        assert_eq!(
            skill_freshness(entry, &installed, Some(&sha)).expect("fresh"),
            SkillFreshness::Current
        );

        fs::write(
            installed.join(SKILL_FILE),
            "---\nname: demo\n---\nChanged.\n",
        )
        .expect("edit");
        assert!(matches!(
            skill_freshness(entry, &installed, Some(&sha)).expect("stale"),
            SkillFreshness::ContentChanged { .. }
        ));
        assert!(matches!(
            skill_freshness(entry, &installed, Some("abc1234")).expect("git"),
            SkillFreshness::ContentChanged { .. }
        ));

        fs::write(
            installed.join(SKILL_FILE),
            "---\nname: demo\n---\nDo the documented task.\n",
        )
        .expect("restore");
        assert!(matches!(
            skill_freshness(entry, &installed, Some("abc1234")).expect("git only"),
            SkillFreshness::GitChanged { .. }
        ));
    }

    #[test]
    fn preview_keeps_files_and_apply_updates_only_named_skills() {
        let repo = tempfile::tempdir().expect("repo");
        for (name, body) in [("demo", "Original demo.\n"), ("other", "Keep this.\n")] {
            let dir = repo.path().join("skills").join(name);
            fs::create_dir_all(&dir).expect("skill dir");
            fs::write(
                dir.join(SKILL_FILE),
                format!("---\nname: {name}\n---\n{body}"),
            )
            .expect("skill file");
        }
        git_commit(repo.path(), "add skills");
        let sha = git_head(repo.path()).expect("sha");
        let dest = tempfile::tempdir().expect("dest");
        let source = RemoteSkillSource {
            owner_repo: repo.path().display().to_string(),
            locator: repo.path().display().to_string(),
            skill_path: None,
            remote: false,
        };
        install_checkout(repo.path(), &source, dest.path(), Some(&sha)).expect("install");
        fs::write(
            repo.path().join("skills").join("demo").join(SKILL_FILE),
            "---\nname: demo\n---\nUpdated demo.\n",
        )
        .expect("edit source");
        git_commit(repo.path(), "edit demo");

        let installed_demo = fs::read(dest.path().join("demo").join(SKILL_FILE)).expect("demo");
        let installed_other = fs::read(dest.path().join("other").join(SKILL_FILE)).expect("other");
        let previews = preview_updates(dest.path(), &["demo".to_string()]).expect("preview");
        assert_eq!(previews.len(), 1);
        assert!(previews[0].changed);
        assert!(previews[0].diff.contains("Updated demo."));
        assert_eq!(
            fs::read(dest.path().join("demo").join(SKILL_FILE)).expect("unchanged"),
            installed_demo
        );
        assert!(preview_updates(dest.path(), &["missing".to_string()]).is_err());

        let actions = apply_updates(dest.path(), &["demo".to_string()]).expect("apply");
        assert!(matches!(
            actions.as_slice(),
            [SkillUpdateAction::Updated(_)]
        ));
        assert_ne!(
            fs::read(dest.path().join("demo").join(SKILL_FILE)).expect("new demo"),
            installed_demo
        );
        assert_eq!(
            fs::read(dest.path().join("other").join(SKILL_FILE)).expect("other stays"),
            installed_other
        );
        let again = apply_updates(dest.path(), &["demo".to_string()]).expect("second apply");
        assert!(matches!(
            again.as_slice(),
            [SkillUpdateAction::Current { .. }]
        ));
    }

    fn git_commit(repo: &Path, message: &str) {
        let init = Command::new("git")
            .args(["init"])
            .arg(repo)
            .status()
            .expect("git init");
        assert!(init.success());
        let add = Command::new("git")
            .args(["-C"])
            .arg(repo)
            .args(["add", "."])
            .status()
            .expect("git add");
        assert!(add.success());
        let commit = Command::new("git")
            .args(["-c", "user.email=test@example.com", "-c", "user.name=test"])
            .args(["-C"])
            .arg(repo)
            .args(["commit", "-m", message])
            .status()
            .expect("git commit");
        assert!(commit.success());
    }
}

#[cfg(all(test, unix))]
mod link_tests {
    use super::{SKILL_FILE, find_skill_dirs};

    #[test]
    fn a_skill_reached_through_a_link_in_the_checkout_is_not_found() {
        let outside = tempfile::tempdir().expect("outside dir");
        std::fs::write(outside.path().join(SKILL_FILE), "---\nname: outside\n---\n")
            .expect("outside skill");
        let checkout = tempfile::tempdir().expect("checkout");
        let skills = checkout.path().join("skills");
        std::fs::create_dir_all(skills.join("real")).expect("real dir");
        std::fs::write(skills.join("real").join(SKILL_FILE), "real").expect("real skill");
        std::os::unix::fs::symlink(outside.path(), skills.join("dir_link")).expect("dir link");
        std::fs::create_dir(skills.join("file_link")).expect("file link dir");
        std::os::unix::fs::symlink(
            outside.path().join(SKILL_FILE),
            skills.join("file_link").join(SKILL_FILE),
        )
        .expect("file link");

        let found = find_skill_dirs(checkout.path(), None).expect("one real skill");
        let named = find_skill_dirs(checkout.path(), Some("skills/dir_link"));

        assert_eq!(found, [skills.join("real")]);
        assert_eq!(
            named.expect_err("a linked skill dir").to_string(),
            format!("no {SKILL_FILE} at {}", skills.join("dir_link").display())
        );
    }
}

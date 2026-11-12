//! Context manager for workspace and file tracking.
//!
//! Tracks workspace state, project type, file access patterns, and git context
//! to provide relevant information to the agent.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Project type detection
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectType {
    /// Rust project (Cargo.toml)
    Rust,
    /// JavaScript/TypeScript project (package.json)
    JavaScript,
    /// Python project (pyproject.toml, setup.py, requirements.txt)
    Python,
    /// Go project (go.mod)
    Go,
    /// Java/Kotlin project (pom.xml, build.gradle)
    Java,
    /// Generic project
    Generic,
}

/// Git repository context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitContext {
    /// Current branch name
    pub branch: String,
    /// Whether there are uncommitted changes
    pub has_changes: bool,
    /// Number of staged files
    pub staged_count: usize,
    /// Number of modified files
    pub modified_count: usize,
    /// Remote URL if available
    pub remote_url: Option<String>,
}

/// Context for a tracked file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileContext {
    /// File path
    pub path: PathBuf,
    /// File size in bytes
    pub size_bytes: u64,
    /// File extension
    pub extension: Option<String>,
    /// Last read timestamp
    pub last_read: Option<chrono::DateTime<chrono::Utc>>,
    /// Number of times read
    pub read_count: usize,
    /// Whether the file was modified
    pub modified: bool,
    /// Language detected from extension
    pub language: Option<String>,
}

impl FileContext {
    /// Create a new file context
    pub fn new(path: PathBuf) -> Self {
        let extension = path.extension().map(|e| e.to_string_lossy().to_string());
        let language = extension
            .as_ref()
            .and_then(|ext| detect_language(ext))
            .map(String::from);

        Self {
            path,
            size_bytes: 0,
            extension,
            last_read: None,
            read_count: 0,
            modified: false,
            language,
        }
    }

    /// Update metadata from file system
    pub async fn refresh(&mut self) -> Result<()> {
        if let Ok(metadata) = tokio::fs::metadata(&self.path).await {
            self.size_bytes = metadata.len();
        }
        Ok(())
    }

    /// Mark as read
    pub fn mark_read(&mut self) {
        self.last_read = Some(chrono::Utc::now());
        self.read_count += 1;
    }

    /// Mark as modified
    pub fn mark_modified(&mut self) {
        self.modified = true;
    }
}

/// An item of context that can be provided to the agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextItem {
    /// Type of context
    pub item_type: ContextItemType,
    /// Content or summary
    pub content: String,
    /// Relevance score (0.0 - 1.0)
    pub relevance: f32,
    /// Source path if applicable
    pub source: Option<PathBuf>,
}

/// Types of context items
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextItemType {
    /// File content
    FileContent,
    /// File summary/structure
    FileSummary,
    /// Directory listing
    DirectoryListing,
    /// Git status
    GitStatus,
    /// Project configuration
    ProjectConfig,
    /// Recent changes
    RecentChanges,
}

/// Context Manager for workspace awareness
#[derive(Debug)]
pub struct ContextManager {
    /// Workspace root directory
    workspace_root: PathBuf,
    /// Tracked files
    active_files: HashMap<PathBuf, FileContext>,
    /// Git context if in a git repository
    git_context: Option<GitContext>,
    /// Detected project type
    project_type: Option<ProjectType>,
    /// Recently accessed files (LRU order)
    recent_files: Vec<PathBuf>,
    /// Maximum number of recent files to track
    max_recent_files: usize,
}

impl ContextManager {
    /// Create a new context manager
    pub fn new(workspace_root: PathBuf) -> Self {
        Self {
            workspace_root,
            active_files: HashMap::new(),
            git_context: None,
            project_type: None,
            recent_files: Vec::new(),
            max_recent_files: 50,
        }
    }

    /// Get the workspace root
    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }

    /// Set the workspace root
    pub fn set_workspace(&mut self, path: PathBuf) {
        self.workspace_root = path;
        self.active_files.clear();
        self.git_context = None;
        self.project_type = None;
        self.recent_files.clear();
    }

    /// Detect the project type based on configuration files
    pub fn detect_project(&mut self) -> Result<ProjectType> {
        if let Some(ref project_type) = self.project_type {
            return Ok(project_type.clone());
        }

        let project_type = self.detect_project_type_internal();
        self.project_type = Some(project_type.clone());
        Ok(project_type)
    }

    fn detect_project_type_internal(&self) -> ProjectType {
        // Check for Rust
        if self.workspace_root.join("Cargo.toml").exists() {
            return ProjectType::Rust;
        }

        // Check for JavaScript/TypeScript
        if self.workspace_root.join("package.json").exists() {
            return ProjectType::JavaScript;
        }

        // Check for Python
        if self.workspace_root.join("pyproject.toml").exists()
            || self.workspace_root.join("setup.py").exists()
            || self.workspace_root.join("requirements.txt").exists()
        {
            return ProjectType::Python;
        }

        // Check for Go
        if self.workspace_root.join("go.mod").exists() {
            return ProjectType::Go;
        }

        // Check for Java/Kotlin
        if self.workspace_root.join("pom.xml").exists()
            || self.workspace_root.join("build.gradle").exists()
            || self.workspace_root.join("build.gradle.kts").exists()
        {
            return ProjectType::Java;
        }

        ProjectType::Generic
    }

    /// Update git context
    pub async fn refresh_git_context(&mut self) -> Result<Option<GitContext>> {
        // Only the workspace's own repository goes into context. A `.git`
        // that is a link, or a workspace without its own `.git`, gives none,
        // so a link committed into a project cannot put another
        // repository's branch and remote into the prompt.
        if !matches!(
            caliche_policy::classify(&self.workspace_root),
            Ok(caliche_policy::WorkspaceGit::Own { .. })
        ) {
            self.git_context = None;
            return Ok(None);
        }

        // Try to get git status
        let context = self.get_git_context_internal().await;
        self.git_context = context.clone();
        Ok(context)
    }

    #[allow(
        clippy::unused_async,
        reason = "callers await it; libgit2 reads are synchronous"
    )]
    async fn get_git_context_internal(&self) -> Option<GitContext> {
        // libgit2 runs no command that repository config names.
        #[cfg(feature = "tools-git")]
        {
            let repo =
                crate::agent::tools::git::open_workspace_repository(&self.workspace_root).ok()?;

            // Get current branch
            let head = repo.head().ok()?;
            let branch = head
                .shorthand()
                .map(|s| s.to_string())
                .unwrap_or_else(|| "HEAD".to_string());

            // Get status
            let statuses = repo.statuses(None).ok()?;
            let mut staged_count = 0;
            let mut modified_count = 0;

            for entry in statuses.iter() {
                let status = entry.status();
                if status.is_index_new() || status.is_index_modified() || status.is_index_deleted()
                {
                    staged_count += 1;
                }
                if status.is_wt_modified() || status.is_wt_deleted() {
                    modified_count += 1;
                }
            }

            // Get remote URL
            let remote_url = repo
                .find_remote("origin")
                .ok()
                .and_then(|remote| remote.url().map(|s| s.to_string()));

            Some(GitContext {
                branch,
                has_changes: staged_count > 0 || modified_count > 0,
                staged_count,
                modified_count,
                remote_url,
            })
        }

        // Without libgit2 the agent does not run host `git`, whose repository
        // config can name commands (filters, fsmonitor) the host would run.
        #[cfg(not(feature = "tools-git"))]
        {
            None
        }
    }

    /// Track a file access
    pub async fn track_file(&mut self, path: &Path) -> Result<FileContext> {
        let abs_path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.workspace_root.join(path)
        };

        // Get or create file context
        let context = self
            .active_files
            .entry(abs_path.clone())
            .or_insert_with(|| FileContext::new(abs_path.clone()));

        // Refresh metadata
        context.refresh().await?;
        context.mark_read();

        // Update recent files
        self.recent_files.retain(|p| p != &abs_path);
        self.recent_files.insert(0, abs_path.clone());
        if self.recent_files.len() > self.max_recent_files {
            self.recent_files.truncate(self.max_recent_files);
        }

        Ok(context.clone())
    }

    /// Mark a file as modified
    pub fn mark_file_modified(&mut self, path: &Path) {
        let abs_path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.workspace_root.join(path)
        };

        if let Some(context) = self.active_files.get_mut(&abs_path) {
            context.mark_modified();
        } else {
            let mut context = FileContext::new(abs_path.clone());
            context.mark_modified();
            self.active_files.insert(abs_path, context);
        }
    }

    /// Get relevant context for a query
    pub fn get_relevant_context(&self, _query: &str) -> Vec<ContextItem> {
        let mut items = Vec::new();

        // Add project type context
        if let Some(ref project_type) = self.project_type {
            items.push(ContextItem {
                item_type: ContextItemType::ProjectConfig,
                content: format!("Project type: {:?}", project_type),
                relevance: 0.5,
                source: None,
            });
        }

        // Add git context
        if let Some(ref git) = self.git_context {
            items.push(ContextItem {
                item_type: ContextItemType::GitStatus,
                content: format!(
                    "Git: branch={}, modified={}, staged={}",
                    git.branch, git.modified_count, git.staged_count
                ),
                relevance: 0.6,
                source: None,
            });
        }

        // Add recently accessed files
        for (idx, path) in self.recent_files.iter().take(10).enumerate() {
            let relevance = 1.0 - (idx as f32 * 0.1);
            if let Some(context) = self.active_files.get(path) {
                items.push(ContextItem {
                    item_type: ContextItemType::FileSummary,
                    content: format!(
                        "File: {} ({} bytes, {:?})",
                        path.display(),
                        context.size_bytes,
                        context.language
                    ),
                    relevance,
                    source: Some(path.clone()),
                });
            }
        }

        items
    }

    /// Get all tracked files
    pub fn tracked_files(&self) -> impl Iterator<Item = &FileContext> {
        self.active_files.values()
    }

    /// Get git context if available
    pub fn git_context(&self) -> Option<&GitContext> {
        self.git_context.as_ref()
    }

    /// Get detected project type
    pub fn project_type(&self) -> Option<&ProjectType> {
        self.project_type.as_ref()
    }
}

/// Detect programming language from file extension.
///
/// Returns a static string slice to avoid heap allocation for every call,
/// since the set of recognised languages is fixed at compile time.
const LANGUAGES: &[(&str, &str)] = &[
    ("rs", "rust"),
    ("py", "python"),
    ("js", "javascript"),
    ("jsx", "javascript"),
    ("ts", "typescript"),
    ("tsx", "typescript"),
    ("go", "go"),
    ("java", "java"),
    ("kt", "kotlin"),
    ("c", "c"),
    ("h", "c"),
    ("cpp", "cpp"),
    ("cc", "cpp"),
    ("cxx", "cpp"),
    ("hpp", "cpp"),
    ("rb", "ruby"),
    ("php", "php"),
    ("swift", "swift"),
    ("sh", "shell"),
    ("bash", "shell"),
    ("sql", "sql"),
    ("md", "markdown"),
    ("json", "json"),
    ("yaml", "yaml"),
    ("yml", "yaml"),
    ("toml", "toml"),
    ("xml", "xml"),
    ("html", "html"),
    ("htm", "html"),
    ("css", "css"),
    ("scss", "scss"),
    ("sass", "scss"),
];

pub fn detect_language(ext: &str) -> Option<&'static str> {
    for (name, language) in LANGUAGES {
        if ext.eq_ignore_ascii_case(name) {
            return Some(language);
        }
    }
    None
}

#[cfg(all(test, unix, feature = "tools-git"))]
mod git_context_tests {
    use super::ContextManager;

    /// A repository with its own branch name, made outside the workspace.
    #[expect(clippy::expect_used, reason = "test helper panics with context")]
    fn outside_repo(branch: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("outside repo");
        let repo = git2::Repository::init(dir.path()).expect("init");
        repo.remote("origin", "https://token@example.invalid/outside.git")
            .expect("remote");
        let sig = git2::Signature::now("t", "t@example.invalid").expect("sig");
        let tree = repo
            .find_tree(repo.index().expect("index").write_tree().expect("tree"))
            .expect("find tree");
        let commit = repo
            .commit(None, &sig, &sig, "init", &tree, &[])
            .expect("commit");
        repo.branch(branch, &repo.find_commit(commit).expect("c"), true)
            .expect("branch");
        repo.set_head(&format!("refs/heads/{branch}"))
            .expect("head");
        dir
    }

    /// A workspace whose `.git` is a link to another repository gives no
    /// git context; one with its own repository gives its own.
    #[tokio::test]
    async fn only_the_workspace_own_repository_goes_into_context() {
        let outside = outside_repo("outside-branch");
        let linked = tempfile::tempdir().expect("linked workspace");
        std::os::unix::fs::symlink(outside.path().join(".git"), linked.path().join(".git"))
            .expect("link");
        let own = outside_repo("own-branch");

        let from_link = ContextManager::new(linked.path().to_path_buf())
            .refresh_git_context()
            .await
            .expect("refresh");
        let from_own = ContextManager::new(own.path().to_path_buf())
            .refresh_git_context()
            .await
            .expect("refresh");

        assert!(from_link.is_none(), "{from_link:?}");
        assert_eq!(
            from_own.map(|context| context.branch).as_deref(),
            Some("own-branch")
        );
    }
}

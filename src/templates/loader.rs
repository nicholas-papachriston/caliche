//! Template loader for loading templates from files or strings.

#[cfg(feature = "s3")]
use crate::data::storage::object::backend::StorageBackend;
use crate::templates::{AgentTemplate, TemplateValidator};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Loads agent templates from various sources
pub struct TemplateLoader {
    validator: TemplateValidator,
}

impl TemplateLoader {
    /// Create a new template loader
    pub fn new() -> Self {
        Self {
            validator: TemplateValidator::new(),
        }
    }

    /// Load a template from a file path
    pub async fn load_template<P: AsRef<Path>>(&self, path: P) -> Result<AgentTemplate> {
        let path = path.as_ref();
        let content = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("Failed to read template file: {}", path.display()))?;

        self.load_template_from_str(&content, Some(path.to_path_buf()))
    }

    /// Load a template from a string
    pub fn load_template_from_str(
        &self,
        content: &str,
        file_path: Option<PathBuf>,
    ) -> Result<AgentTemplate> {
        // Try YAML first, then JSON
        let template: AgentTemplate = if content.trim_start().starts_with('{') {
            // JSON format
            serde_json::from_str(content).context("Failed to parse template as JSON")?
        } else {
            // YAML format
            #[cfg(feature = "templates")]
            {
                serde_norway::from_str(content).context("Failed to parse template as YAML")?
            }
            #[cfg(not(feature = "templates"))]
            {
                return Err(anyhow::anyhow!(
                    "YAML parsing requires the 'templates' feature to be enabled"
                ));
            }
        };

        // Validate the template
        self.validator.validate(&template)?;

        // Set file path if provided
        let mut template = template;
        template.file_path = file_path;

        Ok(template)
    }

    /// Load all templates from a directory (supports nested structure: prompts/<name>/<version>.yaml)
    ///
    /// A relative `dir` is below the current directory, which is usually a
    /// project, so the walk follows no link below the current directory: a
    /// link committed into the project cannot load a template from outside
    /// it. An absolute `dir` is trusted as named, and no link below it is
    /// followed. A link, FIFO or other special file in the tree is skipped.
    ///
    /// # Errors
    ///
    /// Returns an error when the directory, or one below it, cannot be listed
    /// or is a link.
    pub async fn load_templates_from_dir<P: AsRef<Path>>(
        &self,
        dir: P,
    ) -> Result<Vec<AgentTemplate>> {
        let dir = dir.as_ref().to_path_buf();
        let files = tokio::task::spawn_blocking(move || {
            let base = if dir.is_absolute() {
                PathBuf::new()
            } else {
                std::env::current_dir()?.canonicalize()?
            };
            template_files(&base, &dir)
        })
        .await??;
        let mut templates = Vec::new();
        for (path, content) in files {
            let loaded = String::from_utf8(content)
                .context("template is not UTF-8")
                .and_then(|content| self.load_template_from_str(&content, Some(path.clone())));
            match loaded {
                Ok(template) => templates.push(template),
                Err(e) => {
                    tracing::warn!("Failed to load template {}: {}", path.display(), e);
                }
            }
        }
        Ok(templates)
    }

    /// Load templates from a storage backend
    #[cfg(all(feature = "s3", feature = "templates"))]
    pub async fn load_templates_from_storage(
        &self,
        backend: &dyn StorageBackend,
        prefix: Option<&str>,
    ) -> Result<Vec<AgentTemplate>> {
        use crate::templates::storage::load_template_from_storage;

        let prefix = prefix.unwrap_or("templates/");
        let keys = backend.list(prefix).await?;

        let mut templates = Vec::new();
        for key in keys {
            // Filter for template files
            if key.ends_with(".yaml") || key.ends_with(".yml") || key.ends_with(".json") {
                match load_template_from_storage(backend, &key).await {
                    Ok(template) => templates.push(template),
                    Err(e) => {
                        tracing::warn!("Failed to load template {}: {}", key, e);
                    }
                }
            }
        }

        Ok(templates)
    }
}

impl Default for TemplateLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// The anchor a template directory is walked below, and the directory as a
/// path below that anchor. A relative `dir` is below `base`, the canonical
/// current directory.
fn template_root(base: &Path, dir: &Path) -> Result<(PathBuf, PathBuf)> {
    if dir.is_absolute() {
        let root = dir
            .canonicalize()
            .with_context(|| format!("Failed to read directory: {}", dir.display()))?;
        return Ok((root.clone(), root));
    }
    if !dir.components().all(|c| {
        matches!(
            c,
            std::path::Component::Normal(_) | std::path::Component::CurDir
        )
    }) {
        anyhow::bail!(
            "Template directory '{}' leaves the current directory",
            dir.display()
        );
    }
    let target = base.join(dir).components().collect();
    Ok((base.to_path_buf(), target))
}

/// The path and bytes of every template file below `dir`, read without
/// following a link. A relative `dir` is below `base`.
fn template_files(base: &Path, dir: &Path) -> Result<Vec<(PathBuf, Vec<u8>)>> {
    use std::io::Read as _;
    let (anchor, root) = template_root(base, dir)?;
    let mut files = Vec::new();
    let mut dirs_to_process = vec![root];
    while let Some(current_dir) = dirs_to_process.pop() {
        let entries = caliche_sandbox::list_dir_beneath(&anchor, &current_dir)
            .with_context(|| format!("Failed to read directory: {}", current_dir.display()))?;
        for entry in entries {
            let path = current_dir.join(&entry.name);
            match entry.kind {
                caliche_sandbox::EntryKind::Dir => dirs_to_process.push(path),
                // A link or special file with a template name is refused by
                // the read below, which follows no link.
                caliche_sandbox::EntryKind::File | caliche_sandbox::EntryKind::Other => {
                    let ext = path.extension().and_then(|s| s.to_str());
                    if !matches!(ext, Some("yaml" | "yml" | "json")) {
                        continue;
                    }
                    let mut content = Vec::new();
                    match caliche_sandbox::open_file_beneath(&anchor, &path)
                        .and_then(|mut file| file.read_to_end(&mut content))
                    {
                        Ok(_) => files.push((path, content)),
                        Err(e) => {
                            tracing::warn!("Failed to read template {}: {}", path.display(), e);
                        }
                    }
                }
            }
        }
    }
    Ok(files)
}

#[cfg(all(test, unix, feature = "templates"))]
mod link_tests {
    use super::TemplateLoader;

    const TEMPLATE: &str = include_str!("../../templates/tutor.yaml");

    /// A relative template directory is walked below the current directory,
    /// so a link at the directory itself, committed into the project, is
    /// refused instead of loading the templates it points to.
    #[test]
    fn a_relative_template_dir_that_is_a_link_is_refused() {
        let ws = tempfile::tempdir().expect("ws");
        let base = ws.path().canonicalize().expect("canonical ws");
        let outside = tempfile::tempdir().expect("outside");
        std::fs::write(outside.path().join("leak.yaml"), TEMPLATE).expect("leak");
        std::os::unix::fs::symlink(outside.path(), base.join("templates")).expect("dir link");

        let loaded = super::template_files(&base, std::path::Path::new("templates"));

        let err = loaded.expect_err("a linked template dir is refused");
        assert_eq!(
            err.to_string(),
            format!(
                "Failed to read directory: {}",
                base.join("templates").display()
            )
        );
    }

    /// A template directory holding a link to an outside template file and
    /// a link to an outside directory of templates loads only its own file.
    #[tokio::test]
    async fn a_linked_template_is_not_loaded() {
        let dir = tempfile::tempdir().expect("templates");
        let root = dir.path().canonicalize().expect("canonical templates");
        let outside = tempfile::tempdir().expect("outside");
        std::fs::write(root.join("own.yaml"), TEMPLATE).expect("own");
        std::fs::write(outside.path().join("leak.yaml"), TEMPLATE).expect("leak");
        std::os::unix::fs::symlink(outside.path().join("leak.yaml"), root.join("file.yaml"))
            .expect("file link");
        std::os::unix::fs::symlink(outside.path(), root.join("linked")).expect("dir link");

        let templates = TemplateLoader::new()
            .load_templates_from_dir(&root)
            .await
            .expect("load");

        let paths: Vec<_> = templates
            .iter()
            .map(|template| template.file_path.clone())
            .collect();
        assert_eq!(paths, vec![Some(root.join("own.yaml"))]);
    }
}

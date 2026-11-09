use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Default workspace-relative directory for generated artifacts.
pub const DEFAULT_ARTIFACT_DIR: &str = ".caliche/artifacts";

/// Stores generated artifacts under a workspace-scoped root.
#[derive(Debug, Clone)]
pub struct ArtifactStore {
    root: PathBuf,
}

impl ArtifactStore {
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    #[must_use]
    pub fn for_workspace(workspace_root: &Path, override_dir: Option<&Path>) -> Self {
        let root = override_dir.map_or_else(
            || workspace_root.join(DEFAULT_ARTIFACT_DIR),
            |dir| {
                if dir.is_absolute() {
                    dir.to_path_buf()
                } else {
                    workspace_root.join(dir)
                }
            },
        );
        Self::new(root)
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// List artifacts persisted below this store root.
    ///
    /// # Errors
    ///
    /// Returns an error when artifact directories or metadata sidecars cannot be read.
    pub async fn list(&self) -> anyhow::Result<Vec<StoredArtifact>> {
        if !tokio::fs::try_exists(&self.root).await? {
            return Ok(Vec::new());
        }

        let mut artifacts = Vec::new();
        let mut pending_dirs = vec![self.root.clone()];
        while let Some(dir_path) = pending_dirs.pop() {
            let mut entries = tokio::fs::read_dir(&dir_path).await?;
            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();
                let file_type = entry.file_type().await?;
                if file_type.is_dir() {
                    pending_dirs.push(path);
                    continue;
                }
                if !file_type.is_file() || is_metadata_or_temp_file(&path) {
                    continue;
                }

                let file_metadata = entry.metadata().await?;
                let metadata_path = metadata_path_for_artifact(&path);
                let metadata = read_artifact_metadata(&metadata_path).await?;
                let id = path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or_default()
                    .to_string();
                let media_type = metadata
                    .as_ref()
                    .and_then(|metadata| metadata.media_type.clone())
                    .unwrap_or_else(|| crate::messages::detect_media_type(&path));
                let bytes = metadata
                    .as_ref()
                    .and_then(|metadata| metadata.bytes)
                    .unwrap_or(file_metadata.len());
                let created_at = metadata
                    .as_ref()
                    .and_then(|metadata| metadata.created_at.clone());

                artifacts.push(StoredArtifact {
                    id,
                    path,
                    metadata_path: metadata_path.exists().then_some(metadata_path),
                    metadata,
                    media_type,
                    bytes,
                    created_at,
                });
            }
        }
        artifacts.sort_by(|left, right| {
            left.created_at
                .cmp(&right.created_at)
                .then_with(|| left.id.cmp(&right.id))
        });
        Ok(artifacts)
    }

    /// Resolve one artifact selector against this store.
    ///
    /// # Errors
    ///
    /// Returns an error when no artifact matches, a selector is ambiguous, or a relative
    /// path would escape the artifact root.
    pub async fn resolve(&self, selector: &ArtifactSelector) -> anyhow::Result<StoredArtifact> {
        match selector {
            ArtifactSelector::Latest => {
                self.list().await?.into_iter().last().ok_or_else(|| {
                    anyhow::anyhow!("No artifacts found under {}", self.root.display())
                })
            }
            ArtifactSelector::RelativePath(relative_path) => {
                let root = tokio::fs::canonicalize(&self.root).await?;
                let candidate = self.root.join(relative_path);
                let candidate = tokio::fs::canonicalize(&candidate).await?;
                if !candidate.starts_with(&root) {
                    anyhow::bail!(
                        "Artifact path '{}' escapes artifact root '{}'",
                        relative_path.display(),
                        self.root.display()
                    );
                }
                let mut matches = self
                    .list()
                    .await?
                    .into_iter()
                    .filter(|artifact| artifact.path == candidate)
                    .collect::<Vec<_>>();
                matches.pop().ok_or_else(|| {
                    anyhow::anyhow!("Artifact not found: {}", relative_path.display())
                })
            }
            ArtifactSelector::IdOrStemPrefix(prefix) => {
                let matches = self
                    .list()
                    .await?
                    .into_iter()
                    .filter(|artifact| artifact.matches_prefix(prefix))
                    .collect::<Vec<_>>();
                match matches.as_slice() {
                    [] => Err(anyhow::anyhow!("Artifact not found: {prefix}")),
                    [artifact] => Ok(artifact.clone()),
                    _ => Err(anyhow::anyhow!("Artifact selector '{prefix}' is ambiguous")),
                }
            }
        }
    }

    /// Save generated image bytes and a JSON sidecar under `images/`.
    ///
    /// # Errors
    ///
    /// Returns an error if directories cannot be created, files cannot be written,
    /// or metadata cannot be serialized.
    pub async fn save_image(
        &self,
        image: &crate::llm::GeneratedImage,
        metadata: ArtifactMetadata,
    ) -> anyhow::Result<ArtifactRecord> {
        let image_dir = self.root.join("images");
        tokio::fs::create_dir_all(&image_dir).await?;

        let extension = extension_for_media_type(&image.media_type)
            .or_else(|| metadata.output_format.clone())
            .unwrap_or_else(|| "png".to_string());
        let prefix = sanitize_prompt_prefix(&metadata.prompt);
        let id = uuid::Uuid::now_v7().simple().to_string();
        let file_stem = if prefix.is_empty() {
            id
        } else {
            format!("{prefix}-{id}")
        };

        let image_path = image_dir.join(format!("{file_stem}.{extension}"));
        write_atomic(&image_path, &image.data).await?;

        let metadata_path = image_dir.join(format!("{file_stem}.json"));
        let mut metadata = metadata;
        metadata.file_name = image_path
            .file_name()
            .and_then(|name| name.to_str())
            .map(ToOwned::to_owned);
        metadata.bytes = Some(image.data.len() as u64);
        metadata.media_type = Some(image.media_type.clone());
        metadata.created_at = Some(chrono::Utc::now().to_rfc3339());
        if metadata.revised_prompt.is_none() {
            metadata.revised_prompt = image.revised_prompt.clone();
        }
        let metadata_bytes = serde_json::to_vec_pretty(&metadata)?;
        write_atomic(&metadata_path, &metadata_bytes).await?;

        Ok(ArtifactRecord {
            path: image_path,
            media_type: image.media_type.clone(),
            bytes: image.data.len() as u64,
            metadata_path: Some(metadata_path),
        })
    }
}

#[derive(Debug, Clone)]
pub struct StoredArtifact {
    pub id: String,
    pub path: PathBuf,
    pub metadata_path: Option<PathBuf>,
    pub metadata: Option<ArtifactMetadata>,
    pub media_type: String,
    pub bytes: u64,
    pub created_at: Option<String>,
}

impl StoredArtifact {
    #[must_use]
    pub fn matches_prefix(&self, prefix: &str) -> bool {
        let Some(file_name) = self.path.file_name().and_then(|name| name.to_str()) else {
            return self.id.starts_with(prefix);
        };
        self.id.starts_with(prefix) || file_name.starts_with(prefix)
    }
}

#[derive(Debug, Clone)]
pub enum ArtifactSelector {
    Latest,
    IdOrStemPrefix(String),
    RelativePath(PathBuf),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRecord {
    pub path: PathBuf,
    pub media_type: String,
    pub bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArtifactMetadata {
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
    pub provider: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
}

async fn read_artifact_metadata(path: &Path) -> anyhow::Result<Option<ArtifactMetadata>> {
    if !tokio::fs::try_exists(path).await? {
        return Ok(None);
    }
    let bytes = tokio::fs::read(path).await?;
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(anyhow::Error::from)
}

fn metadata_path_for_artifact(path: &Path) -> PathBuf {
    path.with_extension("json")
}

fn is_metadata_or_temp_file(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| matches!(extension, "json" | "tmp"))
}

async fn write_atomic(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let tmp_path = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("artifact")
    ));
    tokio::fs::write(&tmp_path, bytes).await?;
    tokio::fs::rename(&tmp_path, path).await?;
    Ok(())
}

fn extension_for_media_type(media_type: &str) -> Option<String> {
    let media_type = media_type
        .split(';')
        .next()
        .unwrap_or(media_type)
        .trim()
        .to_ascii_lowercase();
    match media_type.as_str() {
        "image/png" => Some("png".to_string()),
        "image/jpeg" | "image/jpg" => Some("jpg".to_string()),
        "image/webp" => Some("webp".to_string()),
        _ => None,
    }
}

fn sanitize_prompt_prefix(prompt: &str) -> String {
    let mut out = String::new();
    for ch in prompt.chars() {
        if out.len() >= 48 {
            break;
        }
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if (ch.is_ascii_whitespace() || matches!(ch, '-' | '_')) && !out.ends_with('-') {
            out.push('-');
        }
    }
    out.trim_matches('-').to_string()
}

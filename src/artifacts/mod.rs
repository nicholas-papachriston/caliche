use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};

/// Default workspace-relative directory for generated artifacts.
pub const DEFAULT_ARTIFACT_DIR: &str = ".caliche/artifacts";

/// Stores generated artifacts under a workspace-scoped root.
///
/// The store reads and writes below an anchor directory without following
/// any link below it: the workspace for a store inside the workspace, and
/// the store root itself otherwise. A store directory, or anything in it,
/// that is a link is refused, so a link committed into a project cannot
/// put a file from outside it into a prompt or take a write outside it.
#[derive(Debug, Clone)]
pub struct ArtifactStore {
    root: PathBuf,
    anchor: PathBuf,
}

impl ArtifactStore {
    /// A store at `root`, which is trusted as named: links below it are
    /// not followed.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self {
            anchor: root.clone(),
            root,
        }
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
        let inside = root.strip_prefix(workspace_root).is_ok_and(|below| {
            below
                .components()
                .all(|c| matches!(c, Component::Normal(_)))
        });
        Self {
            anchor: if inside {
                workspace_root.to_path_buf()
            } else {
                root.clone()
            },
            root,
        }
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The canonical anchor and the store root below it.
    fn beneath(&self) -> std::io::Result<(PathBuf, PathBuf)> {
        let anchor = self.anchor.canonicalize()?;
        let below = self
            .root
            .strip_prefix(&self.anchor)
            .unwrap_or_else(|_| Path::new(""));
        let root = anchor.join(below);
        Ok((anchor, root))
    }

    /// List artifacts persisted below this store root.
    ///
    /// # Errors
    ///
    /// Returns an error when artifact directories or metadata sidecars cannot
    /// be read, or the store directory or one below it is a link.
    pub async fn list(&self) -> anyhow::Result<Vec<StoredArtifact>> {
        let store = self.clone();
        let mut artifacts = tokio::task::spawn_blocking(move || store.list_blocking()).await??;
        artifacts.sort_by(|left, right| {
            left.created_at
                .cmp(&right.created_at)
                .then_with(|| left.id.cmp(&right.id))
        });
        Ok(artifacts)
    }

    fn list_blocking(&self) -> anyhow::Result<Vec<StoredArtifact>> {
        let (anchor, root) = match self.beneath() {
            Ok(found) => found,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => return Err(err.into()),
        };
        let mut artifacts = Vec::new();
        let mut pending_dirs = vec![root.clone()];
        while let Some(dir_path) = pending_dirs.pop() {
            let entries = match caliche_sandbox::list_dir_beneath(&anchor, &dir_path) {
                Ok(entries) => entries,
                Err(err) if err.kind() == std::io::ErrorKind::NotFound && dir_path == root => {
                    return Ok(Vec::new());
                }
                Err(err) => {
                    return Err(anyhow::Error::from(err)
                        .context(format!("cannot list artifacts in {}", dir_path.display())));
                }
            };
            for entry in entries {
                let path = dir_path.join(&entry.name);
                match entry.kind {
                    caliche_sandbox::EntryKind::Dir => {
                        pending_dirs.push(path);
                        continue;
                    }
                    caliche_sandbox::EntryKind::Other => continue,
                    caliche_sandbox::EntryKind::File => {}
                }
                if is_metadata_or_temp_file(&path) {
                    continue;
                }
                let metadata_path = metadata_path_for_artifact(&path);
                let metadata = read_artifact_metadata(&anchor, &metadata_path)?;
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
                    .unwrap_or(entry.len);
                let created_at = metadata
                    .as_ref()
                    .and_then(|metadata| metadata.created_at.clone());

                artifacts.push(StoredArtifact {
                    id,
                    path,
                    metadata_path: metadata.is_some().then_some(metadata_path),
                    metadata,
                    media_type,
                    bytes,
                    created_at,
                });
            }
        }
        Ok(artifacts)
    }

    /// The content of an artifact this store listed, read without following
    /// a link, so a file swapped for a link after the listing is refused.
    ///
    /// # Errors
    ///
    /// Returns an error when the artifact is not below this store, a
    /// component on the way is a link, or it is not a regular file.
    pub async fn read(&self, artifact: &StoredArtifact) -> anyhow::Result<Vec<u8>> {
        let store = self.clone();
        let path = artifact.path.clone();
        tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<u8>> {
            use std::io::Read as _;
            let (anchor, root) = store.beneath()?;
            if !path.starts_with(&root) {
                anyhow::bail!("Artifact '{}' is not in this store", path.display());
            }
            let mut bytes = Vec::new();
            caliche_sandbox::open_file_beneath(&anchor, &path)
                .with_context(|| format!("cannot read artifact {}", path.display()))?
                .read_to_end(&mut bytes)?;
            Ok(bytes)
        })
        .await?
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
                if !relative_path
                    .components()
                    .all(|c| matches!(c, Component::Normal(_) | Component::CurDir))
                {
                    anyhow::bail!(
                        "Artifact path '{}' escapes artifact root '{}'",
                        relative_path.display(),
                        self.root.display()
                    );
                }
                let (_, root) = self.beneath()?;
                let candidate: PathBuf = root.join(relative_path).components().collect();
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
        if self.anchor == self.root {
            tokio::fs::create_dir_all(&self.root).await?;
        }
        let (anchor, root) = self.beneath()?;
        let image_dir = root.join("images");

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
        write_atomic(&anchor, &image_path, &image.data).await?;

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
        write_atomic(&anchor, &metadata_path, &metadata_bytes).await?;

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

/// The sidecar at `path`, below `anchor`, read without following a link.
/// A missing sidecar is `None`; one that is a link or not a regular file is
/// an error.
fn read_artifact_metadata(anchor: &Path, path: &Path) -> anyhow::Result<Option<ArtifactMetadata>> {
    use std::io::Read as _;
    let mut file = match caliche_sandbox::open_file_beneath(anchor, path) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(anyhow::Error::from(err)
                .context(format!("cannot read artifact metadata {}", path.display())));
        }
    };
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
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

/// Write `bytes` to a temporary file beside `path`, below `anchor` and
/// following no link, then rename it over `path`.
async fn write_atomic(anchor: &Path, path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let tmp_path = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("artifact")
    ));
    let (anchor, tmp, bytes) = (anchor.to_path_buf(), tmp_path.clone(), bytes.to_vec());
    tokio::task::spawn_blocking(move || -> std::io::Result<()> {
        use std::io::Write as _;
        let mut file = caliche_sandbox::create_file_beneath(&anchor, &tmp)?;
        file.write_all(&bytes)?;
        file.sync_all()
    })
    .await??;
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

#[cfg(test)]
#[cfg(unix)]
mod link_tests {
    use super::{ArtifactMetadata, ArtifactSelector, ArtifactStore};
    use std::os::unix::fs::symlink;
    use std::path::PathBuf;

    const OUTSIDE: &[u8] = b"OUTSIDE-BYTES";

    /// A canonical workspace with an empty `.caliche/artifacts/images`, and
    /// an outside directory holding `secret.png` and `secret.json`.
    fn layout() -> (tempfile::TempDir, tempfile::TempDir, PathBuf, PathBuf) {
        let ws = tempfile::tempdir().expect("ws");
        let out = tempfile::tempdir().expect("outside");
        let root = ws.path().canonicalize().expect("root");
        let outside = out.path().canonicalize().expect("outside");
        std::fs::create_dir_all(root.join(".caliche/artifacts/images")).expect("images");
        std::fs::write(outside.join("secret.png"), OUTSIDE).expect("secret png");
        std::fs::write(outside.join("secret.json"), r#"{"prompt":"OUTSIDE"}"#)
            .expect("secret json");
        (ws, out, root, outside)
    }

    fn images(root: &std::path::Path) -> PathBuf {
        root.join(".caliche/artifacts/images")
    }

    /// A link in the store is not listed as an artifact.
    #[tokio::test]
    async fn a_link_in_the_store_is_not_listed() {
        let (_ws, _out, root, outside) = layout();
        std::fs::write(images(&root).join("ok.png"), b"inside").expect("ok");
        symlink(outside.join("secret.png"), images(&root).join("leak.png")).expect("link");
        let store = ArtifactStore::for_workspace(&root, None);

        let ids: Vec<String> = store
            .list()
            .await
            .expect("list")
            .into_iter()
            .map(|artifact| artifact.id)
            .collect();

        assert_eq!(ids, vec!["ok".to_string()]);
    }

    /// An artifact swapped for a link after it was listed is not read.
    #[tokio::test]
    async fn an_artifact_swapped_for_a_link_is_not_read() {
        let (_ws, _out, root, outside) = layout();
        let path = images(&root).join("ok.png");
        std::fs::write(&path, b"inside").expect("ok");
        let store = ArtifactStore::for_workspace(&root, None);
        let artifact = store
            .resolve(&ArtifactSelector::Latest)
            .await
            .expect("listed");
        std::fs::remove_file(&path).expect("remove");
        symlink(outside.join("secret.png"), &path).expect("swap");

        let err = store.read(&artifact).await.expect_err("a link is refused");

        assert_eq!(
            err.to_string(),
            format!("cannot read artifact {}", path.display())
        );
    }

    /// A sidecar that is a link to an outside file is refused, not read.
    #[tokio::test]
    async fn a_linked_sidecar_is_refused() {
        let (_ws, _out, root, outside) = layout();
        std::fs::write(images(&root).join("ok.png"), b"inside").expect("ok");
        let sidecar = images(&root).join("ok.json");
        symlink(outside.join("secret.json"), &sidecar).expect("link");
        let store = ArtifactStore::for_workspace(&root, None);

        let err = store.list().await.expect_err("a linked sidecar is refused");

        assert_eq!(
            err.to_string(),
            format!("cannot read artifact metadata {}", sidecar.display())
        );
    }

    /// Saving into a store whose `images` is a link to an outside
    /// directory writes nothing there.
    #[tokio::test]
    async fn a_save_writes_nothing_through_a_linked_images_dir() {
        let (_ws, _out, root, outside) = layout();
        std::fs::remove_dir(images(&root)).expect("remove images");
        let drop_dir = outside.join("drop");
        std::fs::create_dir(&drop_dir).expect("drop");
        symlink(&drop_dir, images(&root)).expect("link");
        let store = ArtifactStore::for_workspace(&root, None);
        let image = crate::llm::GeneratedImage {
            data: b"generated".to_vec(),
            media_type: "image/png".to_string(),
            revised_prompt: None,
        };

        let saved = store.save_image(&image, ArtifactMetadata::default()).await;

        let written: Vec<_> = std::fs::read_dir(&drop_dir).expect("drop").collect();
        assert!(saved.is_err(), "{saved:?}");
        assert_eq!(written.len(), 0);
    }

    /// A relative artifact path with a parent component is refused before
    /// any lookup.
    #[tokio::test]
    async fn a_relative_path_with_a_parent_is_refused() {
        let (_ws, _out, root, _outside) = layout();
        let store = ArtifactStore::for_workspace(&root, None);

        let err = store
            .resolve(&ArtifactSelector::RelativePath(PathBuf::from(
                "images/../../../secret.png",
            )))
            .await
            .expect_err("a parent component is refused");

        assert_eq!(
            err.to_string(),
            format!(
                "Artifact path 'images/../../../secret.png' escapes artifact root '{}'",
                root.join(".caliche/artifacts").display()
            )
        );
    }
}

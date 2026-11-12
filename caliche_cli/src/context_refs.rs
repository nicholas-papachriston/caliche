//! Inline `@` context references for CLI prompts.

use anyhow::Context as _;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ResolvedContext {
    pub message: String,
    pub attachments: Vec<caliche::messages::Attachment>,
    pub allowed_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct ContextMention {
    pub span: std::ops::Range<usize>,
    pub raw: String,
    pub reference: ContextReference,
}

/// A context reference parsed from @ symbol.
#[derive(Debug, Clone)]
pub enum ContextReference {
    /// Reference to a file path.
    File(PathBuf),
    /// Reference to a directory.
    Directory(PathBuf),
    /// Reference to a URL.
    Url(String),
    /// Reference to a symbol/identifier.
    Symbol(String),
    /// Reference to clipboard content.
    Clipboard,
    /// Reference to git diff.
    GitDiff,
    /// Reference to git status.
    GitStatus,
    /// Reference to a generated artifact.
    Artifact(caliche::ArtifactSelector),
    /// Reference to an MCP resource.
    Resource(ResourceSelector),
}

#[derive(Debug, Clone)]
pub struct ResourceSelector {
    pub server: Option<String>,
    pub uri: String,
}

impl ContextReference {
    /// Parse a context reference from a string without the `@` prefix.
    ///
    /// Classification uses the shared contracts helper; host-specific existence
    /// checks (directory vs file on disk) refine the result.
    pub fn parse(input: &str) -> Option<Self> {
        use caliche_contracts::context_ref::{ContextRefKind, parse_context_ref_kind};

        let kind = parse_context_ref_kind(input)?;
        match kind {
            ContextRefKind::Artifact { selector } => {
                Some(Self::Artifact(parse_artifact_selector(&selector)))
            }
            ContextRefKind::Resource { server, uri } => {
                Some(Self::Resource(ResourceSelector { server, uri }))
            }
            ContextRefKind::Clipboard => Some(Self::Clipboard),
            ContextRefKind::GitDiff => Some(Self::GitDiff),
            ContextRefKind::GitStatus => Some(Self::GitStatus),
            ContextRefKind::Url { url } => Some(Self::Url(url)),
            ContextRefKind::Directory { path } => Some(Self::Directory(PathBuf::from(path))),
            ContextRefKind::File { path } => {
                let path = PathBuf::from(path);
                if path.is_dir() {
                    Some(Self::Directory(path))
                } else {
                    Some(Self::File(path))
                }
            }
            ContextRefKind::Symbol { name } => {
                let path = PathBuf::from(&name);
                if path.exists() {
                    if path.is_dir() {
                        Some(Self::Directory(path))
                    } else {
                        Some(Self::File(path))
                    }
                } else {
                    Some(Self::Symbol(name))
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct ContextResolver {
    workspace_root: PathBuf,
    artifact_store: Option<Arc<caliche::ArtifactStore>>,
    mcp_clients: Vec<Arc<caliche::mcp::McpClient>>,
}

impl ContextResolver {
    #[must_use]
    pub fn new(
        workspace_root: impl Into<PathBuf>,
        artifact_store: Option<Arc<caliche::ArtifactStore>>,
        mcp_clients: Vec<Arc<caliche::mcp::McpClient>>,
    ) -> Self {
        Self {
            workspace_root: workspace_root.into(),
            artifact_store,
            mcp_clients,
        }
    }

    #[must_use]
    pub fn from_service(service: &caliche::AgentService) -> Self {
        let workspace_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Self::new(
            workspace_root,
            service.artifact_store(),
            service.mcp_clients(),
        )
    }

    /// Resolve every `@` reference in a prompt and preserve the original prompt shape.
    pub async fn resolve_prompt(&self, input: &str) -> ResolvedContext {
        let mentions = parse_context_mentions(input);
        if mentions.is_empty() {
            return ResolvedContext {
                message: input.to_string(),
                attachments: Vec::new(),
                allowed_paths: Vec::new(),
            };
        }

        let mut rewritten = String::with_capacity(input.len());
        let mut last = 0;
        let mut sections = Vec::new();
        let mut attachments = Vec::new();
        let mut allowed_paths = Vec::new();

        for (index, mention) in mentions.iter().enumerate() {
            rewritten.push_str(&input[last..mention.span.start]);
            let marker = marker_for(index + 1, &mention.reference);
            rewritten.push_str(&marker);
            last = mention.span.end;

            match self.resolve_reference(&mention.reference).await {
                Ok(resolved) => {
                    sections.push(format!("{marker}\n{}", resolved.text));
                    attachments.extend(resolved.attachments);
                    allowed_paths.extend(resolved.allowed_paths);
                }
                Err(error) => {
                    sections.push(format!(
                        "{marker}\n[Error loading @{}: {error}]",
                        mention.raw
                    ));
                }
            }
        }
        rewritten.push_str(&input[last..]);

        let prompt = if rewritten.trim().is_empty() {
            "Explain or summarize the attached content.".to_string()
        } else {
            rewritten.trim().to_string()
        };
        let message = if sections.is_empty() {
            prompt
        } else {
            format!("{prompt}\n\nAttached context:\n{}", sections.join("\n\n"))
        };

        ResolvedContext {
            message,
            attachments,
            allowed_paths,
        }
    }

    async fn resolve_reference(
        &self,
        reference: &ContextReference,
    ) -> anyhow::Result<ResolvedItem> {
        match reference {
            ContextReference::File(path) => self.resolve_file(path).await,
            ContextReference::Directory(path) => {
                resolve_directory(&self.workspace_root, path).await
            }
            ContextReference::Url(url) => resolve_url(url).await,
            ContextReference::Symbol(symbol) => Ok(ResolvedItem::text(format!(
                "[Reference to symbol: {symbol}]"
            ))),
            ContextReference::Clipboard => resolve_clipboard(),
            ContextReference::GitDiff => resolve_git(["diff", "HEAD"], "Git Diff").await,
            ContextReference::GitStatus => resolve_git(["status", "--short"], "Git Status").await,
            ContextReference::Artifact(selector) => self.resolve_artifact(selector).await,
            ContextReference::Resource(selector) => self.resolve_resource(selector).await,
        }
    }

    async fn resolve_file(&self, path: &Path) -> anyhow::Result<ResolvedItem> {
        let bytes = read_reference(&self.workspace_root, path).await?;
        let filename = path
            .file_name()
            .map(|name| name.to_string_lossy())
            .unwrap_or_default();
        let mut item = if let Ok(content) = String::from_utf8(bytes.clone()) {
            ResolvedItem::text(format!("--- File: {filename} ---\n{content}"))
        } else {
            let media_type = caliche::messages::detect_media_type(path);
            let attachment_type = caliche::messages::detect_attachment_type(&media_type);
            let max_bytes = 1_000_000usize;
            let clipped = if bytes.len() > max_bytes {
                &bytes[..max_bytes]
            } else {
                &bytes[..]
            };
            let encoded = BASE64.encode(clipped);
            ResolvedItem::text(format!(
                "--- Attachment: {filename} ---\nType: {attachment_type:?}\nMedia-Type: {media_type}\nBytes: {}\nBase64: {encoded}\n{}",
                bytes.len(),
                if bytes.len() > max_bytes {
                    "[truncated binary payload]"
                } else {
                    ""
                }
            ))
        };
        item.allowed_paths
            .push(resolve_allowed_path(path, &self.workspace_root));
        Ok(item)
    }

    async fn resolve_artifact(
        &self,
        selector: &caliche::ArtifactSelector,
    ) -> anyhow::Result<ResolvedItem> {
        let store = self
            .artifact_store
            .as_ref()
            .context("No artifact store configured")?;
        let artifact = store.resolve(selector).await?;
        let metadata = artifact
            .metadata
            .as_ref()
            .and_then(|metadata| serde_json::to_string_pretty(metadata).ok())
            .unwrap_or_else(|| "{}".to_string());
        let header = format!(
            "--- Artifact: {} ---\nPath: {}\nMedia-Type: {}\nBytes: {}\nMetadata:\n{}",
            artifact.id,
            artifact.path.display(),
            artifact.media_type,
            artifact.bytes,
            metadata
        );
        let attachment_type = caliche::messages::detect_attachment_type(&artifact.media_type);
        let mut item = if matches!(
            attachment_type,
            caliche::messages::AttachmentType::Image
                | caliche::messages::AttachmentType::Audio
                | caliche::messages::AttachmentType::Video
                | caliche::messages::AttachmentType::Document
        ) {
            let attachment = caliche::messages::Attachment::from_file(&artifact.path).await?;
            ResolvedItem {
                text: format!("{header}\n[Attached as native media]"),
                attachments: vec![attachment],
                allowed_paths: vec![artifact.path.clone()],
            }
        } else {
            let content = tokio::fs::read_to_string(&artifact.path).await?;
            ResolvedItem::text(format!("{header}\nContent:\n{content}"))
        };
        if item.allowed_paths.is_empty() {
            item.allowed_paths.push(artifact.path);
        }
        Ok(item)
    }

    async fn resolve_resource(&self, selector: &ResourceSelector) -> anyhow::Result<ResolvedItem> {
        let client = if let Some(server) = &selector.server {
            self.mcp_clients
                .iter()
                .find(|client| client.name() == server)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("MCP resource server not found: {server}"))?
        } else {
            self.find_unique_resource_client(&selector.uri).await?
        };
        let result = client.read_resource(selector.uri.clone()).await?;
        let mut sections = Vec::new();
        let mut attachments = Vec::new();
        for content in result.contents {
            let mime_type = content
                .mime_type
                .clone()
                .unwrap_or_else(|| "text/plain".to_string());
            if let Some(text) = content.text {
                sections.push(format!(
                    "--- Resource: {} ---\nMedia-Type: {}\n{}",
                    content.uri, mime_type, text
                ));
            }
            if let Some(blob) = content.blob {
                let bytes = BASE64
                    .decode(blob.as_bytes())
                    .context("Invalid base64 resource blob")?;
                let attachment_type = caliche::messages::detect_attachment_type(&mime_type);
                let filename = resource_filename(&content.uri);
                attachments.push(caliche::messages::Attachment::new(
                    bytes,
                    mime_type.clone(),
                    filename,
                    attachment_type,
                ));
                sections.push(format!(
                    "--- Resource: {} ---\nMedia-Type: {}\n[Attached as native media]",
                    content.uri, mime_type
                ));
            }
        }
        Ok(ResolvedItem {
            text: sections.join("\n\n"),
            attachments,
            allowed_paths: Vec::new(),
        })
    }

    async fn find_unique_resource_client(
        &self,
        uri: &str,
    ) -> anyhow::Result<Arc<caliche::mcp::McpClient>> {
        let mut matches = Vec::new();
        for client in &self.mcp_clients {
            match client.list_resources().await {
                Ok(result) => {
                    if result.resources.iter().any(|resource| resource.uri == uri) {
                        matches.push(Arc::clone(client));
                    }
                }
                Err(error) => {
                    tracing::debug!(
                        server = %client.name(),
                        error = %error,
                        "failed to list MCP resources"
                    );
                }
            }
        }
        match matches.as_slice() {
            [] => Err(anyhow::anyhow!("MCP resource not found: {uri}")),
            [client] => Ok(Arc::clone(client)),
            _ => Err(anyhow::anyhow!(
                "MCP resource URI is ambiguous; use @resource:<server>::{uri}"
            )),
        }
    }
}

struct ResolvedItem {
    text: String,
    attachments: Vec<caliche::messages::Attachment>,
    allowed_paths: Vec<PathBuf>,
}

impl ResolvedItem {
    fn text(text: String) -> Self {
        Self {
            text,
            attachments: Vec::new(),
            allowed_paths: Vec::new(),
        }
    }
}

/// Parse all @ references from input and return the rewritten-clean input plus references.
///
/// This compatibility helper preserves the old public API. New call sites should use
/// [`ContextResolver::resolve_prompt`] so multiple inline tags retain their positions.
pub fn parse_context_references(input: &str) -> (String, Vec<ContextReference>) {
    let mentions = parse_context_mentions(input);
    let mut clean = String::with_capacity(input.len());
    let mut last = 0;
    let mut refs = Vec::with_capacity(mentions.len());
    for mention in mentions {
        clean.push_str(&input[last..mention.span.start]);
        last = mention.span.end;
        refs.push(mention.reference);
    }
    clean.push_str(&input[last..]);
    (clean.trim().to_string(), refs)
}

#[must_use]
pub fn parse_context_mentions(input: &str) -> Vec<ContextMention> {
    let mut mentions = Vec::new();
    let mut iter = input.char_indices().peekable();
    while let Some((index, ch)) = iter.next() {
        if ch != '@' || !is_tag_start(input, index) {
            continue;
        }
        let token_start = index + ch.len_utf8();
        let mut token_end = input.len();
        while let Some(&(next_index, next_ch)) = iter.peek() {
            if next_ch.is_whitespace() {
                token_end = next_index;
                break;
            }
            let _ = iter.next();
        }
        let trimmed_end = trim_tag_end(input, token_start, token_end);
        if trimmed_end <= token_start {
            continue;
        }
        let raw = input[token_start..trimmed_end].to_string();
        if let Some(reference) = ContextReference::parse(&raw) {
            mentions.push(ContextMention {
                span: index..trimmed_end,
                raw,
                reference,
            });
        }
    }
    mentions
}

fn parse_artifact_selector(input: &str) -> caliche::ArtifactSelector {
    if input.is_empty() || input == "latest" {
        caliche::ArtifactSelector::Latest
    } else if input.contains('/') || input.contains('\\') {
        caliche::ArtifactSelector::RelativePath(PathBuf::from(input))
    } else {
        caliche::ArtifactSelector::IdOrStemPrefix(input.to_string())
    }
}

async fn resolve_directory(workspace_root: &Path, path: &Path) -> anyhow::Result<ResolvedItem> {
    let mut entries = Vec::new();
    for (name, is_dir) in list_reference(workspace_root, path).await? {
        let suffix = if is_dir { "/" } else { "" };
        entries.push(format!("  {name}{suffix}"));
    }
    entries.sort();
    Ok(ResolvedItem {
        text: format!(
            "--- Directory: {} ---\n{}",
            path.display(),
            entries.join("\n")
        ),
        attachments: Vec::new(),
        allowed_paths: vec![path.to_path_buf()],
    })
}

async fn resolve_url(url: &str) -> anyhow::Result<ResolvedItem> {
    let response = reqwest::get(url).await?;
    let text = response.text().await?;
    let truncated = if text.len() > 10000 {
        format!(
            "{}...\n[truncated, {} total chars]",
            caliche_contracts::text::truncate_utf8(&text, 10000),
            text.len()
        )
    } else {
        text
    };
    Ok(ResolvedItem::text(format!(
        "--- URL: {url} ---\n{truncated}"
    )))
}

fn resolve_clipboard() -> anyhow::Result<ResolvedItem> {
    match arboard::Clipboard::new() {
        Ok(mut clipboard) => match clipboard.get_text() {
            Ok(text) => {
                let truncated = if text.len() > 50000 {
                    format!(
                        "{}...\n[truncated, {} total chars]",
                        caliche_contracts::text::truncate_utf8(&text, 50000),
                        text.len()
                    )
                } else {
                    text
                };
                Ok(ResolvedItem::text(format!(
                    "--- Clipboard Content ---\n{truncated}"
                )))
            }
            Err(error) => Ok(ResolvedItem::text(format!("[Clipboard error: {error}]"))),
        },
        Err(error) => Ok(ResolvedItem::text(format!(
            "[Clipboard unavailable: {error}]"
        ))),
    }
}

async fn resolve_git<const N: usize>(args: [&str; N], title: &str) -> anyhow::Result<ResolvedItem> {
    let output = tokio::process::Command::new("git")
        .args(args)
        .output()
        .await?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    if stdout.is_empty() {
        Ok(ResolvedItem::text(format!("--- {title} ---\nNo changes")))
    } else {
        Ok(ResolvedItem::text(format!("--- {title} ---\n{stdout}")))
    }
}

fn marker_for(index: usize, reference: &ContextReference) -> String {
    let label = match reference {
        ContextReference::File(path) => format!("file {}", path.display()),
        ContextReference::Directory(path) => format!("directory {}", path.display()),
        ContextReference::Url(url) => format!("url {url}"),
        ContextReference::Symbol(symbol) => format!("symbol {symbol}"),
        ContextReference::Clipboard => "clipboard".to_string(),
        ContextReference::GitDiff => "git diff".to_string(),
        ContextReference::GitStatus => "git status".to_string(),
        ContextReference::Artifact(selector) => {
            format!("artifact {}", artifact_selector_label(selector))
        }
        ContextReference::Resource(selector) => {
            if let Some(server) = &selector.server {
                format!("resource {server}::{}", selector.uri)
            } else {
                format!("resource {}", selector.uri)
            }
        }
    };
    format!("[context {index}: {label}]")
}

fn artifact_selector_label(selector: &caliche::ArtifactSelector) -> String {
    match selector {
        caliche::ArtifactSelector::Latest => "latest".to_string(),
        caliche::ArtifactSelector::IdOrStemPrefix(prefix) => prefix.clone(),
        caliche::ArtifactSelector::RelativePath(path) => path.display().to_string(),
    }
}

fn is_tag_start(input: &str, at_index: usize) -> bool {
    input[..at_index]
        .chars()
        .next_back()
        .is_none_or(|ch| ch.is_whitespace() || matches!(ch, '(' | '[' | '{' | '<' | '"' | '\''))
}

fn trim_tag_end(input: &str, token_start: usize, mut token_end: usize) -> usize {
    while token_end > token_start {
        let Some(ch) = input[token_start..token_end].chars().next_back() else {
            break;
        };
        if matches!(ch, ')' | ']' | '}' | '>' | ',' | ';' | '.') {
            token_end -= ch.len_utf8();
        } else {
            break;
        }
    }
    token_end
}

fn resource_filename(uri: &str) -> Option<String> {
    uri.rsplit('/')
        .next()
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

/// Where a reference reads from: the canonical workspace and the part of
/// the path below it, or `None` for a path the user named outside it.
fn below_workspace(workspace_root: &Path, path: &Path) -> std::io::Result<Option<PathBuf>> {
    let joined = resolve_allowed_path(path, workspace_root);
    let Ok(below) = joined.strip_prefix(workspace_root) else {
        return Ok(None);
    };
    Ok(Some(workspace_root.canonicalize()?.join(below)))
}

/// The bytes of the file a reference names.
///
/// A path inside the workspace is opened below it without following a
/// link, so a link in the project cannot pull in a file from outside. A
/// path the user names outside the workspace is read as named.
async fn read_reference(workspace_root: &Path, path: &Path) -> std::io::Result<Vec<u8>> {
    let (workspace_root, path) = (workspace_root.to_path_buf(), path.to_path_buf());
    tokio::task::spawn_blocking(move || {
        use std::io::Read;
        let Some(target) = below_workspace(&workspace_root, &path)? else {
            return std::fs::read(&path);
        };
        let root = workspace_root.canonicalize()?;
        let mut bytes = Vec::new();
        caliche_sandbox::open_file_beneath(&root, &target)?.read_to_end(&mut bytes)?;
        Ok(bytes)
    })
    .await
    .map_err(std::io::Error::other)?
}

/// The entries of the directory a reference names, with whether each is a
/// directory. A directory inside the workspace is listed as
/// [`read_reference`] reads a file there.
async fn list_reference(
    workspace_root: &Path,
    path: &Path,
) -> std::io::Result<Vec<(String, bool)>> {
    let (workspace_root, path) = (workspace_root.to_path_buf(), path.to_path_buf());
    tokio::task::spawn_blocking(move || {
        let Some(target) = below_workspace(&workspace_root, &path)? else {
            return std::fs::read_dir(&path)?
                .map(|entry| {
                    let entry = entry?;
                    let is_dir = entry.file_type()?.is_dir();
                    Ok((entry.file_name().to_string_lossy().into_owned(), is_dir))
                })
                .collect();
        };
        let root = workspace_root.canonicalize()?;
        Ok(caliche_sandbox::list_dir_beneath(&root, &target)?
            .into_iter()
            .map(|entry| {
                let is_dir = entry.kind == caliche_sandbox::EntryKind::Dir;
                (entry.name.to_string_lossy().into_owned(), is_dir)
            })
            .collect())
    })
    .await
    .map_err(std::io::Error::other)?
}

fn resolve_allowed_path(path: &Path, workspace_root: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        workspace_root.join(path)
    }
}

#[cfg(all(test, unix))]
mod link_tests {
    use super::ContextResolver;

    /// A canonical workspace with `ok.txt`, a file link `leak.txt` and a
    /// directory link `linked`, both to a file outside it.
    fn workspace() -> (tempfile::TempDir, tempfile::TempDir, std::path::PathBuf) {
        let outside = tempfile::tempdir().expect("outside dir");
        std::fs::write(outside.path().join("secret.txt"), "outside text").expect("secret");
        let ws = tempfile::tempdir().expect("workspace");
        let root = ws.path().canonicalize().expect("canonical workspace");
        std::fs::write(root.join("ok.txt"), "inside text").expect("ok");
        std::os::unix::fs::symlink(outside.path().join("secret.txt"), root.join("leak.txt"))
            .expect("file link");
        std::os::unix::fs::symlink(outside.path(), root.join("linked")).expect("dir link");
        (outside, ws, root)
    }

    #[tokio::test]
    async fn a_reference_to_a_workspace_file_is_attached() {
        let (_outside, _ws, root) = workspace();
        let resolver = ContextResolver::new(root.clone(), None, Vec::new());

        let resolved = resolver
            .resolve_prompt(&format!("@{}", root.join("ok.txt").display()))
            .await;

        assert!(
            resolved
                .message
                .ends_with("--- File: ok.txt ---\ninside text"),
            "{}",
            resolved.message
        );
    }

    #[tokio::test]
    async fn a_reference_to_a_linked_file_or_directory_attaches_nothing_from_outside() {
        let (_outside, _ws, root) = workspace();
        let resolver = ContextResolver::new(root.clone(), None, Vec::new());

        for name in ["leak.txt", "linked/secret.txt", "linked/"] {
            let resolved = resolver
                .resolve_prompt(&format!("@{}", root.join(name).display()))
                .await;

            assert!(
                resolved.message.contains("[Error loading @"),
                "{name}: {}",
                resolved.message
            );
            assert!(
                !resolved.message.contains("outside text"),
                "{name}: {}",
                resolved.message
            );
            assert!(
                !resolved.message.contains("secret.txt\n"),
                "{name}: {}",
                resolved.message
            );
        }
    }
}

//! Video processing utilities for frame extraction and temporal aggregation.

use crate::embedding::EmbeddingVector;
use crate::errors::DripstoneError;
use crate::multimodal::media::VideoInput;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Maximum downloaded video size for [`VideoInput::Url`] (`http`/`https`).
#[cfg(feature = "video-processing")]
const MAX_VIDEO_DOWNLOAD_BYTES: u64 = 512 * 1024 * 1024;

/// HTTP(S) download timeout for remote videos.
#[cfg(feature = "video-processing")]
const VIDEO_DOWNLOAD_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

/// Suffix for temp download files: last path segment extension, or `.bin` if missing or unsafe.
#[cfg(feature = "video-processing")]
fn safe_download_suffix(parsed: &url::Url) -> String {
    let raw = parsed
        .path_segments()
        .and_then(|mut s| s.next_back())
        .filter(|s| !s.is_empty())
        .map(Path::new)
        .and_then(Path::extension)
        .and_then(|e| e.to_str())
        .unwrap_or("");
    let raw = raw.trim_start_matches('.');
    if raw.is_empty() || raw.len() > 16 || !raw.bytes().all(|b| b.is_ascii_alphanumeric()) {
        ".bin".to_string()
    } else {
        format!(".{raw}")
    }
}

/// Removes a downloaded temp video on drop (HTTP(S) inputs only).
#[cfg(feature = "video-processing")]
struct TempVideoDownload(PathBuf);

#[cfg(feature = "video-processing")]
impl Drop for TempVideoDownload {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Resolve a video URL to a local path: `file:` returns the path; `http`/`https` download to a temp file.
#[cfg(feature = "video-processing")]
pub(crate) async fn resolve_video_url(url_str: &str) -> Result<PathBuf, DripstoneError> {
    use url::Url;

    let url_str = url_str.trim();
    if url_str.is_empty() {
        return Err(DripstoneError::storage("video URL is empty"));
    }

    let parsed = Url::parse(url_str)
        .map_err(|e| DripstoneError::storage_with_source("invalid video URL", e))?;

    match parsed.scheme() {
        "file" => {
            let path = parsed.to_file_path().map_err(|_| {
                DripstoneError::storage("video file URL must be a local filesystem path")
            })?;
            if !path.is_file() {
                return Err(DripstoneError::storage(format!(
                    "video file URL does not exist or is not a file: {:?}",
                    path
                )));
            }
            Ok(path)
        }
        "http" | "https" => {
            let client = reqwest::Client::builder()
                .timeout(VIDEO_DOWNLOAD_TIMEOUT)
                .redirect(reqwest::redirect::Policy::limited(8))
                .build()
                .map_err(|e| DripstoneError::storage_with_source("HTTP client build failed", e))?;

            let response = client.get(url_str).send().await.map_err(|e| {
                DripstoneError::storage_with_source("video download request failed", e)
            })?;

            if !response.status().is_success() {
                return Err(DripstoneError::storage(format!(
                    "video download failed with HTTP {}",
                    response.status()
                )));
            }

            if let Some(len) = response.content_length() {
                if len > MAX_VIDEO_DOWNLOAD_BYTES {
                    return Err(DripstoneError::storage(format!(
                        "video Content-Length {len} exceeds limit {MAX_VIDEO_DOWNLOAD_BYTES}"
                    )));
                }
            }

            let bytes = response.bytes().await.map_err(|e| {
                DripstoneError::storage_with_source("video download read failed", e)
            })?;

            if bytes.len() as u64 > MAX_VIDEO_DOWNLOAD_BYTES {
                return Err(DripstoneError::storage(format!(
                    "downloaded video size {} exceeds limit {}",
                    bytes.len(),
                    MAX_VIDEO_DOWNLOAD_BYTES
                )));
            }

            let ext = safe_download_suffix(&parsed);
            let tmp_name = format!("dripstone_vid_{}{}", uuid::Uuid::now_v7(), ext);
            let tmp_path = std::env::temp_dir().join(tmp_name);
            std::fs::write(&tmp_path, &bytes).map_err(|e| {
                DripstoneError::storage_with_source("failed to write downloaded video", e)
            })?;
            Ok(tmp_path)
        }
        other => Err(DripstoneError::storage(format!(
            "unsupported video URL scheme '{other}', expected http, https, or file"
        ))),
    }
}

/// Options for video embedding generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoOptions {
    /// Strategy for processing the video
    pub strategy: VideoStrategy,
    /// Maximum number of frames to extract
    pub max_frames: usize,
    /// Sample frames at this FPS (None = extract all frames up to max_frames)
    pub fps_sample: Option<f32>,
}

impl Default for VideoOptions {
    fn default() -> Self {
        Self {
            strategy: VideoStrategy::FrameExtraction {
                aggregation: TemporalAggregation::Mean,
            },
            max_frames: 32,
            fps_sample: Some(1.0), // Sample 1 frame per second by default
        }
    }
}

impl VideoOptions {
    /// Create new video options with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the video processing strategy.
    pub fn with_strategy(mut self, strategy: VideoStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the maximum number of frames.
    pub fn with_max_frames(mut self, max_frames: usize) -> Self {
        self.max_frames = max_frames;
        self
    }

    /// Set the FPS sampling rate.
    pub fn with_fps_sample(mut self, fps: Option<f32>) -> Self {
        self.fps_sample = fps;
        self
    }
}

/// Strategy for processing videos.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum VideoStrategy {
    /// Extract frames and aggregate using the specified method
    FrameExtraction {
        /// How to aggregate frame embeddings
        aggregation: TemporalAggregation,
    },
    /// Dedicated video encoder (not supported; use [`FrameExtraction`](Self::FrameExtraction)).
    DedicatedModel,
}

/// Temporal aggregation strategy for frame embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalAggregation {
    /// Mean pooling across all frames
    Mean,
    /// Max pooling across all frames
    Max,
    /// Return all frame embeddings separately
    All,
}

/// Error detail when [`VideoStrategy::DedicatedModel`] is used (ORT/Candle vision backends).
#[cfg(feature = "video-processing")]
pub const DEDICATED_VIDEO_UNSUPPORTED: &str = "Dedicated video models are not supported. Use VideoStrategy::FrameExtraction with fps_sample and max_frames (frame embeddings via the vision backend).";

/// Video processor for extracting frames and processing videos.
#[cfg(feature = "video-processing")]
#[derive(Debug)]
pub struct VideoProcessor;

#[cfg(feature = "video-processing")]
impl VideoProcessor {
    /// Create a new video processor.
    pub fn new() -> Self {
        Self
    }

    /// Extract frames from a video file or URL (`VideoInput::Url` downloads to a temp file, removed after ffmpeg runs).
    ///
    /// Returns paths to extracted frame images. If `max_frames == 0`, returns an empty list without invoking ffmpeg.
    pub async fn extract_frames(
        &self,
        video: &VideoInput,
        output_dir: &PathBuf,
        fps_sample: Option<f32>,
        max_frames: usize,
    ) -> Result<Vec<PathBuf>, DripstoneError> {
        if max_frames == 0 {
            return Ok(Vec::new());
        }

        if let Some(fps) = fps_sample {
            if !fps.is_finite() || fps <= 0.0 {
                return Err(DripstoneError::Misconfiguration(format!(
                    "fps_sample must be a finite number > 0, got {fps}"
                )));
            }
        }

        let (video_path, _temp_download): (PathBuf, Option<TempVideoDownload>) = match video {
            VideoInput::Path(p) => (p.clone(), None),
            VideoInput::Url(url) => {
                let path = resolve_video_url(url).await?;
                let path_for_ffmpeg = path.clone();
                (path_for_ffmpeg, Some(TempVideoDownload(path)))
            }
        };

        let output_dir = output_dir.clone();
        let vf_arg = if let Some(fps) = fps_sample {
            format!("fps={fps}")
        } else {
            "fps=1".to_string()
        };

        let join = tokio::task::spawn_blocking(move || {
            use std::process::Command;

            std::fs::create_dir_all(&output_dir).map_err(|e| {
                DripstoneError::storage_with_source("failed to create output directory", e)
            })?;

            let frame_pattern = output_dir.join("frame_%04d.jpg");
            let output = Command::new("ffmpeg")
                .arg("-i")
                .arg(&video_path)
                .arg("-vf")
                .arg(&vf_arg)
                .arg("-frames:v")
                .arg(max_frames.to_string())
                .arg("-q:v")
                .arg("2")
                .arg(&frame_pattern)
                .output()
                .map_err(|e| DripstoneError::storage_with_source("failed to execute ffmpeg", e))?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(DripstoneError::storage(format!(
                    "ffmpeg failed: {}",
                    stderr
                )));
            }

            let mut frames = Vec::new();
            for i in 1..=max_frames {
                let frame_path = output_dir.join(format!("frame_{:04}.jpg", i));
                if frame_path.exists() {
                    frames.push(frame_path);
                } else {
                    break;
                }
            }

            Ok(frames)
        })
        .await;

        let inner = match join {
            Ok(r) => r,
            Err(e) => {
                return Err(DripstoneError::storage_with_source(
                    "frame extraction task failed",
                    e,
                ));
            }
        };

        inner
    }

    /// Aggregate frame embeddings using the specified strategy.
    pub fn aggregate_embeddings(
        &self,
        frame_embeddings: &[EmbeddingVector],
        aggregation: TemporalAggregation,
    ) -> Result<Vec<EmbeddingVector>, DripstoneError> {
        if frame_embeddings.is_empty() {
            return Ok(Vec::new());
        }

        match aggregation {
            TemporalAggregation::All => Ok(frame_embeddings.to_vec()),
            TemporalAggregation::Mean => {
                let dimension = frame_embeddings[0].len();
                let mut aggregated = vec![0.0f32; dimension];

                for embedding in frame_embeddings {
                    if embedding.len() != dimension {
                        return Err(DripstoneError::InvalidVectorLength {
                            expected: dimension,
                            actual: embedding.len(),
                        });
                    }
                    for (i, val) in embedding.iter().enumerate() {
                        aggregated[i] += val;
                    }
                }

                let count = frame_embeddings.len() as f32;
                for val in aggregated.iter_mut() {
                    *val /= count;
                }

                Ok(vec![aggregated.into()])
            }
            TemporalAggregation::Max => {
                let dimension = frame_embeddings[0].len();
                let mut aggregated = vec![f32::NEG_INFINITY; dimension];

                for embedding in frame_embeddings {
                    if embedding.len() != dimension {
                        return Err(DripstoneError::InvalidVectorLength {
                            expected: dimension,
                            actual: embedding.len(),
                        });
                    }
                    for (i, val) in embedding.iter().enumerate() {
                        aggregated[i] = aggregated[i].max(*val);
                    }
                }

                Ok(vec![aggregated.into()])
            }
        }
    }
}

#[cfg(test)]
mod resolve_url_tests {
    use super::resolve_video_url;
    use std::io::Write;

    #[tokio::test]
    async fn resolve_file_url_existing_file() {
        let mut f = tempfile::NamedTempFile::new().expect("temp file");
        writeln!(f, "stub").expect("write");
        let p = f.path().canonicalize().expect("canonicalize");
        let url = format!("file://{}", p.display());
        let got = resolve_video_url(&url).await.expect("resolve");
        assert_eq!(got, p);
    }

    #[tokio::test]
    async fn reject_unknown_scheme() {
        let err = resolve_video_url("ftp://example.com/x.mp4")
            .await
            .expect_err("ftp");
        let msg = err.to_string();
        assert!(
            msg.contains("unsupported") && msg.contains("scheme"),
            "unexpected message: {msg}"
        );
    }

    #[tokio::test]
    async fn reject_empty_url_after_trim() {
        let err = resolve_video_url("  \t  ").await.expect_err("empty");
        assert!(
            err.to_string().to_lowercase().contains("empty"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn safe_suffix_accepts_simple_extension() {
        use url::Url;
        let u = Url::parse("https://cdn.example/v/trailer.mp4").expect("url");
        assert_eq!(super::safe_download_suffix(&u), ".mp4");
    }

    #[test]
    fn safe_suffix_rejects_odd_extensions() {
        use url::Url;
        let u = Url::parse("https://x/a/foo.%2e%2e%5c%2e%2e").expect("url");
        let s = super::safe_download_suffix(&u);
        assert_eq!(s, ".bin");
    }
}

#[cfg(not(feature = "video-processing"))]
pub struct VideoProcessor;

#[cfg(not(feature = "video-processing"))]
impl VideoProcessor {
    pub fn new() -> Self {
        Self
    }

    pub async fn extract_frames(
        &self,
        _video: &VideoInput,
        _output_dir: &PathBuf,
        _fps_sample: Option<f32>,
        _max_frames: usize,
    ) -> Result<Vec<PathBuf>, DripstoneError> {
        Err(DripstoneError::Misconfiguration(
            "Video processing requires the 'video-processing' feature".to_string(),
        ))
    }

    pub fn aggregate_embeddings(
        &self,
        _frame_embeddings: &[EmbeddingVector],
        _aggregation: TemporalAggregation,
    ) -> Result<Vec<EmbeddingVector>, DripstoneError> {
        Err(DripstoneError::Misconfiguration(
            "Video processing requires the 'video-processing' feature".to_string(),
        ))
    }
}

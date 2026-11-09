//! Configuration for media understanding.

#[derive(Debug, Clone)]
pub struct MediaConfig {
    pub audio_enabled: bool,
    pub image_enabled: bool,
    pub video_enabled: bool,
    pub max_file_size_mb: usize,
    pub timeout_seconds: u64,
}

impl Default for MediaConfig {
    fn default() -> Self {
        Self {
            audio_enabled: true,
            image_enabled: true,
            video_enabled: true,
            max_file_size_mb: 25,
            timeout_seconds: 60,
        }
    }
}

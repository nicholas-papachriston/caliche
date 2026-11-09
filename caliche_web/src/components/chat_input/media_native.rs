//! Native file picker via `rfd` for desktop/mobile builds.
//!
//! Opens the OS-native file dialog so attachments feel integrated with the
//! host, and reads bytes directly off disk without going through a webview
//! FileReader. Previews are only produced for common image types to keep the
//! payload small.

use dioxus::prelude::*;

use crate::api::types::MediaAttachment;

fn guess_media_type(ext: &str) -> &'static str {
    match ext.to_ascii_lowercase().as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "svg" => "image/svg+xml",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "ogg" => "audio/ogg",
        "m4a" => "audio/mp4",
        "mp4" => "video/mp4",
        "mov" => "video/quicktime",
        "webm" => "video/webm",
        "pdf" => "application/pdf",
        "txt" => "text/plain",
        "csv" => "text/csv",
        "doc" => "application/msword",
        "docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xlsx" => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        _ => "application/octet-stream",
    }
}

fn preview_for(media_type: &str, bytes: &[u8]) -> Option<String> {
    if !media_type.starts_with("image/") || bytes.len() > 4 * 1024 * 1024 {
        return None;
    }
    use base64::Engine;
    let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
    Some(format!("data:{media_type};base64,{encoded}"))
}

pub(super) fn open_media_picker(on_add_media: EventHandler<Vec<MediaAttachment>>) {
    spawn(async move {
        let handles = rfd::AsyncFileDialog::new()
            .add_filter(
                "media",
                &[
                    "png", "jpg", "jpeg", "gif", "webp", "bmp", "svg", "mp3", "wav", "ogg", "m4a",
                    "mp4", "mov", "webm", "pdf", "txt", "csv", "doc", "docx", "xlsx",
                ],
            )
            .pick_files()
            .await;
        let Some(handles) = handles else {
            return;
        };

        let mut attachments = Vec::with_capacity(handles.len());
        for handle in handles {
            let filename = handle.file_name();
            let ext = std::path::Path::new(&filename)
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            let media_type = guess_media_type(ext).to_string();
            let bytes = handle.read().await;
            let preview_url = preview_for(&media_type, &bytes);
            attachments.push(MediaAttachment {
                filename,
                media_type,
                bytes,
                preview_url,
            });
        }

        if !attachments.is_empty() {
            on_add_media.call(attachments);
        }
    });
}

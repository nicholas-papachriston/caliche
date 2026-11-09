//! Web/WebView-based media attachment picker and drag-drop handler.
//!
//! Uses `dioxus::document::eval` to trigger a hidden `<input type=file>` and
//! read files via `FileReader`. This path is the default for the `web` feature
//! and also works inside the `desktop`/`mobile` Dioxus webview bundles.

use base64::Engine;
use dioxus::prelude::*;
use serde_json::Value;

use crate::api::types::MediaAttachment;

pub(super) fn parse_media_attachments(value: Value) -> Vec<MediaAttachment> {
    value
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|item| {
            let filename = item.get("filename")?.as_str()?.to_string();
            let media_type = item
                .get("media_type")
                .and_then(Value::as_str)
                .unwrap_or("application/octet-stream")
                .to_string();
            let preview_url = item
                .get("preview_url")
                .and_then(Value::as_str)
                .map(ToString::to_string);
            let data_b64 = item.get("data_b64").and_then(Value::as_str)?;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(data_b64)
                .ok()?;

            Some(MediaAttachment {
                filename,
                media_type,
                bytes,
                preview_url,
            })
        })
        .collect()
}

#[cfg_attr(
    not(target_arch = "wasm32"),
    expect(
        dead_code,
        reason = "only the wasm build opens the browser file picker"
    )
)]
pub(super) fn open_media_picker(on_add_media: EventHandler<Vec<MediaAttachment>>) {
    spawn(async move {
        let js = r#"
            (function() {
                var input = document.createElement('input');
                input.type = 'file';
                input.accept = 'image/*,audio/*,video/*,.pdf,.txt,.doc,.docx,.xlsx,.csv';
                input.multiple = true;
                input.onchange = function() {
                    var files = Array.from(input.files || []);
                    if (files.length === 0) { dioxus.send([]); return; }
                    Promise.all(files.map(function(file) {
                        return new Promise(function(resolve) {
                            var reader = new FileReader();
                            reader.onload = function(e) {
                                var result = typeof e.target.result === 'string' ? e.target.result : '';
                                resolve({
                                    filename: file.name,
                                    media_type: file.type || 'application/octet-stream',
                                    data_b64: result.split(',')[1] || '',
                                    preview_url: file.type.startsWith('image/') ? result : null
                                });
                            };
                            reader.onerror = function() { resolve(null); };
                            reader.readAsDataURL(file);
                        });
                    })).then(function(results) {
                        dioxus.send(results.filter(Boolean));
                    });
                };
                document.body.appendChild(input);
                input.click();
                document.body.removeChild(input);
            })();
        "#;

        let mut eval = dioxus::document::eval(js);
        if let Ok(value) = eval.recv::<Value>().await {
            let attachments = parse_media_attachments(value);
            if !attachments.is_empty() {
                on_add_media.call(attachments);
            }
        }
    });
}

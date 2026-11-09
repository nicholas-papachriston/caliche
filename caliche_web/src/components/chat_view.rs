//! Chat view component - monospace terminal style.

use dioxus::prelude::*;
use futures_util::StreamExt;

use super::chat_input::ChatInput;
use super::file_browser::FileBrowser;
use super::message_list::MessageList;
use crate::api::types::{MediaAttachment, MessageMediaItem, StreamingEventType};
use crate::api::{ApiClient, FileEntry};
use crate::context::{ContextReference, parse_context_references};
use crate::state::{ChatState, ConfigState, Message, ToolCall};

/// A file attached to a message with its content
#[derive(Debug, Clone, PartialEq)]
pub struct AttachedFileContext {
    pub file: FileEntry,
    pub content: Option<String>,
}

/// Props for the ChatView component
#[derive(Props, Clone, PartialEq)]
pub struct ChatViewProps {
    /// Chat state signal
    pub chat: Signal<ChatState>,
    /// Config state signal
    pub config: Signal<ConfigState>,
    /// API client for file operations
    pub api_client: ApiClient,
    /// Whether file browsing is enabled
    pub file_browsing_enabled: bool,
    /// True when a turn is active and new sends must be blocked.
    pub turn_locked: bool,
    /// Callback when a message is sent (includes file context)
    pub on_send: EventHandler<(String, Vec<AttachedFileContext>)>,
    /// Callback when stop is clicked
    pub on_stop: EventHandler<()>,
}

/// Main chat view component
#[component]
pub fn ChatView(props: ChatViewProps) -> Element {
    let chat = props.chat.read();
    let is_streaming = chat.is_streaming;
    let is_input_locked = is_streaming || props.turn_locked;
    let messages = chat.messages.clone();
    let total_tokens = chat.total_tokens.total;
    let mode = chat.mode;
    let show_sources = props.config.read().show_sources;

    // State for file browser, attachments, context references, and media
    let mut show_file_browser = use_signal(|| false);
    let mut attached_files = use_signal(|| Vec::<AttachedFileContext>::new());
    let mut context_refs = use_signal(|| Vec::<ContextReference>::new());
    let mut media_attachments = use_signal(|| Vec::<MediaAttachment>::new());

    let api_client = props.api_client.clone();
    let file_browsing_enabled = props.file_browsing_enabled;

    let handle_clear = {
        let mut chat = props.chat;
        move |_| {
            chat.write().clear();
            attached_files.write().clear();
            context_refs.write().clear();
            media_attachments.write().clear();
        }
    };

    // Handle file selection from browser
    let handle_file_select = {
        let api = api_client.clone();
        move |file: FileEntry| {
            let api = api.clone();
            let file_clone = file.clone();

            // Load file content
            spawn(async move {
                let content = match api.read_file(&file_clone.path, Some(500)).await {
                    Ok(response) => Some(response.content),
                    Err(_) => None,
                };

                attached_files.write().push(AttachedFileContext {
                    file: file_clone,
                    content,
                });
            });
        }
    };

    // Handle removing an attachment
    let handle_remove_attachment = move |path: String| {
        attached_files.write().retain(|f| f.file.path != path);
    };

    // Handle removing a context reference
    let handle_remove_context = move |idx: usize| {
        let mut refs = context_refs.write();
        if idx < refs.len() {
            refs.remove(idx);
        }
    };

    let handle_add_media = move |attachments: Vec<MediaAttachment>| {
        if attachments.is_empty() {
            return;
        }
        media_attachments.write().extend(attachments);
    };

    // Handle removing a media attachment by index
    let handle_remove_media = move |idx: usize| {
        let mut media = media_attachments.write();
        if idx < media.len() {
            media.remove(idx);
        }
    };

    // Handle send with file context, context references, and optional media
    let handle_send = {
        let on_send = props.on_send.clone();
        let api = api_client.clone();
        let mut chat = props.chat;
        let config = props.config;
        let turn_locked = props.turn_locked;
        move |message: String| {
            if chat.read().is_streaming || turn_locked {
                return;
            }
            let media = media_attachments.read().clone();

            // Parse any @ references from the message
            let parsed = parse_context_references(&message);
            let mut all_refs = context_refs.read().clone();
            all_refs.extend(parsed.references);
            let cleaned_message = parsed.message.clone();

            if media.is_empty() {
                // No media — resolve @ references against the backend, prepend
                // the expanded block to the outgoing text, and delegate to the
                // parent's SSE / WebSocket streaming path.
                let files = attached_files.read().clone();
                let api_for_refs = api.clone();
                let chat_for_banner = chat;
                let session_id = chat_for_banner.read().current_session_id.clone();
                let on_send = on_send.clone();
                attached_files.write().clear();
                context_refs.write().clear();
                spawn(async move {
                    let resolved = if all_refs.is_empty() {
                        crate::context::ResolvedContext::default()
                    } else {
                        crate::context::resolve_context(
                            &all_refs,
                            &api_for_refs,
                            session_id.as_deref(),
                        )
                        .await
                    };
                    let final_message = if resolved.block.is_empty() {
                        cleaned_message
                    } else {
                        format!("{}\n\n{}", resolved.block, cleaned_message)
                    };
                    on_send.call((final_message, files));
                });
                return;
            }
            // For the multipart/media path we still pass the raw references as
            // a compact banner, since binary attachments take precedence over
            // pulled-in text context.
            let final_message = if all_refs.is_empty() {
                cleaned_message
            } else {
                let ref_info: Vec<String> = all_refs.iter().map(|r| r.display_name()).collect();
                format!("[Context: {}]\n\n{}", ref_info.join(", "), cleaned_message)
            };

            // Has media — stream multipart directly from this component
            let media_items: Vec<MessageMediaItem> = media
                .iter()
                .map(|m| MessageMediaItem {
                    filename: m.filename.clone(),
                    media_type: m.media_type.clone(),
                    preview_url: m.preview_url.clone(),
                })
                .collect();
            let user_msg = Message::user_with_media(&final_message, media_items);
            chat.write().add_message(user_msg);

            let assistant_msg = Message::streaming_assistant();
            let msg_id = assistant_msg.message_id.clone();
            chat.write().add_message(assistant_msg);
            chat.write().is_streaming = true;
            chat.write().streaming_message_id = Some(msg_id.clone());

            attached_files.write().clear();
            context_refs.write().clear();
            media_attachments.write().clear();

            let provider = config.read().provider.clone();
            let model_name = config.read().model.clone();
            let api_clone = api.clone();

            spawn(async move {
                match api_clone
                    .chat_multipart_stream(
                        &final_message,
                        media.iter().collect(),
                        provider.as_deref(),
                        model_name.as_deref(),
                    )
                    .await
                {
                    Ok(stream) => {
                        let mut stream = Box::pin(stream);
                        while let Some(evt_result) = stream.next().await {
                            match evt_result {
                                Ok(event) => match event.event_type {
                                    StreamingEventType::TextChunk
                                    | StreamingEventType::ThinkingChunk
                                    | StreamingEventType::Content => {
                                        if let Some(content) = event.content {
                                            chat.write()
                                                .update_streaming_content(&msg_id, &content);
                                        }
                                    }
                                    StreamingEventType::ToolCallStart => {
                                        if let Some(name) = event.name {
                                            let tool_id = event
                                                .action_id
                                                .or(event.id)
                                                .unwrap_or_else(|| name.clone());
                                            chat.write().add_tool_call(
                                                &msg_id,
                                                ToolCall::new(tool_id, name),
                                            );
                                        }
                                    }
                                    StreamingEventType::ToolCallResult => {
                                        if let Some(name) = event.name {
                                            let result = event.result.unwrap_or_default();
                                            let tool_key =
                                                event.action_id.or(event.id).unwrap_or(name);
                                            chat.write().complete_tool_call(
                                                &msg_id, &tool_key, result, true,
                                            );
                                        }
                                    }
                                    StreamingEventType::Done => {
                                        chat.write().finalize_message(&msg_id, event.usage);
                                        break;
                                    }
                                    StreamingEventType::Usage
                                    | StreamingEventType::Metadata
                                    | StreamingEventType::LimitReached => {
                                        if let Some(usage) = event.usage {
                                            chat.write().finalize_message(&msg_id, Some(usage));
                                        }
                                    }
                                    StreamingEventType::Error => {
                                        let err = event
                                            .error
                                            .unwrap_or_else(|| "Streaming error".to_string());
                                        chat.write().mark_error(&msg_id, &err);
                                        break;
                                    }
                                    _ => {}
                                },
                                Err(e) => {
                                    chat.write().mark_error(&msg_id, &e.to_string());
                                    break;
                                }
                            }
                        }
                        // Ensure streaming state is cleaned up
                        if chat.read().is_streaming {
                            chat.write().finalize_message(&msg_id, None);
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Multipart streaming unavailable, falling back to non-streaming multipart: {}",
                            e
                        );
                        match api_clone
                            .chat_multipart(
                                &final_message,
                                media.iter().collect(),
                                provider.as_deref(),
                                model_name.as_deref(),
                            )
                            .await
                        {
                            Ok(response) => {
                                chat.write()
                                    .update_streaming_content(&msg_id, &response.response);
                                chat.write().finalize_message(&msg_id, Some(response.usage));
                            }
                            Err(fallback_err) => {
                                chat.write().mark_error(&msg_id, &fallback_err.to_string());
                            }
                        }
                    }
                }
            });
        }
    };

    // Toggle file browser
    let toggle_browser = move |_| {
        let current = *show_file_browser.read();
        show_file_browser.set(!current);
    };

    let attachments = attached_files.read().clone();
    let refs = context_refs.read().clone();
    let media = media_attachments.read().clone();
    let show_browser = *show_file_browser.read();

    rsx! {
        div { class: "chat-view",
            style: "display: flex; flex-direction: column; height: 100%; width: 100%; padding: 1ch; gap: 1ch; overflow: hidden;",

            // File browser panel (when open)
            if show_browser && file_browsing_enabled {
                div { style: "flex: 0 0 40%; min-height: 200px; max-height: 50%;",
                    FileBrowser {
                        api_client: api_client.clone(),
                        on_file_select: handle_file_select,
                        on_close: move |_| show_file_browser.set(false),
                    }
                }
            }

            // Message list
            MessageList {
                messages: messages,
                mode: mode,
                show_sources: show_sources,
            }

            // Chat input with file attachment, context reference, and media support
            ChatInput {
                on_send: handle_send,
                on_stop: props.on_stop,
                on_clear: handle_clear,
                is_streaming: is_input_locked,
                total_tokens: total_tokens,
                attached_files: attachments,
                context_refs: refs,
                on_remove_attachment: handle_remove_attachment,
                on_remove_context: handle_remove_context,
                on_toggle_files: toggle_browser,
                file_browsing_enabled: file_browsing_enabled,
                media_attachments: media,
                on_add_media: handle_add_media,
                on_remove_media: handle_remove_media,
            }
        }
    }
}

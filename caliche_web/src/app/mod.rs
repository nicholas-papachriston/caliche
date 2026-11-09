//! Main application component.

mod connection;
pub(crate) mod settings_merge;
mod streaming;

pub use connection::{AppState, resolve_effective_initial_config, spawn_connection_loop};
#[cfg(target_arch = "wasm32")]
pub(crate) use connection::{spawn_ws_connection_loop, spawn_ws_url_watch};

use dioxus::prelude::*;

use crate::api::ApiClient;
use crate::api::CreateSessionRequest;
use crate::api::LiveSessionInfo;
use crate::api::ReasoningPatternInfo;
#[cfg(target_arch = "wasm32")]
use crate::api::{WsClient, WsClientMessage};
use crate::app::settings_merge::{settings_from_config, should_put_live_settings};
use crate::commands::{self, Command, execute_command};
use crate::components::SpawnRequest;
pub use crate::components::app_shell::ActiveTab;
use crate::components::app_shell::AppShell;
use crate::components::chat_view::AttachedFileContext;
use crate::components::chrome::session_display_title;
use crate::components::fleet_view::fleet_tab_label;
use crate::components::history_view::OpenedSession;
use crate::components::nav::InspectPaneFlags;
use crate::hooks::use_loaded_once;
use crate::state::agents::{AgentSource, AgentStatus, SubAgent};
use crate::state::streaming::{EpochCounter, StreamTicket};
use crate::state::{ChatState, Message, SavedConversation, TokenTotals};

/// Format file context as a string to prepend to the message
fn format_file_context(files: &[AttachedFileContext]) -> String {
    if files.is_empty() {
        return String::new();
    }

    let mut context = String::from("I'm providing the following files as context:\n\n");

    for file_ctx in files {
        context.push_str(&format!("--- {} ---\n", file_ctx.file.path));
        if let Some(ref content) = file_ctx.content {
            context.push_str(content);
            context.push_str("\n");
        } else {
            context.push_str("[File content not available]\n");
        }
        context.push_str("---\n\n");
    }

    context
}

async fn ensure_chat_session(client: &ApiClient, chat: &mut Signal<ChatState>) {
    if chat.read().has_session() {
        return;
    }
    let response = match client
        .create_session(&CreateSessionRequest::default())
        .await
    {
        Ok(response) => response,
        Err(err) => {
            tracing::warn!(error = %err, "failed to create session");
            return;
        }
    };
    let mode = chat.read().mode;
    let session_id = response.session_id;
    chat.write().set_session(session_id.clone());
    if let Err(err) = client.set_mode(&session_id, mode).await {
        tracing::warn!(
            error = %err,
            session_id = %session_id,
            "setMode after create failed"
        );
    }
}

fn persist_live_settings(client: ApiClient, config: &crate::state::ConfigState) {
    let settings = settings_from_config(config);
    // An empty provider is not a clear. The settings handler returns 400.
    if !should_put_live_settings(&settings.provider, &settings.model) {
        return;
    }
    spawn(async move {
        if let Err(err) = client.put_settings(&settings).await {
            tracing::warn!(error = %err, "failed to apply settings");
        }
    });
}

/// Browser tab title. Must match `Dioxus.toml` `[web.app] title`.
///
/// `index.html` keeps the `{app_title}` placeholder. dx 0.7.6 replaces that
/// token, and otherwise inserts `[web.app] title` immediately before
/// `</title>`. A literal `<title>Caliche</title>` plus that insert is
/// served as one tag whose text is `CalicheCaliche`.
pub const DOCUMENT_TITLE: &str = "Caliche";

/// Root application component
#[component]
pub fn App() -> Element {
    // Global state signals
    // Load any previously-persisted history/config so the tab looks the same
    // between reloads. Failures are logged and fall through to defaults.
    let history_persist = crate::state::default_history_store();
    let config_persist = crate::state::default_config_store();
    let restored_history = match history_persist.load() {
        Ok(h) => SavedConversation::normalize_list(h),
        Err(err) => {
            tracing::warn!(error = %err, "failed to load saved history");
            Vec::new()
        }
    };
    let restored_config = match config_persist.load() {
        Ok(c) => c,
        Err(err) => {
            tracing::warn!(error = %err, "failed to load saved config");
            None
        }
    };
    let effective_initial_config = resolve_effective_initial_config(restored_config);

    let chat_state = use_signal(ChatState::default);
    let mut config_state = use_signal(|| effective_initial_config);
    let mut app_state = use_signal(AppState::default);
    let mut active_tab = use_signal(|| ActiveTab::Chat);
    let mut history_store = use_signal(|| restored_history);
    let mut agents_store = use_signal(Vec::<SubAgent>::new);
    let mut live_sessions = use_signal(Vec::<LiveSessionInfo>::new);
    let mut live_sessions_error: Signal<Option<String>> = use_signal(|| None);
    let mut live_sessions_loading = use_signal(|| false);
    let mut live_sessions_reload = use_signal(|| 0_u32);

    // Persist history / config whenever they change. We re-use the loader
    // boxes above; each write takes the current snapshot and hands it to the
    // platform store (localStorage on WASM, config file on native).
    let history_persist_for_save = std::sync::Arc::new(history_persist);
    let config_persist_for_save = std::sync::Arc::new(config_persist);
    {
        let store = history_persist_for_save.clone();
        use_effect(move || {
            let snap = history_store.read().clone();
            if let Err(err) = store.save(&snap) {
                tracing::warn!(error = %err, "failed to save history");
            }
        });
    }
    {
        let store = config_persist_for_save.clone();
        use_effect(move || {
            let snap = config_state.read().clone();
            if let Err(err) = store.save(&snap) {
                tracing::warn!(error = %err, "failed to save config");
            }
        });
    }
    {
        let app = app_state;
        use_effect(move || {
            let connected = app.read().is_connected;
            let _reload = *live_sessions_reload.read();
            if !connected {
                live_sessions_loading.set(false);
                return;
            }
            let client = app.read().api_client.clone();
            if live_sessions.peek().is_empty() {
                live_sessions_loading.set(true);
            }
            spawn(async move {
                match client.list_live_sessions().await {
                    Ok(list) => {
                        live_sessions.set(list);
                        live_sessions_error.set(None);
                    }
                    Err(err) => {
                        live_sessions_error.set(Some(err.to_string()));
                    }
                }
                live_sessions_loading.set(false);
            });
        });
    }

    // Shared endpoint list — written by FleetView, read here for the tab label.
    let endpoints_store = use_signal(Vec::<crate::api::EndpointInfo>::new);
    // Available reasoning patterns — loaded once on connect.
    let mut reasoning_patterns: Signal<Vec<ReasoningPatternInfo>> = use_signal(Vec::new);
    let mut reasoning_patterns_loaded = use_signal(|| false);
    let mut inspect_panes = use_signal(|| InspectPaneFlags::ALL);
    let mut inspect_probed = use_signal(|| false);
    // Whether the session inspector sidebar is visible.
    let mut show_session_panel = use_signal(|| false);
    // Tracks the previous streaming state to detect the true→false transition.
    let mut prev_streaming = use_signal(|| false);
    // Strict turn gate: while true, reject/disable new sends until current turn ends.
    let mut turn_in_flight = use_signal(|| false);

    // Cancellation signal for stopping generation
    let cancel_signal = use_signal(|| false);
    // Monotonically-increasing stream epoch. Every time we start a new stream
    // we capture a ticket; iteration stops as soon as it is superseded.
    // This prevents late SSE/WS chunks from writing into a stale assistant
    // message (for example after the user switched sessions or hit Cancel).
    let stream_epoch: Signal<EpochCounter> = use_signal(EpochCounter::default);
    let mut connection_loop_started = use_signal(|| false);

    #[cfg(target_arch = "wasm32")]
    let ws_client_ref = use_signal(|| None::<WsClient>);
    #[cfg(target_arch = "wasm32")]
    let stream_gate = use_signal(|| None::<streaming::TurnGate>);
    #[cfg(target_arch = "wasm32")]
    let mut ws_loop_started = use_signal(|| false);

    spawn_connection_loop(config_state, app_state, connection_loop_started);
    #[cfg(target_arch = "wasm32")]
    spawn_ws_url_watch(config_state, ws_client_ref, stream_gate);
    #[cfg(target_arch = "wasm32")]
    spawn_ws_connection_loop(app_state, ws_client_ref, stream_gate, ws_loop_started);

    // Probe feature-gated inspect RPCs once per connection. Method-not-found
    // hides that pane. A transport error leaves the panes visible.
    use_effect(move || {
        let connected = app_state.read().is_connected;
        if !connected {
            if *inspect_probed.read() {
                inspect_probed.set(false);
            }
            return;
        }
        if *inspect_probed.read() {
            return;
        }
        inspect_probed.set(true);
        let api = app_state.read().api_client.clone();
        spawn(async move {
            let documents = api.rpc_method_available("documents.listCollections").await;
            let chains = api.rpc_method_available("chains.llm").await;
            let reasoning = api.rpc_method_available("reasoning.listPatterns").await;
            match (documents, chains, reasoning) {
                (Ok(documents), Ok(chains), Ok(reasoning)) => {
                    inspect_panes.set(InspectPaneFlags {
                        documents,
                        chains,
                        reasoning,
                    });
                }
                (documents, chains, reasoning) => {
                    tracing::warn!(
                        documents = ?documents.as_ref().err().map(ToString::to_string),
                        chains = ?chains.as_ref().err().map(ToString::to_string),
                        reasoning = ?reasoning.as_ref().err().map(ToString::to_string),
                        "inspect pane probe failed"
                    );
                }
            }
        });
    });

    // Fetch available reasoning patterns once when we first connect.
    use_loaded_once(
        move || app_state.read().is_connected && !*reasoning_patterns_loaded.read(),
        reasoning_patterns_loaded,
        move || {
            let api = app_state.read().api_client.clone();
            spawn(async move {
                if let Ok(patterns) = api.get_reasoning_patterns().await {
                    reasoning_patterns.set(patterns);
                    reasoning_patterns_loaded.set(true);
                }
            });
        },
    );

    // Auto-save to history whenever a streaming response completes (true → false transition).
    // All signal mutations are deferred via spawn to avoid writing signals while Dioxus is
    // still inside its diff/render loop (which would cause AlreadyBorrowedMut panics).
    use_effect(move || {
        let is_streaming = chat_state.read().is_streaming;
        let was_streaming = *prev_streaming.read();

        // Snapshot the data we need before the spawn (no borrows cross the async boundary).
        let save_data = if was_streaming && !is_streaming {
            let chat = chat_state.read();
            let messages = chat.messages.clone();
            let session_id = chat.current_session_id.clone();
            let total_tokens = chat.total_tokens.total;
            let has_user = messages
                .iter()
                .any(|m| m.role == crate::api::types::MessageRole::User);
            if has_user {
                Some((messages, session_id, total_tokens))
            } else {
                None
            }
        } else {
            None
        };

        let streaming_changed = is_streaming != was_streaming;

        spawn(async move {
            if let Some((messages, session_id, total_tokens)) = save_data {
                let conv = SavedConversation::from_chat(messages, session_id.clone(), total_tokens);
                let mut store = history_store.write();
                SavedConversation::upsert(&mut store, conv);
            }
            if streaming_changed {
                prev_streaming.set(is_streaming);
            }
            // Release strict turn lock when streaming fully settles.
            if !is_streaming && *turn_in_flight.read() {
                turn_in_flight.set(false);
            }
        });
    });

    // Handle sending a message (with optional file context)
    let handle_send = {
        let chat_state = chat_state;
        let config_state = config_state;
        let app_state = app_state;
        let cancel_signal = cancel_signal;
        let stream_epoch = stream_epoch;
        let mut turn_in_flight = turn_in_flight;
        #[cfg(target_arch = "wasm32")]
        let ws_client_ref = ws_client_ref;
        #[cfg(target_arch = "wasm32")]
        let stream_gate = stream_gate;

        move |(content, files): (String, Vec<AttachedFileContext>)| {
            let mut chat = chat_state;
            let mut config = config_state;
            let app = app_state;
            let mut cancel = cancel_signal;
            let mut stream_epoch = stream_epoch;

            // Prevent overlapping turns (strict one-response-per-input).
            if chat.read().is_streaming || *turn_in_flight.read() {
                return;
            }
            turn_in_flight.set(true);

            // Begin a new stream epoch. Any still-running stream iteration
            // will see its ticket superseded and exit before writing into the
            // new message. Must happen before we clear `cancel` so any in-flight
            // loop still treats the pre-existing cancel as authoritative.
            stream_epoch.write().bump();
            let stream_ticket: StreamTicket = stream_epoch.read().capture();

            spawn(async move {
                // Reset cancel signal
                cancel.set(false);

                // Double-check we're not already streaming/locked (race condition guard)
                if chat.read().is_streaming {
                    turn_in_flight.set(false);
                    return;
                }

                // Check if this is a slash command
                if commands::is_command(&content) {
                    // Show the command as a user message
                    chat.write().add_message(Message::user(&content));

                    // Parse and execute the command
                    match commands::parse_command(&content) {
                        Ok(cmd) => {
                            // Get data needed for command execution
                            let client = app.read().api_client.clone();
                            let session_id = chat.read().current_session_id.clone();
                            let providers = app.read().providers.clone();
                            let models = app.read().all_models.clone();
                            let config_snapshot = config.read().clone();

                            // Handle special commands that modify local state
                            let is_clear = matches!(cmd, Command::Clear);
                            let is_switch_provider = matches!(cmd, Command::SwitchProvider(_));
                            let is_set_model = matches!(cmd, Command::SetModel(_));
                            let is_set_mode = matches!(cmd, Command::SetMode(_));
                            let switch_tab = match &cmd {
                                Command::ShowAgents => Some(ActiveTab::Agents),
                                Command::ShowSettings => Some(ActiveTab::Settings),
                                Command::ShowHistory | Command::ResumeHelp => {
                                    Some(ActiveTab::History)
                                }
                                Command::ShowChat => Some(ActiveTab::Chat),
                                Command::ShowDocuments => Some(ActiveTab::Documents),
                                Command::ShowPlugins => Some(ActiveTab::Plugins),
                                Command::ShowReasoning => Some(ActiveTab::Reasoning),
                                _ => None,
                            };
                            let spawn_task = if let Command::SpawnAgent(ref t) = cmd {
                                Some(t.clone())
                            } else {
                                None
                            };

                            // Extract values before executing
                            let new_provider = if let Command::SwitchProvider(ref p) = cmd {
                                Some(p.clone())
                            } else {
                                None
                            };
                            let new_model = if let Command::SetModel(ref m) = cmd {
                                Some(m.clone())
                            } else {
                                None
                            };
                            let new_mode = if let Command::SetMode(m) = cmd {
                                Some(m)
                            } else {
                                None
                            };

                            // Execute the command
                            let result = execute_command(
                                cmd,
                                &client,
                                session_id.as_deref(),
                                &providers,
                                &models,
                                &config_snapshot,
                            )
                            .await;

                            // Handle state updates for successful commands
                            if result.success {
                                if is_clear {
                                    // Clear conversation but keep session
                                    if let Some(sid) = session_id.as_deref() {
                                        let _ = client.clear_history(sid).await;
                                    }
                                    chat.write().clear_conversation();
                                }

                                if is_switch_provider {
                                    if let Some(p) = new_provider {
                                        config.write().provider = Some(p);
                                        config.write().model = None;
                                        persist_live_settings(client.clone(), &config.read());
                                    }
                                }

                                if is_set_model {
                                    if let Some(m) = new_model {
                                        config.write().model = Some(m);
                                        persist_live_settings(client.clone(), &config.read());
                                    }
                                }

                                if is_set_mode {
                                    if let Some(m) = new_mode {
                                        chat.write().set_mode(m);
                                    }
                                }

                                if let Some(task) = spawn_task {
                                    let agent = SubAgent::new(
                                        format!("agent-{}", agents_store.read().len() + 1),
                                        &task,
                                        AgentSource::Local,
                                    );
                                    agents_store.write().insert(0, agent);
                                    active_tab.set(ActiveTab::Agents);
                                } else if let Some(tab) = switch_tab {
                                    active_tab.set(tab);
                                }
                            }

                            // Show the result
                            chat.write().add_system_message(&result.message);
                        }
                        Err(e) => {
                            chat.write().add_system_message(format!("**Error:** {}", e));
                        }
                    }
                    turn_in_flight.set(false);
                    return;
                }

                // Regular message handling below...

                // Format file context and combine with user message
                let file_context = format_file_context(&files);
                let full_message = if file_context.is_empty() {
                    content.clone()
                } else {
                    format!("{}{}", file_context, content)
                };

                // Display message to user (without full file content for cleaner display)
                let display_message = if files.is_empty() {
                    content.clone()
                } else {
                    let file_names: Vec<&str> =
                        files.iter().map(|f| f.file.name.as_str()).collect();
                    format!("[attached: {}]\n\n{}", file_names.join(", "), content)
                };

                // Add user message
                let user_msg = Message::user(&display_message);
                chat.write().add_message(user_msg);

                // Create streaming assistant message
                let assistant_msg = Message::streaming_assistant();
                let message_id = assistant_msg.message_id.clone();
                chat.write().add_message(assistant_msg);
                chat.write().is_streaming = true;
                chat.write().streaming_message_id = Some(message_id.clone());

                let client = app.read().api_client.clone();
                ensure_chat_session(&client, &mut chat).await;

                #[cfg(target_arch = "wasm32")]
                {
                    let base_url = config.read().api_base_url.clone();
                    streaming::send_via_websocket(
                        chat,
                        config,
                        chat_state,
                        config_state,
                        ws_client_ref,
                        base_url,
                        full_message,
                        message_id,
                        cancel,
                        stream_ticket,
                        stream_gate,
                    );
                }

                #[cfg(not(target_arch = "wasm32"))]
                {
                    let request = {
                        let chat_read = chat.read();
                        let config_read = config.read();
                        streaming::build_chat_stream_request(&chat_read, &config_read, full_message)
                    };
                    streaming::consume_sse_stream(
                        &client,
                        request,
                        chat,
                        cancel,
                        stream_ticket,
                        message_id,
                    )
                    .await;
                }
            });
        }
    };

    // Stop the visible turn immediately. The epoch bump drops late chunks.
    // The streaming row is finalized here so a fast or empty cancel still
    // clears the placeholder and unblocks the composer.
    let handle_stop = {
        let mut cancel_signal = cancel_signal;
        let mut stream_epoch = stream_epoch;
        let mut chat = chat_state;
        let app = app_state;
        let mut turn_in_flight = turn_in_flight;
        #[cfg(target_arch = "wasm32")]
        let mut ws_client_ref = ws_client_ref;
        #[cfg(target_arch = "wasm32")]
        let mut stream_gate = stream_gate;
        move |_| {
            cancel_signal.set(true);
            stream_epoch.write().bump();
            chat.write().stop_generation();
            turn_in_flight.set(false);
            #[cfg(target_arch = "wasm32")]
            {
                stream_gate.set(None);
                if let Some(client) = ws_client_ref.write().take() {
                    let _ = client.send(&WsClientMessage::Cancel);
                    client.close();
                }
            }
            let session_id = chat.read().current_session_id.clone();
            let Some(session_id) = session_id else {
                return;
            };
            let client = app.read().api_client.clone();
            let mut chat = chat;
            spawn(async move {
                match client.cancel_session(&session_id).await {
                    Ok(()) => {
                        let mut state = chat.write();
                        if state.current_session_id.as_deref() == Some(session_id.as_str()) {
                            state.current_session_id = None;
                            state.session_created = false;
                        }
                    }
                    Err(err) => {
                        tracing::warn!(
                            error = %err,
                            session_id = %session_id,
                            "failed to cancel session"
                        );
                    }
                }
            });
        }
    };

    // Clone data from app_state to avoid holding borrow during render
    let is_connected = app_state.read().is_connected;
    let providers = app_state.read().providers.clone();
    let all_models = app_state.read().all_models.clone();
    let api_client = app_state.read().api_client.clone();
    let file_browsing_enabled = app_state.read().file_browsing_enabled;
    let current_mode = chat_state.read().mode;

    // Handle mode change from header toggle
    let on_mode_change = {
        let chat = chat_state;
        let app = app_state;
        move |new_mode: crate::api::types::InteractionMode| {
            let mut chat = chat;
            spawn(async move {
                let client = app.read().api_client.clone();
                let session_id = chat.read().current_session_id.clone();
                if let Some(sid) = session_id.as_deref() {
                    let _ = client.set_mode(sid, new_mode).await;
                }
                chat.write().set_mode(new_mode);
                chat.write().add_system_message(format!(
                    "**Mode: {}** — {}",
                    new_mode,
                    new_mode.description()
                ));
            });
        }
    };

    let on_settings_changed = {
        let client = api_client.clone();
        let config_ref = config_state;
        Some(EventHandler::new(move |_| {
            persist_live_settings(client.clone(), &config_ref.read());
        }))
    };

    // Open a server session in chat. A local transcript is restored when this
    // browser already saved that session id; otherwise the title is shown and
    // the chat adopts the server session. Spawn avoids re-entrant signal borrows.
    let on_resume = {
        let history = history_store;
        let mut chat = chat_state;
        let mut tab = active_tab;
        let mut stream_epoch = stream_epoch;
        move |opened: OpenedSession| {
            let cached = history
                .read()
                .iter()
                .find(|entry| entry.session_id.as_deref() == Some(opened.session_id.as_str()))
                .cloned();
            spawn(async move {
                // Bump the stream epoch so any in-flight stream loop from
                // the previous session stops writing chunks into the now-
                // restored message list.
                stream_epoch.write().bump();
                {
                    let mut c = chat.write();
                    c.is_streaming = false;
                    c.streaming_message_id = None;
                    if let Some(saved) = cached {
                        c.messages = saved.messages;
                        c.total_tokens = TokenTotals {
                            total: saved.total_tokens,
                            ..TokenTotals::default()
                        };
                    } else {
                        let title = session_display_title(opened.title.as_deref());
                        c.clear();
                        c.add_system_message(format!("opened session: {title}"));
                    }
                    c.set_session(opened.session_id);
                }
                tab.set(ActiveTab::Chat);
            });
        }
    };

    let on_reload_sessions = {
        move |_| {
            let next = *live_sessions_reload.read() + 1;
            live_sessions_reload.set(next);
        }
    };

    // Spawn a new sub-agent (from the agents tab launch form).
    // For `AgentSource::Local` or `Remote`, we start a backend coding run via
    // `spawn_coding_run` so the agent has an independent identity and can be
    // cancelled server-side. The pending UI state is inserted synchronously so
    // the operator sees the card immediately; the backend call then attaches
    // the session ID and the first streamed response once it completes.
    let on_agent_spawn = {
        let mut agents = agents_store;
        let app = app_state;
        move |req: SpawnRequest| {
            let SpawnRequest {
                task,
                source,
                provider,
                model,
            } = req;
            let mut agent = SubAgent::new(
                format!("agent-{}", agents.read().len() + 1),
                task.clone(),
                source,
            );
            agent.set_provider_and_model(provider.clone(), model.clone());
            let agent_id = agent.id.clone();
            agents.write().insert(0, agent);

            let client = app.read().api_client.clone();
            let mut agents_w = agents;
            spawn(async move {
                match client.spawn_coding_run(&task, provider, model).await {
                    Ok(resp) => {
                        let mut store = agents_w.write();
                        if let Some(a) = store.iter_mut().find(|a| a.id == agent_id) {
                            a.attach_backend_session(resp.session_id);
                            a.status = AgentStatus::Running;
                            if !resp.response.is_empty() {
                                a.push_output(resp.response);
                            }
                            a.status = AgentStatus::Completed;
                        }
                    }
                    Err(err) => {
                        let mut store = agents_w.write();
                        if let Some(a) = store.iter_mut().find(|a| a.id == agent_id) {
                            a.status = AgentStatus::Failed;
                            a.push_output(format!("Spawn failed: {err}"));
                        }
                    }
                }
            });
        }
    };

    // Cancel a running agent by ID. The card updates immediately, then
    // POST /api/agent/sessions/{id}/cancel signals the turn and removes it.
    let on_agent_cancel = {
        let mut agents = agents_store;
        let app = app_state;
        move |id: String| {
            let backend_session = {
                let mut store = agents.write();
                let found = store.iter_mut().find(|a| a.id == id);
                if let Some(agent) = found {
                    if agent.status.is_active() {
                        agent.status = AgentStatus::Cancelled;
                    }
                    agent.backend_session_id.clone()
                } else {
                    None
                }
            };

            if let Some(session_id) = backend_session {
                let client = app.read().api_client.clone();
                spawn(async move {
                    if let Err(err) = client.cancel_session(&session_id).await {
                        tracing::warn!(
                            error = %err,
                            session_id = %session_id,
                            "Failed to cancel sub-agent session on backend"
                        );
                    }
                });
            }
        }
    };

    // Open an agent's output in the chat view (switches tab, loads output as messages).
    let on_agent_open_output = {
        let agents = agents_store;
        let mut chat = chat_state;
        let mut tab = active_tab;
        move |id: String| {
            let store = agents.read();
            if let Some(agent) = store.iter().find(|a| a.id == id) {
                let task = agent.task.clone();
                let output = agent.full_output();
                let agent_label = format!("[agent: {}]", agent.name);
                spawn(async move {
                    let mut c = chat.write();
                    c.messages.push(Message::user(&format!(
                        "{}\n\n**Task:** {}",
                        agent_label, task
                    )));
                    if !output.is_empty() {
                        c.messages.push(Message::system(&output));
                    }
                    tab.set(ActiveTab::Chat);
                });
            }
        }
    };

    // Switch the active API URL when user clicks "connect" on an endpoint.
    // Updating api_base_url is enough — the connection loop detects the change
    // on its next iteration and re-creates the ApiClient for the new target.
    let on_connect_endpoint = {
        let mut cfg = config_state;
        let mut tab = active_tab;
        move |url: String| {
            cfg.write().api_base_url = url;
            // Switch to chat immediately so the user can start talking to the
            // newly-connected endpoint without an extra click.
            tab.set(ActiveTab::Chat);
        }
    };

    let fleet_label = {
        let eps = endpoints_store.read();
        fleet_tab_label(&eps, &agents_store.read())
    };

    rsx! {
        document::Title { "{DOCUMENT_TITLE}" }
        AppShell {
            is_connected,
            config: config_state,
            chat: chat_state,
            active_tab,
            live_sessions,
            live_sessions_error,
            live_sessions_loading,
            agents: agents_store,
            endpoints: endpoints_store,
            show_session_panel,
            turn_in_flight,
            stream_epoch,
            providers: providers.clone(),
            all_models: all_models.clone(),
            api_client: api_client.clone(),
            file_browsing_enabled,
            current_mode,
            fleet_label,
            inspect_panes: *inspect_panes.read(),
            reasoning_patterns: reasoning_patterns.read().clone(),
            on_mode_change,
            on_settings_changed,
            on_send: handle_send,
            on_stop: handle_stop,
            on_resume,
            on_reload_sessions,
            on_spawn_task: on_agent_spawn,
            on_cancel_task: on_agent_cancel,
            on_open_task_output: on_agent_open_output,
            on_connect_endpoint,
        }
    }
}

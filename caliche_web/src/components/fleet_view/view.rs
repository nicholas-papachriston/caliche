use std::collections::{HashMap, HashSet};

use dioxus::prelude::*;

use super::broadcast::BroadcastPanel;
use super::broadcast_run::{
    dispatch_selected_endpoints, dispatch_selected_harnesses, inject_selected_tasks,
};
use super::endpoints_section::EndpointsSection;
use super::harnesses_section::HarnessesSection;
use super::helpers::default_provider_and_model;
use super::stats_bar::FleetStatsBar;
use super::tasks_section::TasksSection;
use super::types::{BroadcastResult, ChatMsg, FleetViewProps};
use crate::api::{ApiClient, HarnessInfo};

#[component]
pub fn FleetView(mut props: FleetViewProps) -> Element {
    let initial_providers = props.providers.clone();
    let initial_models = props.all_models.clone();
    let initial_current_provider = props.current_provider.clone();
    let initial_current_model = props.current_model.clone();
    let default_selection = default_provider_and_model(
        &initial_providers,
        &initial_models,
        initial_current_provider.as_deref(),
        initial_current_model.as_deref(),
    );
    let initial_form_provider = default_selection.0.clone();
    let initial_form_model = default_selection.1.clone();
    let initial_spawn_provider = default_selection.0;
    let initial_spawn_model = default_selection.1;

    let mut endpoints = props.endpoints;
    let provider_options = use_signal(move || initial_providers.clone());
    let model_options = use_signal(move || initial_models.clone());
    let current_provider_setting = use_signal(move || initial_current_provider.clone());
    let current_model_setting = use_signal(move || initial_current_model.clone());
    let mut ep_error = use_signal(|| Option::<String>::None);
    let mut poll_started = use_signal(|| false);
    let mut ep_expanded = use_signal(|| Option::<String>::None);
    let mut ep_op_busy = use_signal(|| Option::<String>::None);
    let mut show_launch = use_signal(|| false);

    let mut form_name = use_signal(String::new);
    let mut form_workspace = use_signal(String::new);
    let mut form_port = use_signal(|| "3001".to_string());
    let mut form_provider = use_signal(move || initial_form_provider.clone());
    let mut form_model = use_signal(move || initial_form_model.clone());
    let mut form_auto_start = use_signal(|| true);
    let mut form_error = use_signal(|| Option::<String>::None);
    let mut form_busy = use_signal(|| false);

    let mut ep_chat = use_signal(HashMap::<String, Vec<ChatMsg>>::new);
    let mut ep_chat_inp = use_signal(HashMap::<String, String>::new);
    let mut ep_chat_busy = use_signal(HashSet::<String>::new);

    let mut task_expanded = use_signal(|| Option::<String>::None);
    let mut task_inject = use_signal(HashMap::<String, String>::new);
    let mut show_spawn = use_signal(|| false);
    let mut spawn_task_str = use_signal(String::new);
    let mut spawn_provider = use_signal(move || initial_spawn_provider.clone());
    let mut spawn_model = use_signal(move || initial_spawn_model.clone());

    let mut show_broadcast = use_signal(|| false);
    let mut selected_ids = use_signal(HashSet::<String>::new);
    let mut bcast_msg = use_signal(String::new);
    let mut bcast_busy = use_signal(|| false);
    let mut bcast_results = use_signal(Vec::<BroadcastResult>::new);

    let mut ep_ping_ms: Signal<HashMap<String, Option<u64>>> = use_signal(HashMap::new);
    let mut ep_ping_busy: Signal<HashSet<String>> = use_signal(HashSet::new);
    let mut ep_metrics: Signal<HashMap<String, Vec<(String, u64)>>> = use_signal(HashMap::new);
    let mut ep_metrics_busy: Signal<HashSet<String>> = use_signal(HashSet::new);
    let mut ep_show_metrics: Signal<HashSet<String>> = use_signal(HashSet::new);

    let mut task_filter = use_signal(String::new);
    let mut show_output: Signal<HashSet<String>> = use_signal(HashSet::new);
    let harnesses = use_signal(Vec::<HarnessInfo>::new);

    {
        let client = props.api_client.clone();
        use_effect(move || {
            if *poll_started.read() {
                return;
            }
            poll_started.set(true);
            let client = client.clone();
            spawn(async move {
                loop {
                    match client.list_endpoints().await {
                        Ok(list) => {
                            endpoints.set(list);
                            ep_error.set(None);
                        }
                        Err(e) => {
                            ep_error.set(Some(e.to_string()));
                        }
                    }
                    #[cfg(target_arch = "wasm32")]
                    gloo_timers::future::TimeoutFuture::new(10_000).await;
                    #[cfg(not(target_arch = "wasm32"))]
                    tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                }
            });
        });
    }

    let refresh = {
        let client = props.api_client.clone();
        move || {
            let client = client.clone();
            spawn(async move {
                if let Ok(list) = client.list_endpoints().await {
                    endpoints.set(list);
                }
            });
        }
    };

    let on_refresh = {
        let refresh = refresh.clone();
        move |_| refresh()
    };

    let on_ep_start = {
        let client = props.api_client.clone();
        let refresh = refresh.clone();
        move |id: String| {
            ep_op_busy.set(Some(id.clone()));
            let client = client.clone();
            let refresh = refresh.clone();
            spawn(async move {
                let _ = client.start_endpoint(&id).await;
                ep_op_busy.set(None);
                refresh();
            });
        }
    };

    let on_ep_stop = {
        let client = props.api_client.clone();
        let refresh = refresh.clone();
        move |id: String| {
            ep_op_busy.set(Some(id.clone()));
            let client = client.clone();
            let refresh = refresh.clone();
            spawn(async move {
                let _ = client.stop_endpoint(&id).await;
                ep_op_busy.set(None);
                refresh();
            });
        }
    };

    let on_ep_delete = {
        let client = props.api_client.clone();
        let refresh = refresh.clone();
        move |id: String| {
            ep_op_busy.set(Some(id.clone()));
            let client = client.clone();
            let refresh = refresh.clone();
            spawn(async move {
                let _ = client.delete_endpoint(&id).await;
                ep_op_busy.set(None);
                refresh();
            });
        }
    };

    let on_ep_chat_send = move |(ep_id, ep_url): (String, String)| {
        let msg = ep_chat_inp.read().get(&ep_id).cloned().unwrap_or_default();
        let msg = msg.trim().to_string();
        if msg.is_empty() {
            return;
        }

        ep_chat_inp.write().insert(ep_id.clone(), String::new());
        ep_chat
            .write()
            .entry(ep_id.clone())
            .or_default()
            .push(ChatMsg {
                role: "user".to_string(),
                text: msg.clone(),
            });
        ep_chat_busy.write().insert(ep_id.clone());

        spawn(async move {
            let client = ApiClient::for_endpoint(&ep_url);
            let result = client.chat_simple(&msg, None, None).await;
            let reply = match result {
                Ok(r) => ChatMsg {
                    role: "assistant".to_string(),
                    text: r,
                },
                Err(e) => ChatMsg {
                    role: "error".to_string(),
                    text: e.to_string(),
                },
            };
            ep_chat
                .write()
                .entry(ep_id.clone())
                .or_default()
                .push(reply);
            ep_chat_busy.write().remove(&ep_id);
        });
    };

    let on_task_inject = move |task_id: String| {
        let text = task_inject
            .read()
            .get(&task_id)
            .cloned()
            .unwrap_or_default();
        let text = text.trim().to_string();
        if text.is_empty() {
            return;
        }
        task_inject.write().insert(task_id.clone(), String::new());
        let mut store = props.agents.write();
        if let Some(a) = store.iter_mut().find(|a| a.id == task_id) {
            a.inject_command(text);
        }
    };

    let on_broadcast = {
        let client = props.api_client.clone();
        move |_| {
            let msg = bcast_msg.read().trim().to_string();
            if msg.is_empty() {
                return;
            }
            let sel = selected_ids.read().clone();
            if sel.is_empty() {
                return;
            }

            let ep_list = endpoints.read().clone();
            let task_list = props.agents.read().clone();
            let harness_list = harnesses.read().clone();
            let client = client.clone();

            bcast_busy.set(true);
            bcast_results.set(vec![]);

            let msg_clone = msg.clone();
            spawn(async move {
                let mut results = Vec::new();
                dispatch_selected_endpoints(&ep_list, &sel, &msg_clone, &mut results, ep_chat)
                    .await;
                bcast_results.set(results.clone());
                inject_selected_tasks(&task_list, &sel, &msg_clone, props.agents, &mut results);
                let harness_results =
                    dispatch_selected_harnesses(&client, &harness_list, &sel, &msg_clone).await;
                results.extend(harness_results);
                bcast_results.set(results);
                bcast_busy.set(false);
            });
        }
    };

    let ep_list = endpoints.read().clone();
    let task_list = props.agents.read().clone();
    let harness_list = harnesses.read().clone();

    let ep_running = ep_list.iter().filter(|e| e.status == "running").count();
    let task_active = task_list.iter().filter(|a| a.status.is_active()).count();
    let harness_count = harness_list.len();

    let sel = selected_ids.read();
    let sel_count = sel.len();
    let sel_ep_running = ep_list
        .iter()
        .filter(|e| sel.contains(&e.id) && e.status == "running")
        .count();
    let sel_task_active = task_list
        .iter()
        .filter(|a| sel.contains(&a.id) && !a.status.is_terminal())
        .count();
    let sel_harness = harness_list.iter().filter(|h| sel.contains(&h.id)).count();
    drop(sel);

    let ep_label = if ep_list.is_empty() {
        "no endpoints".to_string()
    } else if ep_running > 0 {
        format!("{} running, {} total", ep_running, ep_list.len())
    } else {
        format!("{} endpoints", ep_list.len())
    };

    let task_label = if task_list.is_empty() {
        "no tasks".to_string()
    } else if task_active > 0 {
        format!("{} active, {} total", task_active, task_list.len())
    } else {
        format!("{} tasks", task_list.len())
    };

    let total_tokens: u32 = task_list.iter().map(|a| a.token_usage).sum();
    let total_output_chunks: usize = task_list.iter().map(|a| a.output.len()).sum();
    let filter_str = task_filter.read().to_lowercase();
    let filtered_tasks: Vec<_> = task_list
        .iter()
        .filter(|a| {
            filter_str.is_empty()
                || a.name.to_lowercase().contains(&filter_str)
                || a.task.to_lowercase().contains(&filter_str)
                || a.status.label().contains(&filter_str)
        })
        .cloned()
        .collect();
    let completed_count = task_list.iter().filter(|a| a.status.is_terminal()).count();
    let task_list_empty = task_list.is_empty();

    let bcast_sel_label = if sel_count == 0 {
        "select targets below to push a command to all of them at once".to_string()
    } else {
        format!(
            "{} selected ({} running endpoints, {} harnesses, {} active tasks)",
            sel_count, sel_ep_running, sel_harness, sel_task_active
        )
    };
    let bcast_btn_label = if *bcast_busy.read() {
        "broadcasting…".to_string()
    } else {
        format!("send to {} targets", sel_count)
    };

    rsx! {
        div {
            style: "flex: 1; min-height: 0; display: flex; flex-direction: column; overflow: hidden;",

            FleetStatsBar {
                ep_running,
                harness_count,
                task_active,
                total_tokens,
                total_output_chunks,
            }

            BroadcastPanel {
                bcast_sel_label,
                bcast_btn_label,
                sel_count,
                show_broadcast,
                selected_ids,
                bcast_msg,
                bcast_busy,
                bcast_results,
                endpoints,
                harnesses,
                agents: props.agents,
                on_broadcast,
            }

            EndpointsSection {
                api_client: props.api_client.clone(),
                address: props.api_base_url.clone(),
                ep_label,
                ep_list: ep_list.clone(),
                ep_error,
                endpoints,
                provider_options,
                model_options,
                current_provider_setting,
                current_model_setting,
                show_launch,
                form_name,
                form_workspace,
                form_port,
                form_provider,
                form_model,
                form_auto_start,
                form_error,
                form_busy,
                ep_expanded,
                selected_ids,
                ep_op_busy,
                ep_chat,
                ep_chat_inp,
                ep_chat_busy,
                ep_ping_ms,
                ep_ping_busy,
                ep_metrics,
                ep_metrics_busy,
                ep_show_metrics,
                on_refresh,
                on_connect: props.on_connect_endpoint,
                on_start: on_ep_start,
                on_stop: on_ep_stop,
                on_delete: on_ep_delete,
                on_chat_send: on_ep_chat_send,
            }

            HarnessesSection {
                api_client: props.api_client.clone(),
                address: props.api_base_url.clone(),
                selected_ids,
                harnesses,
            }

            TasksSection {
                agents: props.agents,
                task_label,
                filtered_tasks,
                task_list_empty,
                completed_count,
                provider_options,
                model_options,
                current_provider_setting,
                current_model_setting,
                task_filter,
                show_spawn,
                spawn_task_str,
                spawn_provider,
                spawn_model,
                task_expanded,
                selected_ids,
                task_inject,
                show_output,
                on_spawn: props.on_spawn_task,
                on_cancel: props.on_cancel_task,
                on_open_output: props.on_open_task_output,
                on_inject: on_task_inject,
            }
        }
    }
}

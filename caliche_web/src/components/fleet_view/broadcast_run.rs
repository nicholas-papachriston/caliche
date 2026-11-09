//! Fan-out a single operator message to selected endpoints, tasks, and harnesses.

use std::collections::{HashMap, HashSet};

use dioxus::prelude::*;

use crate::api::{ApiClient, EndpointInfo, HarnessInfo};
use crate::state::agents::SubAgent;

use super::types::{BroadcastResult, ChatMsg};

/// Dispatch `message` to every selected registered CLI harness.
pub async fn dispatch_selected_harnesses(
    client: &ApiClient,
    harnesses: &[HarnessInfo],
    selected: &HashSet<String>,
    message: &str,
) -> Vec<BroadcastResult> {
    let mut results = Vec::new();
    for harness in harnesses {
        if !selected.contains(&harness.id) {
            continue;
        }
        match client.dispatch_harness(&harness.id, message).await {
            Ok(response) => results.push(BroadcastResult {
                name: harness.name.clone(),
                kind: "harness".to_string(),
                ok: response.ok,
                text: response.text,
            }),
            Err(err) => results.push(BroadcastResult {
                name: harness.name.clone(),
                kind: "harness".to_string(),
                ok: false,
                text: err.to_string(),
            }),
        }
    }
    results
}

/// Chat one selected running endpoint and append the reply to `ep_chat`.
pub async fn dispatch_selected_endpoints(
    endpoints: &[EndpointInfo],
    selected: &HashSet<String>,
    message: &str,
    results: &mut Vec<BroadcastResult>,
    mut ep_chat: Signal<HashMap<String, Vec<ChatMsg>>>,
) {
    let ep_targets: Vec<(String, String, String)> = endpoints
        .iter()
        .filter(|endpoint| selected.contains(&endpoint.id) && endpoint.status == "running")
        .map(|endpoint| {
            (
                endpoint.id.clone(),
                endpoint.name.clone(),
                endpoint.url.clone(),
            )
        })
        .collect();

    for (ep_id, ep_name, ep_url) in ep_targets {
        let ep_client = ApiClient::for_endpoint(&ep_url);
        let result = ep_client.chat_simple(message, None, None).await;
        let ok = result.is_ok();
        let text = result.unwrap_or_else(|err| err.to_string());
        results.push(BroadcastResult {
            name: ep_name,
            kind: "endpoint".to_string(),
            ok,
            text: text.clone(),
        });
        {
            let mut chats = ep_chat.write();
            let history = chats.entry(ep_id).or_default();
            history.push(ChatMsg {
                role: "user".to_string(),
                text: message.to_string(),
            });
            history.push(ChatMsg {
                role: if ok {
                    "assistant".to_string()
                } else {
                    "error".to_string()
                },
                text,
            });
        }
    }
}

/// Inject `message` into every selected non-terminal local task.
pub fn inject_selected_tasks(
    tasks: &[SubAgent],
    selected: &HashSet<String>,
    message: &str,
    mut agents: Signal<Vec<SubAgent>>,
    results: &mut Vec<BroadcastResult>,
) {
    let task_targets: Vec<(String, String)> = tasks
        .iter()
        .filter(|agent| selected.contains(&agent.id) && !agent.status.is_terminal())
        .map(|agent| (agent.id.clone(), agent.name.clone()))
        .collect();

    let mut store = agents.write();
    for (task_id, task_name) in &task_targets {
        if let Some(agent) = store.iter_mut().find(|agent| &agent.id == task_id) {
            agent.inject_command(message);
            results.push(BroadcastResult {
                name: task_name.clone(),
                kind: "task".to_string(),
                ok: true,
                text: "command injected".to_string(),
            });
        }
    }
}

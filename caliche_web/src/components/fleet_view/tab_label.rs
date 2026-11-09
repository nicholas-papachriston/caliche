use crate::api::EndpointInfo;
use crate::state::agents::SubAgent;

/// Tab label showing running endpoint + active task counts.
pub fn fleet_tab_label(endpoints: &[EndpointInfo], agents: &[SubAgent]) -> String {
    let ep_running = endpoints.iter().filter(|e| e.status == "running").count();
    let task_running = agents.iter().filter(|a| a.status.is_active()).count();
    match (ep_running, task_running) {
        (0, 0) => "fleet".to_string(),
        (e, 0) => format!("fleet ({e} ep)"),
        (0, t) => format!("fleet ({t} tasks)"),
        (e, t) => format!("fleet ({e} ep, {t} tasks)"),
    }
}

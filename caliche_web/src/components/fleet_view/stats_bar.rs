use dioxus::prelude::*;

use crate::components::chrome::FLEET_SENTENCE;

#[derive(Props, Clone, PartialEq)]
pub struct FleetStatsBarProps {
    pub ep_running: usize,
    pub harness_count: usize,
    pub task_active: usize,
    pub total_tokens: u32,
    pub total_output_chunks: usize,
}

#[component]
pub fn FleetStatsBar(props: FleetStatsBarProps) -> Element {
    let ep_running = props.ep_running;
    let harness_count = props.harness_count;
    let task_active = props.task_active;
    let total_tokens = props.total_tokens;
    let total_output_chunks = props.total_output_chunks;

    rsx! {
        div { class: "section-header",
            span { class: "section-title", "FLEET" }
            span { class: "section-summary",
                "{FLEET_SENTENCE} · {ep_running} endpoints running · {harness_count} harnesses · {task_active} tasks active"
            }
            div { class: "section-actions",
                if total_tokens > 0 {
                    span { class: "worker-meta", "{total_tokens} tokens" }
                }
                if total_output_chunks > 0 {
                    span { class: "worker-meta", "{total_output_chunks} chunks" }
                }
            }
        }
    }
}

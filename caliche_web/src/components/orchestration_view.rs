//! Orchestration view — multi-agent patterns plus coordinator routing.

use crate::api::{
    AgentConfig, ApiClient, MultiAgentPattern, MultiAgentRequest, MultiAgentResponse,
};
use crate::components::chrome::{ChipButton, FieldRow, SectionHeader};
use crate::components::orchestration_coordinator::OrchestrationCoordinator;
use dioxus::prelude::*;

#[derive(Debug, Clone, PartialEq)]
enum OrchestrationMode {
    MultiAgent,
    Coordinator,
}

#[component]
pub fn OrchestrationView(api_client: ApiClient) -> Element {
    let mut mode = use_signal(|| OrchestrationMode::MultiAgent);
    let mut selected_pattern = use_signal(|| MultiAgentPattern::Pipeline);
    let mut task = use_signal(String::new);
    let mut agents: Signal<Vec<AgentConfig>> = use_signal(|| {
        vec![
            AgentConfig {
                id: "agent-1".to_string(),
                name: "Agent 1".to_string(),
                role: Some("primary".to_string()),
                provider: None,
                model: None,
            },
            AgentConfig {
                id: "agent-2".to_string(),
                name: "Agent 2".to_string(),
                role: Some("secondary".to_string()),
                provider: None,
                model: None,
            },
        ]
    });
    let mut is_running = use_signal(|| false);
    let mut result: Signal<Option<MultiAgentResponse>> = use_signal(|| None);
    let mut error_msg: Signal<Option<String>> = use_signal(|| None);

    let run_multi_agent = {
        let api = api_client.clone();
        move |_| {
            if *is_running.read() {
                return;
            }
            let task_val = task.read().trim().to_string();
            if task_val.is_empty() {
                return;
            }
            if agents.read().is_empty() {
                return;
            }

            let request = MultiAgentRequest {
                pattern: selected_pattern.read().clone(),
                agents: agents.read().clone(),
                task: task_val,
            };

            is_running.set(true);
            result.set(None);
            error_msg.set(None);

            let api = api.clone();
            spawn(async move {
                match api.multi_agent(&request).await {
                    Ok(r) => result.set(Some(r)),
                    Err(e) => error_msg.set(Some(e.to_string())),
                }
                is_running.set(false);
            });
        }
    };

    let add_agent = move |_| {
        let idx = agents.read().len() + 1;
        agents.write().push(AgentConfig {
            id: format!("agent-{}", idx),
            name: format!("Agent {}", idx),
            role: None,
            provider: None,
            model: None,
        });
    };

    let current_mode = mode.read().clone();

    rsx! {
        div { class: "orch-page",
            SectionHeader {
                title: "ORCHESTRATION".to_string(),
                summary: "Run a pattern across agents, or route a task by capability.".to_string(),
            }

            div { class: "chip-row",
                ChipButton {
                    label: "multi-agent".to_string(),
                    active: current_mode == OrchestrationMode::MultiAgent,
                    on_click: move |_| mode.set(OrchestrationMode::MultiAgent),
                }
                ChipButton {
                    label: "coordinator".to_string(),
                    active: current_mode == OrchestrationMode::Coordinator,
                    on_click: move |_| mode.set(OrchestrationMode::Coordinator),
                }
            }

            if current_mode == OrchestrationMode::Coordinator {
                OrchestrationCoordinator {
                    api_client: api_client.clone(),
                }
            }

            if current_mode == OrchestrationMode::MultiAgent {
                div { class: "chip-row",
                    for pat in [
                        MultiAgentPattern::Pipeline,
                        MultiAgentPattern::Debate,
                        MultiAgentPattern::Hierarchical,
                        MultiAgentPattern::Supervisor,
                        MultiAgentPattern::FanOutCollect,
                    ] {
                        {
                            let p = pat.clone();
                            let is_sel = *selected_pattern.read() == pat;
                            let label = pat.label();
                            rsx! {
                                ChipButton {
                                    label: label.to_string(),
                                    active: is_sel,
                                    on_click: move |_| selected_pattern.set(p.clone()),
                                }
                            }
                        }
                    }
                }

                p { class: "hint-banner", "{selected_pattern.read().description()}" }

                div { class: "orch-scroll",
                    div {
                        style: "display: flex; flex-direction: column; gap: 0.4ch; flex-shrink: 0;",
                        SectionHeader {
                            title: "ROLES".to_string(),
                            summary: format!(
                                "{} roles in this pattern, not the fleet list",
                                agents.read().len()
                            ),
                            button {
                                class: "chip",
                                onclick: add_agent,
                                "+ add agent"
                            }
                        }

                        div { class: "agent-grid header",
                            span { class: "worker-meta", "id" }
                            span { class: "worker-meta", "name" }
                            span { class: "worker-meta", "role" }
                            span { class: "worker-meta", "provider" }
                            span { class: "worker-meta", "model" }
                            span {}
                        }

                        for idx in 0..agents.read().len() {
                            div { class: "agent-grid",
                                input {
                                    value: "{agents.read()[idx].id}",
                                    oninput: move |e| { agents.write()[idx].id = e.value(); }
                                }
                                input {
                                    value: "{agents.read()[idx].name}",
                                    oninput: move |e| { agents.write()[idx].name = e.value(); }
                                }
                                input {
                                    value: "{agents.read()[idx].role.as_deref().unwrap_or(\"\")}",
                                    placeholder: "optional",
                                    oninput: move |e| {
                                        let v = e.value();
                                        agents.write()[idx].role = if v.is_empty() { None } else { Some(v) };
                                    }
                                }
                                input {
                                    value: "{agents.read()[idx].provider.as_deref().unwrap_or(\"\")}",
                                    placeholder: "default",
                                    oninput: move |e| {
                                        let v = e.value();
                                        agents.write()[idx].provider = if v.is_empty() { None } else { Some(v) };
                                    }
                                }
                                input {
                                    value: "{agents.read()[idx].model.as_deref().unwrap_or(\"\")}",
                                    placeholder: "default",
                                    oninput: move |e| {
                                        let v = e.value();
                                        agents.write()[idx].model = if v.is_empty() { None } else { Some(v) };
                                    }
                                }
                                button {
                                    style: "background: transparent; color: var(--error); border: none; cursor: pointer; padding: 0;",
                                    onclick: move |_| {
                                        let mut w = agents.write();
                                        if w.len() > 1 { w.remove(idx); }
                                    },
                                    "x"
                                }
                            }
                        }
                    }

                    FieldRow {
                        label: "task".to_string(),
                        help: "Sent to every agent in the pattern.".to_string(),
                        textarea {
                            style: "height: 6ch; resize: vertical; width: 100%; box-sizing: border-box;",
                            placeholder: "Describe the task for the agent team",
                            value: "{task}",
                            oninput: move |e| task.set(e.value()),
                        }
                    }

                    button {
                        class: "primary",
                        style: "padding: 0.4ch 2ch; align-self: flex-start;",
                        disabled: *is_running.read() || task.read().trim().is_empty(),
                        onclick: run_multi_agent,
                        if *is_running.read() { "running agents…" } else { "run orchestration" }
                    }

                    if let Some(ref err) = *error_msg.read() {
                        div { class: "form-error", "error: {err}" }
                    }

                    if let Some(ref r) = result.read().clone() {
                        div {
                            style: "display: flex; flex-direction: column; gap: 0.8ch;",
                            div { class: "worker-row-main",
                                span { class: "worker-kind", "[{r.pattern}]" }
                                span { class: "worker-meta", "{r.agent_results.len()} agents completed" }
                            }

                            for agent_result in &r.agent_results {
                                div { class: "result-block",
                                    div { class: "worker-row-main",
                                        span { class: "worker-kind", "{agent_result.agent_name}" }
                                        if let Some(ref err) = agent_result.error {
                                            span { class: "worker-error", "error: {err}" }
                                        }
                                    }
                                    pre { class: "result-pre", "{agent_result.output}" }
                                }
                            }

                            div { class: "result-block final",
                                span { class: "worker-kind", "final result" }
                                pre { class: "result-pre", "{r.final_result}" }
                            }
                        }
                    }
                }
            }
        }
    }
}

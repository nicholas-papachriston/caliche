//! Workflow builder and executor — DAG-based workflow orchestration.

use crate::api::{
    ApiClient, InterruptsListResponse, ResumeInterruptRequest, WorkflowEdge, WorkflowNode,
    WorkflowRequest, WorkflowResponse, WorkflowStatusResponse,
};
use dioxus::prelude::*;

#[component]
pub fn WorkflowView(api_client: ApiClient) -> Element {
    // Nodes
    let mut nodes: Signal<Vec<WorkflowNode>> = use_signal(Vec::new);
    // Edges
    let mut edges: Signal<Vec<WorkflowEdge>> = use_signal(Vec::new);
    // New node form
    let mut new_node_type = use_signal(|| "llm".to_string());
    let mut new_node_label = use_signal(String::new);
    let mut new_node_prompt = use_signal(String::new);
    // New edge form
    let mut new_edge_from = use_signal(String::new);
    let mut new_edge_to = use_signal(String::new);
    // Execution
    let mut input_json = use_signal(|| "{}".to_string());
    let mut is_running = use_signal(|| false);
    let mut result: Signal<Option<WorkflowResponse>> = use_signal(|| None);
    let mut error_msg: Signal<Option<String>> = use_signal(|| None);
    // Workflow status polling
    let mut active_workflow_id: Signal<Option<String>> = use_signal(|| None);
    let mut workflow_status: Signal<Option<WorkflowStatusResponse>> = use_signal(|| None);
    // Interrupts
    let mut interrupts: Signal<InterruptsListResponse> =
        use_signal(InterruptsListResponse::default);
    let mut interrupt_feedback = use_signal(String::new);
    let mut interrupts_loading = use_signal(|| false);
    // node counter for unique ids
    let mut node_counter = use_signal(|| 0u32);

    let add_node = {
        let _api = api_client.clone();
        move |_| {
            let label = new_node_label.read().trim().to_string();
            if label.is_empty() {
                return;
            }
            let counter = *node_counter.read();
            node_counter.set(counter + 1);
            let node = WorkflowNode {
                id: format!("node-{}", counter),
                node_type: new_node_type.read().clone(),
                label: label.clone(),
                prompt: {
                    let p = new_node_prompt.read().trim().to_string();
                    if p.is_empty() { None } else { Some(p) }
                },
                tool: None,
                config: None,
            };
            nodes.write().push(node);
            new_node_label.set(String::new());
            new_node_prompt.set(String::new());
        }
    };

    let add_edge = move |_| {
        let from = new_edge_from.read().trim().to_string();
        let to = new_edge_to.read().trim().to_string();
        if from.is_empty() || to.is_empty() || from == to {
            return;
        }
        let edge = WorkflowEdge {
            from,
            to,
            condition: None,
        };
        edges.write().push(edge);
    };

    let run_workflow = {
        let api = api_client.clone();
        move |_| {
            if *is_running.read() {
                return;
            }
            let current_nodes = nodes.read().clone();
            if current_nodes.is_empty() {
                return;
            }
            let input_str = input_json.read().clone();
            let input_val: serde_json::Value = serde_json::from_str(&input_str)
                .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
            let req = WorkflowRequest {
                nodes: current_nodes,
                edges: edges.read().clone(),
                input: input_val,
                metadata: None,
            };
            is_running.set(true);
            result.set(None);
            error_msg.set(None);
            active_workflow_id.set(None);
            workflow_status.set(None);
            let api = api.clone();
            spawn(async move {
                match api.execute_workflow(&req).await {
                    Ok(r) => {
                        active_workflow_id.set(Some(r.workflow_id.clone()));
                        result.set(Some(r));
                    }
                    Err(e) => error_msg.set(Some(e.to_string())),
                }
                is_running.set(false);
            });
        }
    };

    let refresh_status = {
        let api = api_client.clone();
        move |_| {
            if let Some(wid) = active_workflow_id.read().clone() {
                let api = api.clone();
                spawn(async move {
                    if let Ok(s) = api.get_workflow_status(&wid).await {
                        workflow_status.set(Some(s));
                    }
                });
            }
        }
    };

    let load_interrupts = {
        let api = api_client.clone();
        move |_| {
            interrupts_loading.set(true);
            let api = api.clone();
            spawn(async move {
                match api.list_interrupts().await {
                    Ok(r) => interrupts.set(r),
                    Err(_) => {}
                }
                interrupts_loading.set(false);
            });
        }
    };

    rsx! {
        div {
            style: "display:flex;flex-direction:column;height:100%;overflow:hidden;padding:1.5ch;gap:1.2ch;box-sizing:border-box;",

            div {
                style: "color:var(--accent-color,#4af);font-family:var(--font-mono);font-size:var(--font-size-sm,0.85rem);font-weight:bold;border-bottom:1px solid var(--border-color);padding-bottom:0.6ch;flex-shrink:0;",
                "workflow builder"
            }

            div {
                style: "display:flex;gap:1.5ch;flex:1;min-height:0;",

                // Left: Builder
                div {
                    style: "flex:1;display:flex;flex-direction:column;gap:1ch;overflow-y:auto;min-width:0;",

                    // Node form
                    div {
                        style: "background:var(--secondary-bg);border:1px solid var(--border-color);padding:0.8ch;display:flex;flex-direction:column;gap:0.6ch;",
                        div { style: "color:var(--muted-color);font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);", "add node" }
                        div {
                            style: "display:flex;gap:0.6ch;align-items:center;flex-wrap:wrap;",
                            select {
                                style: "background:var(--secondary-bg);color:var(--text-color);border:1px solid var(--border-color);padding:0.2ch 0.5ch;font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);",
                                oninput: move |e| new_node_type.set(e.value()),
                                option { value: "llm", "llm" }
                                option { value: "tool", "tool" }
                                option { value: "condition", "condition" }
                                option { value: "input", "input" }
                                option { value: "output", "output" }
                            }
                            input {
                                r#type: "text",
                                placeholder: "node label",
                                value: "{new_node_label}",
                                style: "flex:1;background:var(--secondary-bg);color:var(--text-color);border:1px solid var(--border-color);padding:0.2ch 0.5ch;font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);",
                                oninput: move |e| new_node_label.set(e.value()),
                            }
                            button {
                                style: "background:var(--accent-color,#4af);color:var(--background);border:none;padding:0.2ch 1ch;font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);cursor:pointer;",
                                onclick: add_node,
                                "+"
                            }
                        }
                        input {
                            r#type: "text",
                            placeholder: "prompt (optional, for llm nodes)",
                            value: "{new_node_prompt}",
                            style: "background:var(--secondary-bg);color:var(--text-color);border:1px solid var(--border-color);padding:0.2ch 0.5ch;font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);width:100%;box-sizing:border-box;",
                            oninput: move |e| new_node_prompt.set(e.value()),
                        }
                    }

                    // Node list
                    if !nodes.read().is_empty() {
                        div {
                            style: "display:flex;flex-direction:column;gap:0.3ch;",
                            for (idx, node) in nodes.read().iter().enumerate() {
                                {
                                    let ntype = node.node_type.clone();
                                    let nlabel = node.label.clone();
                                    let nid = node.id.clone();
                                    let type_color = match ntype.as_str() {
                                        "llm" => "#4af",
                                        "tool" => "#fa4",
                                        "condition" => "#a4f",
                                        "input" => "#4f4",
                                        "output" => "#f44",
                                        _ => "#888",
                                    };
                                    rsx! {
                                        div {
                                            key: "{nid}",
                                            style: "display:flex;align-items:center;gap:0.6ch;background:var(--secondary-bg);border:1px solid var(--border-color);padding:0.3ch 0.6ch;",
                                            span { style: "color:{type_color};font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);min-width:8ch;", "[{ntype}]" }
                                            span { style: "flex:1;font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);color:var(--text-color);", "{nlabel}" }
                                            span { style: "color:var(--muted-color);font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);", "{nid}" }
                                            button {
                                                style: "background:transparent;color:#f44;border:none;font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);cursor:pointer;padding:0;",
                                                onclick: move |_| { nodes.write().remove(idx); },
                                                "×"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Edge form
                    div {
                        style: "background:var(--secondary-bg);border:1px solid var(--border-color);padding:0.8ch;display:flex;flex-direction:column;gap:0.6ch;",
                        div { style: "color:var(--muted-color);font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);", "add edge" }
                        div {
                            style: "display:flex;gap:0.6ch;align-items:center;",
                            select {
                                style: "flex:1;background:var(--secondary-bg);color:var(--text-color);border:1px solid var(--border-color);padding:0.2ch 0.4ch;font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);",
                                oninput: move |e| new_edge_from.set(e.value()),
                                option { value: "", "from..." }
                                for node in nodes.read().iter() {
                                    option { value: "{node.id}", "{node.label}" }
                                }
                            }
                            span { style: "color:var(--muted-color);font-family:var(--font-mono);", "→" }
                            select {
                                style: "flex:1;background:var(--secondary-bg);color:var(--text-color);border:1px solid var(--border-color);padding:0.2ch 0.4ch;font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);",
                                oninput: move |e| new_edge_to.set(e.value()),
                                option { value: "", "to..." }
                                for node in nodes.read().iter() {
                                    option { value: "{node.id}", "{node.label}" }
                                }
                            }
                            button {
                                style: "background:var(--accent-color,#4af);color:var(--background);border:none;padding:0.2ch 1ch;font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);cursor:pointer;",
                                onclick: add_edge,
                                "+"
                            }
                        }
                    }

                    // Edge list
                    if !edges.read().is_empty() {
                        div {
                            style: "display:flex;flex-direction:column;gap:0.3ch;",
                            for (idx, edge) in edges.read().iter().enumerate() {
                                {
                                    let ef = edge.from.clone();
                                    let et = edge.to.clone();
                                    rsx! {
                                        div {
                                            key: "{ef}-{et}",
                                            style: "display:flex;align-items:center;gap:0.6ch;background:var(--secondary-bg);border:1px solid var(--border-color);padding:0.3ch 0.6ch;",
                                            span { style: "flex:1;font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);color:var(--text-color);", "{ef} → {et}" }
                                            button {
                                                style: "background:transparent;color:#f44;border:none;font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);cursor:pointer;padding:0;",
                                                onclick: move |_| { edges.write().remove(idx); },
                                                "×"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Input JSON
                    div {
                        style: "display:flex;flex-direction:column;gap:0.4ch;",
                        span { style: "color:var(--muted-color);font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);", "input (json):" }
                        textarea {
                            style: "height:5ch;background:var(--secondary-bg);color:var(--text-color);border:1px solid var(--border-color);padding:0.4ch;font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);resize:vertical;box-sizing:border-box;width:100%;",
                            value: "{input_json}",
                            oninput: move |e| input_json.set(e.value()),
                        }
                    }

                    // Execute button
                    button {
                        style: if *is_running.read() {
                            "background:var(--secondary-bg);color:var(--muted-color);border:1px solid var(--border-color);padding:0.4ch 2ch;font-family:var(--font-mono);font-size:var(--font-size-sm,0.85rem);cursor:not-allowed;"
                        } else {
                            "background:var(--accent-color,#4af);color:var(--background);border:none;padding:0.4ch 2ch;font-family:var(--font-mono);font-size:var(--font-size-sm,0.85rem);cursor:pointer;"
                        },
                        disabled: *is_running.read(),
                        onclick: run_workflow,
                        if *is_running.read() { "executing..." } else { "execute workflow" }
                    }
                }

                // Right: Results + Interrupts
                div {
                    style: "flex:1;display:flex;flex-direction:column;gap:1ch;overflow-y:auto;min-width:0;",

                    // Error
                    if let Some(ref err) = *error_msg.read() {
                        div {
                            style: "color:#f44;font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);background:var(--secondary-bg);padding:0.6ch 1ch;border-left:2px solid #f44;",
                            "error: {err}"
                        }
                    }

                    // Result
                    if let Some(ref r) = *result.read() {
                        div {
                            style: "background:var(--secondary-bg);border:1px solid var(--border-color);padding:0.8ch;display:flex;flex-direction:column;gap:0.6ch;",
                            div {
                                style: "display:flex;align-items:center;gap:1ch;",
                                span { style: "color:var(--accent-color,#4af);font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);", "workflow: {r.workflow_id}" }
                                span {
                                    style: if r.state == "completed" || r.state == "done" {
                                        "color:#4f4;font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);"
                                    } else if r.state == "failed" {
                                        "color:#f44;font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);"
                                    } else {
                                        "color:#fa4;font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);"
                                    },
                                    "[{r.state}]"
                                }
                                button {
                                    style: "background:var(--secondary-bg);border:1px solid var(--border-color);color:var(--text-color);padding:0.2ch 0.8ch;font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);cursor:pointer;margin-left:auto;",
                                    onclick: refresh_status,
                                    "refresh status"
                                }
                            }
                            if let Some(ref e) = r.error {
                                div { style: "color:#f44;font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);", "{e}" }
                            }
                            if !r.results.is_empty() {
                                div {
                                    style: "display:flex;flex-direction:column;gap:0.4ch;",
                                    span { style: "color:var(--muted-color);font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);", "results ({r.results.len()}):" }
                                    for (i, res) in r.results.iter().enumerate() {
                                        div {
                                            key: "{i}",
                                            style: "background:var(--background);border:1px solid var(--border-color);padding:0.4ch;",
                                            pre {
                                                style: "margin:0;font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);color:var(--text-color);white-space:pre-wrap;word-break:break-word;",
                                                { serde_json::to_string_pretty(res).unwrap_or_default() }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Status
                    if let Some(ref s) = *workflow_status.read() {
                        div {
                            style: "background:var(--secondary-bg);border:1px solid var(--border-color);padding:0.8ch;display:flex;flex-direction:column;gap:0.4ch;",
                            div { style: "color:var(--muted-color);font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);", "status: [{s.state}]" }
                            if let Some(ref cn) = s.current_node {
                                div { style: "color:#fa4;font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);", "current: {cn}" }
                            }
                            if !s.completed_nodes.is_empty() {
                                div { style: "color:#4f4;font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);", "completed: {s.completed_nodes.join(\", \")}" }
                            }
                        }
                    }

                    // Interrupts panel
                    div {
                        style: "background:var(--secondary-bg);border:1px solid var(--border-color);padding:0.8ch;display:flex;flex-direction:column;gap:0.6ch;",
                        div {
                            style: "display:flex;align-items:center;gap:1ch;",
                            span { style: "color:var(--muted-color);font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);", "pending interrupts" }
                            button {
                                style: "background:transparent;border:1px solid var(--border-color);color:var(--text-color);padding:0.2ch 0.6ch;font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);cursor:pointer;margin-left:auto;",
                                disabled: *interrupts_loading.read(),
                                onclick: load_interrupts,
                                if *interrupts_loading.read() { "loading..." } else { "load" }
                            }
                        }
                        if interrupts.read().interrupts.is_empty() {
                            div { style: "color:var(--muted-color);font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);", "no pending interrupts" }
                        }
                        for interrupt in interrupts.read().interrupts.iter() {
                            {
                                let iid = interrupt.id.clone();
                                let iprompt = interrupt.prompt.clone();
                                let iwf = interrupt.workflow_id.clone();
                                let api2 = api_client.clone();
                                rsx! {
                                    div {
                                        key: "{iid}",
                                        style: "border:1px solid #fa4;padding:0.6ch;display:flex;flex-direction:column;gap:0.4ch;",
                                        div { style: "color:#fa4;font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);", "interrupt: {iid} (wf: {iwf})" }
                                        div { style: "color:var(--text-color);font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);", "{iprompt}" }
                                        div {
                                            style: "display:flex;gap:0.6ch;",
                                            input {
                                                r#type: "text",
                                                placeholder: "feedback (json value)",
                                                value: "{interrupt_feedback}",
                                                style: "flex:1;background:var(--secondary-bg);color:var(--text-color);border:1px solid var(--border-color);padding:0.2ch 0.4ch;font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);",
                                                oninput: move |e| interrupt_feedback.set(e.value()),
                                            }
                                            button {
                                                style: "background:var(--accent-color,#4af);color:var(--background);border:none;padding:0.2ch 0.8ch;font-family:var(--font-mono);font-size:var(--font-size-xs,0.75rem);cursor:pointer;",
                                                onclick: {
                                                    let iid2 = iid.clone();
                                                    let api3 = api2.clone();
                                                    move |_| {
                                                        let iid3 = iid2.clone();
                                                        let fb_str = interrupt_feedback.read().trim().to_string();
                                                        let fb_val: serde_json::Value =
                                                            serde_json::from_str(&fb_str).unwrap_or(
                                                                serde_json::Value::String(fb_str.clone()),
                                                            );
                                                        let req = ResumeInterruptRequest { feedback: fb_val };
                                                        let api4 = api3.clone();
                                                        spawn(async move {
                                                            let _ = api4.resume_interrupt(&iid3, &req).await;
                                                        });
                                                    }
                                                },
                                                "resume"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

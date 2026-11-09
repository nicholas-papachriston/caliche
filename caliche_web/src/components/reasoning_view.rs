//! Reasoning patterns view — run ReAct, Plan-Execute, and Tree-of-Thoughts.

use crate::api::{
    ApiClient, ReasoningPatternInfo, ReasoningPatternRequest, ReasoningPatternResponse,
};
use dioxus::prelude::*;

#[derive(Debug, Clone, PartialEq)]
enum SelectedPattern {
    React,
    PlanExecute,
    TreeOfThoughts,
    SelfAsk,
    Reflexion,
}

impl SelectedPattern {
    fn label(&self) -> &'static str {
        match self {
            Self::React => "ReAct",
            Self::PlanExecute => "Plan-Execute",
            Self::TreeOfThoughts => "Tree of Thoughts",
            Self::SelfAsk => "Self-Ask",
            Self::Reflexion => "Reflexion",
        }
    }

    fn description(&self) -> &'static str {
        match self {
            Self::React => {
                "Reason-Act loop: the agent alternates between reasoning about the problem and taking actions until it reaches a final answer."
            }
            Self::PlanExecute => {
                "Generate a multi-step plan up-front then execute each step in sequence, synthesizing a final answer from the results."
            }
            Self::TreeOfThoughts => {
                "Explore multiple reasoning branches simultaneously, evaluate them, and return the best solution found."
            }
            Self::SelfAsk => {
                "Decompose a complex question into sub-questions, answer each, then synthesize a final answer."
            }
            Self::Reflexion => {
                "Generate a response, reflect on its quality, then iteratively refine it based on self-critique."
            }
        }
    }
}

#[component]
pub fn ReasoningView(
    api_client: ApiClient,
    #[props(default)] reasoning_patterns: Vec<ReasoningPatternInfo>,
) -> Element {
    let mut selected = use_signal(|| SelectedPattern::React);
    let mut prompt = use_signal(String::new);
    let mut max_iterations = use_signal(|| 5usize);
    let mut is_running = use_signal(|| false);
    let mut result: Signal<Option<ReasoningPatternResponse>> = use_signal(|| None);
    let mut error_msg: Signal<Option<String>> = use_signal(|| None);
    let mut expanded_steps: Signal<std::collections::HashSet<usize>> =
        use_signal(std::collections::HashSet::new);

    let run_reasoning = {
        let api = api_client.clone();
        move |_| {
            if *is_running.read() {
                return;
            }
            let prompt_val = prompt.read().trim().to_string();
            if prompt_val.is_empty() {
                return;
            }

            let pattern = selected.read().clone();
            let iters = *max_iterations.read();
            let request = ReasoningPatternRequest {
                message: prompt_val,
                max_iterations: Some(iters),
                provider: None,
                model: None,
            };

            is_running.set(true);
            result.set(None);
            error_msg.set(None);

            let api = api.clone();
            spawn(async move {
                let res = match pattern {
                    SelectedPattern::React => api.reasoning_react(&request).await,
                    SelectedPattern::PlanExecute => api.reasoning_plan_execute(&request).await,
                    SelectedPattern::TreeOfThoughts => {
                        api.reasoning_tree_of_thoughts(&request).await
                    }
                    SelectedPattern::SelfAsk => api.reasoning_self_ask(&request).await,
                    SelectedPattern::Reflexion => api.reasoning_reflexion(&request).await,
                };
                match res {
                    Ok(r) => result.set(Some(r)),
                    Err(e) => error_msg.set(Some(e.to_string())),
                }
                is_running.set(false);
            });
        }
    };

    rsx! {
        div {
            style: "display: flex; flex-direction: column; height: 100%; overflow: hidden; padding: 1.5ch; gap: 1.2ch; box-sizing: border-box;",

            // Header
            div {
                style: "color: var(--accent-color, #4af); font-family: var(--font-mono); font-size: var(--font-size-sm, 0.85rem); font-weight: bold; border-bottom: 1px solid var(--border-color); padding-bottom: 0.6ch;",
                "reasoning patterns"
            }

            // Pattern selector
            div {
                style: "display: flex; gap: 0.8ch; flex-shrink: 0;",
                for pat in [SelectedPattern::React, SelectedPattern::PlanExecute, SelectedPattern::TreeOfThoughts, SelectedPattern::SelfAsk, SelectedPattern::Reflexion] {
                    {
                        let pat_clone = pat.clone();
                        let is_sel = *selected.read() == pat;
                        let label = pat.label();
                        rsx! {
                            button {
                                style: if is_sel {
                                    "background: var(--accent-color, #4af); color: var(--background); border: 1px solid var(--accent-color, #4af); padding: 0.3ch 1ch; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem); cursor: pointer; border-radius: 2px;"
                                } else {
                                    "background: transparent; color: var(--text-color); border: 1px solid var(--border-color); padding: 0.3ch 1ch; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem); cursor: pointer; border-radius: 2px;"
                                },
                                onclick: move |_| selected.set(pat_clone.clone()),
                                "{label}"
                            }
                        }
                    }
                }
            }

            // Pattern description
            div {
                style: "color: var(--muted-color); font-size: var(--font-size-xs, 0.75rem); font-family: var(--font-mono); background: var(--secondary-bg); padding: 0.6ch 1ch; border-left: 2px solid var(--accent-color, #4af); flex-shrink: 0;",
                "{selected.read().description()}"
            }

            // Configuration row
            div {
                style: "display: flex; gap: 1ch; align-items: center; flex-shrink: 0;",
                span { style: "color: var(--muted-color); font-size: var(--font-size-xs, 0.75rem); font-family: var(--font-mono);", "max iterations:" }
                input {
                    r#type: "number",
                    min: "1",
                    max: "20",
                    value: "{max_iterations}",
                    style: "width: 6ch; background: var(--secondary-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.2ch 0.5ch; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);",
                    oninput: move |e| {
                        if let Ok(n) = e.value().parse::<usize>() {
                            if n >= 1 && n <= 20 { max_iterations.set(n); }
                        }
                    }
                }
            }

            // Prompt input
            textarea {
                style: "flex-shrink: 0; height: 8ch; background: var(--secondary-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.6ch; font-family: var(--font-mono); font-size: var(--font-size-sm, 0.85rem); resize: none; box-sizing: border-box; width: 100%;",
                placeholder: "Enter a task or question for the reasoning agent...",
                value: "{prompt}",
                oninput: move |e| prompt.set(e.value()),
            }

            // Run button
            button {
                style: if *is_running.read() {
                    "background: var(--secondary-bg); color: var(--muted-color); border: 1px solid var(--border-color); padding: 0.4ch 2ch; font-family: var(--font-mono); font-size: var(--font-size-sm, 0.85rem); cursor: not-allowed; flex-shrink: 0;"
                } else {
                    "background: var(--accent-color, #4af); color: var(--background); border: none; padding: 0.4ch 2ch; font-family: var(--font-mono); font-size: var(--font-size-sm, 0.85rem); cursor: pointer; flex-shrink: 0;"
                },
                disabled: *is_running.read(),
                onclick: run_reasoning,
                if *is_running.read() { "running..." } else { "run pattern" }
            }

            // Error
            if let Some(ref err) = *error_msg.read() {
                div {
                    style: "color: #f44; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem); background: var(--secondary-bg); padding: 0.6ch 1ch; border-left: 2px solid #f44; flex-shrink: 0;",
                    "error: {err}"
                }
            }

            // Results
            if let Some(ref r) = *result.read() {
                div {
                    style: "flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 0.8ch; min-height: 0;",

                    // Summary header
                    div {
                        style: "display: flex; align-items: center; gap: 1ch; flex-shrink: 0;",
                        span {
                            style: if r.success { "color: #4f4; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);" } else { "color: #f44; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);" },
                            if r.success { "[success]" } else { "[failed]" }
                        }
                        span {
                            style: "color: var(--muted-color); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);",
                            "{r.steps.len()} steps"
                        }
                    }

                    // Reasoning steps
                    div {
                        style: "display: flex; flex-direction: column; gap: 0.4ch;",
                        for (idx, step) in r.steps.iter().enumerate() {
                            {
                                let is_expanded = expanded_steps.read().contains(&idx);
                                let step_type = step.step_type.clone();
                                let content = step.content.clone();
                                let tool_name = step.tool_name.clone();
                                let tool_input = step.tool_input.clone();
                                let tool_output = step.tool_output.clone();
                                let has_tool = tool_name.is_some();
                                let step_color = match step_type.as_str() {
                                    "thought" | "reasoning" => "#4af",
                                    "action" | "tool_use" => "#fa4",
                                    "observation" | "tool_result" => "#4f4",
                                    "plan" => "#a4f",
                                    _ => "#888",
                                };
                                rsx! {
                                    div {
                                        style: "border: 1px solid var(--border-color); background: var(--secondary-bg);",
                                        // Step header (always visible)
                                        div {
                                            style: "display: flex; align-items: center; gap: 0.8ch; padding: 0.4ch 0.8ch; cursor: pointer;",
                                            onclick: move |_| {
                                                let mut set = expanded_steps.write();
                                                if set.contains(&idx) { set.remove(&idx); } else { set.insert(idx); }
                                            },
                                            span {
                                                style: "color: {step_color}; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem); min-width: 12ch;",
                                                "[{step_type}]"
                                            }
                                            span {
                                                style: "color: var(--text-color); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1;",
                                                {
                                                    let preview = if content.len() > 80 { format!("{}...", caliche_contracts::text::truncate_utf8(&content, 80)) } else { content.clone() };
                                                    preview
                                                }
                                            }
                                            if has_tool {
                                                span {
                                                    style: "color: #fa4; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);",
                                                    "[tool: {tool_name.as_deref().unwrap_or(\"\")}]"
                                                }
                                            }
                                            span {
                                                style: "color: var(--muted-color); font-size: var(--font-size-xs, 0.75rem);",
                                                if is_expanded { "▲" } else { "▼" }
                                            }
                                        }
                                        // Expanded content
                                        if is_expanded {
                                            div {
                                                style: "padding: 0.6ch 1ch; border-top: 1px solid var(--border-color); display: flex; flex-direction: column; gap: 0.4ch;",
                                                pre {
                                                    style: "margin: 0; color: var(--text-color); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem); white-space: pre-wrap; word-break: break-word;",
                                                    "{content}"
                                                }
                                                if let Some(ref ti) = tool_input {
                                                    div {
                                                        style: "margin-top: 0.4ch;",
                                                        span { style: "color: #fa4; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);", "input: " }
                                                        pre { style: "margin: 0; display: inline; color: var(--muted-color); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem); white-space: pre-wrap;", "{ti}" }
                                                    }
                                                }
                                                if let Some(ref to) = tool_output {
                                                    div {
                                                        style: "margin-top: 0.4ch;",
                                                        span { style: "color: #4f4; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);", "output: " }
                                                        pre { style: "margin: 0; display: inline; color: var(--muted-color); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem); white-space: pre-wrap;", "{to}" }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Final answer
                    div {
                        style: "border: 1px solid var(--accent-color, #4af); background: var(--secondary-bg); padding: 0.8ch 1ch; flex-shrink: 0;",
                        div {
                            style: "color: var(--accent-color, #4af); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem); margin-bottom: 0.4ch;",
                            "final answer"
                        }
                        pre {
                            style: "margin: 0; color: var(--text-color); font-family: var(--font-mono); font-size: var(--font-size-sm, 0.85rem); white-space: pre-wrap; word-break: break-word;",
                            "{r.final_answer}"
                        }
                    }
                }
            }
        }
    }
}

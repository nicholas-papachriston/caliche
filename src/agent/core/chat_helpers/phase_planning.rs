use crate::agent::core::ToolIntelligenceState;
use crate::agent::tools::control::HANDOFF_TOOL_NAME;
use crate::agent::tools::metadata::{metadata_from_schema, order_tool_schemas_by_relevance};
use crate::agent::tools::names::{TOOL_BASH, TOOL_FILESYSTEM};
use crate::agent::tools::{
    AdaptiveSelector, ExecutionPlanner, TaskType, ToolCategory, ToolSelectionContext, ToolSelector,
};
use crate::errors::AgentError;
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::Mutex;

fn query_likely_needs_build_or_test(query: &str) -> bool {
    let q = query.to_lowercase();
    [
        "fix", "issue", "failing", "error", "build", "compile", "test", "lint", "ci", "broken",
    ]
    .iter()
    .any(|k| q.contains(k))
}

/// Heuristic: questions that benefit from running structural commands in phase 1
/// (`wc -l`, `cargo metadata`, `git log`, `find`, etc.) -- typically code review,
/// audit, exploration, or "what's in here?" prompts. Keeping `bash` available
/// from phase 1 avoids forcing the model to fabricate measurements when it
/// could have just measured.
fn query_likely_needs_exploration(query: &str) -> bool {
    let q = query.to_lowercase();
    [
        "improve",
        "improvement",
        "improvements",
        "deep dive",
        "deep-dive",
        "explore",
        "exploration",
        "review",
        "audit",
        "assess",
        "assessment",
        "analyze",
        "analysis",
        "compare",
        "comparison",
        "what's in",
        "whats in",
        "what is in",
        "structure",
        "complexity",
        "areas",
        "metrics",
        "stats",
        "statistics",
        "summarize",
        "summary",
        "overview",
        "walk me through",
        "walk through",
        "tour",
        "how does",
        "how is",
        "how are",
    ]
    .iter()
    .any(|k| q.contains(k))
}

pub async fn ensure_phase_plan_for_query(
    phased_enabled: bool,
    query: &str,
    tool_schemas_all: &Arc<Vec<crate::messages::ToolSchema>>,
    allowed_tools: Option<&Arc<HashSet<String>>>,
    tool_intelligence: &Arc<Mutex<ToolIntelligenceState>>,
) -> Result<(), AgentError> {
    if !phased_enabled {
        return Ok(());
    }

    {
        let state = tool_intelligence.lock().await;
        if state.last_planned_query.as_deref() == Some(query) && state.phases.is_some() {
            return Ok(());
        }
    }

    let ordered_all = order_tool_schemas_by_relevance(tool_schemas_all, query);
    let mut state = tool_intelligence.lock().await;

    let needs_replan = state.last_planned_query.as_deref() != Some(query) || state.phases.is_none();

    if !needs_replan {
        return Ok(());
    }

    let mut metas: Vec<_> = ordered_all.iter().map(metadata_from_schema).collect();
    if let Some(allowed) = allowed_tools {
        metas.retain(|m| allowed.contains(&m.name));
    }

    let ctx = ToolSelectionContext {
        query: query.to_string(),
        available_tools: metas.clone(),
        execution_history: state.execution_history.get_recent_global(200),
        task_type: TaskType::General,
        constraints: Default::default(),
    };
    let selector = AdaptiveSelector::default();
    let selection = selector.select(&ctx);

    let mut selected: HashSet<String> = selection.selected_tools.into_iter().collect();
    // Keep filesystem discovery/editing always available in phased mode when present
    // in the registry. Many task-oriented prompts require local repo inspection first.
    if metas.iter().any(|m| m.name == TOOL_FILESYSTEM) {
        selected.insert(TOOL_FILESYSTEM.to_string());
    }
    // For fix/build/test-style tasks, keep bash available from phase planning so
    // the model can run actual project commands (bun/npm/pnpm/yarn/cargo/etc.).
    // Also keep bash available for analytical/exploration queries -- those benefit
    // from `wc -l`, `cargo metadata`, `git log`, `find`, etc., and forcing the
    // model to wait for phase 2 just makes it guess instead of measure.
    let needs_bash_phase1 = (query_likely_needs_build_or_test(query)
        || query_likely_needs_exploration(query))
        && metas.iter().any(|m| m.name == TOOL_BASH);
    if needs_bash_phase1 {
        selected.insert(TOOL_BASH.to_string());
    }
    selected.insert(HANDOFF_TOOL_NAME.to_string());

    // Build dependencies automatically: execution/analysis tools depend on all information tools.
    let info_tools: HashSet<String> = metas
        .iter()
        .filter(|m| selected.contains(&m.name) && matches!(m.category, ToolCategory::Information))
        .map(|m| m.name.clone())
        .collect();

    let mut selected_metas = Vec::new();
    for mut meta in metas {
        if !selected.contains(&meta.name) {
            continue;
        }
        if matches!(
            meta.category,
            ToolCategory::Execution | ToolCategory::Analysis
        ) && !info_tools.is_empty()
        {
            let mut deps = meta.dependencies.clone();
            deps.requires_prior_tools.extend(info_tools.iter().cloned());
            meta.dependencies = deps;
        }
        selected_metas.push(meta);
    }

    let phases = ExecutionPlanner::plan(&selected_metas).map_or_else(
        |_| vec![selected.iter().cloned().collect()],
        |p| p.phases.into_iter().map(|ph| ph.tools).collect::<Vec<_>>(),
    );

    state.last_planned_query = Some(query.to_string());
    state.phases = Some(phases);
    state.phase_idx = 0;

    Ok(())
}

pub async fn current_phase_allowed_tools(
    phased_enabled: bool,
    tool_intelligence: &Arc<Mutex<ToolIntelligenceState>>,
) -> Option<HashSet<String>> {
    if !phased_enabled {
        return None;
    }

    let state = tool_intelligence.lock().await;
    let mut allowed = HashSet::new();
    if let Some(phases) = state.phases.as_ref()
        && let Some(phase) = phases.get(state.phase_idx)
    {
        allowed.extend(phase.iter().cloned());
    }
    allowed.insert(HANDOFF_TOOL_NAME.to_string());
    Some(allowed)
}

/// Return (`phase_index`, `phase_total`, `tool_names`) for the current phase when phased execution is enabled.
pub async fn get_current_phase_display(
    tool_intelligence: &Arc<Mutex<ToolIntelligenceState>>,
) -> Option<(usize, usize, Vec<String>)> {
    let state = tool_intelligence.lock().await;
    let phases = state.phases.as_ref()?;
    let phase = phases.get(state.phase_idx)?;
    let mut tools: Vec<String> = phase.clone();
    tools.retain(|t| t != HANDOFF_TOOL_NAME);
    Some((state.phase_idx, phases.len(), tools))
}

pub async fn advance_phase_if_needed(
    phased_enabled: bool,
    tool_intelligence: &Arc<Mutex<ToolIntelligenceState>>,
) -> bool {
    if !phased_enabled {
        return false;
    }

    let mut advanced = false;
    let mut state = tool_intelligence.lock().await;
    if let Some(phases) = state.phases.as_ref() {
        if state.phase_idx + 1 < phases.len() {
            state.phase_idx += 1;
            advanced = true;
        } else {
            state.reset_plan();
            state.last_planned_query = None;
        }
    }
    advanced
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_or_test_keywords_match() {
        for q in [
            "fix the failing test",
            "build is broken",
            "lint errors in CI",
            "compile error",
        ] {
            assert!(
                query_likely_needs_build_or_test(q),
                "expected match for {q:?}"
            );
        }
    }

    #[test]
    fn build_or_test_keywords_dont_overmatch() {
        for q in ["what areas can it improve in?", "deep dive on shotput"] {
            assert!(
                !query_likely_needs_build_or_test(q),
                "expected NO build/test match for {q:?}"
            );
        }
    }

    #[test]
    fn exploration_keywords_match() {
        for q in [
            "what areas can it improve in?",
            "do a deep dive on shotput",
            "review this module",
            "audit the auth flow",
            "analyze complexity of the builder",
            "compare these two backends",
            "what's in this repo?",
            "give me an overview",
            "walk me through the runtime",
            "how does the planner work",
            "summarize the changes",
        ] {
            assert!(
                query_likely_needs_exploration(q),
                "expected exploration match for {q:?}"
            );
        }
    }

    #[test]
    fn exploration_keywords_dont_overmatch_trivial_questions() {
        for q in ["hi", "ok", "thanks", "cancel that"] {
            assert!(
                !query_likely_needs_exploration(q),
                "expected NO exploration match for {q:?}"
            );
        }
    }

    #[tokio::test]
    async fn unchanged_query_does_not_rebuild_phase_plan() {
        let mut state = ToolIntelligenceState::new();
        state.last_planned_query = Some("fix the failing test".into());
        state.phases = Some(vec![vec!["filesystem".into()], vec!["bash".into()]]);
        state.phase_idx = 1;
        let intelligence = Arc::new(Mutex::new(state));
        let schemas = Arc::new(Vec::new());
        ensure_phase_plan_for_query(true, "fix the failing test", &schemas, None, &intelligence)
            .await
            .expect("existing plan should be reused");
        let locked = intelligence.lock().await;
        assert_eq!(locked.phase_idx, 1);
        assert_eq!(locked.phases.as_ref().map(Vec::len), Some(2));
        assert_eq!(
            locked.last_planned_query.as_deref(),
            Some("fix the failing test")
        );
    }
}

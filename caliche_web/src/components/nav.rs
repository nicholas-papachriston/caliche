//! Primary tab grouping for the shell. Related surfaces share one top-level tab
//! and a pane strip so the bar stays short.

use super::app_shell::ActiveTab;

/// Top-level navigation group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TabGroup {
    Work,
    Agents,
    Inspect,
    System,
}

/// Group that owns `tab`.
#[must_use]
pub fn tab_group(tab: ActiveTab) -> TabGroup {
    match tab {
        ActiveTab::Chat | ActiveTab::History => TabGroup::Work,
        ActiveTab::Agents | ActiveTab::Orchestration | ActiveTab::Workflow => TabGroup::Agents,
        ActiveTab::Metrics | ActiveTab::Reasoning | ActiveTab::Documents | ActiveTab::Chains => {
            TabGroup::Inspect
        }
        ActiveTab::Plugins | ActiveTab::Settings => TabGroup::System,
    }
}

/// True when the primary button for `button` should look active.
#[must_use]
pub fn is_primary_active(button: ActiveTab, current: ActiveTab) -> bool {
    match button {
        ActiveTab::Agents => matches!(tab_group(current), TabGroup::Agents),
        ActiveTab::Metrics => matches!(tab_group(current), TabGroup::Inspect),
        other => current == other,
    }
}

/// Inner panes for a group. Empty when the group has no pane strip.
#[must_use]
pub fn panes_for(group: TabGroup) -> &'static [ActiveTab] {
    match group {
        TabGroup::Agents => &[
            ActiveTab::Agents,
            ActiveTab::Orchestration,
            ActiveTab::Workflow,
        ],
        TabGroup::Inspect => &[
            ActiveTab::Metrics,
            ActiveTab::Reasoning,
            ActiveTab::Documents,
            ActiveTab::Chains,
        ],
        TabGroup::Work | TabGroup::System => &[],
    }
}

/// Lowercase pane or tab label.
#[must_use]
pub fn pane_label(tab: ActiveTab) -> &'static str {
    match tab {
        ActiveTab::Chat => "chat",
        ActiveTab::History => "history",
        ActiveTab::Agents => "fleet",
        ActiveTab::Orchestration => "orchestration",
        ActiveTab::Workflow => "workflow",
        ActiveTab::Metrics => "metrics",
        ActiveTab::Reasoning => "reasoning",
        ActiveTab::Documents => "documents",
        ActiveTab::Chains => "chains",
        ActiveTab::Plugins => "plugins",
        ActiveTab::Settings => "settings",
    }
}

/// Which feature-gated inspect panes the connected server actually serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InspectPaneFlags {
    pub reasoning: bool,
    pub documents: bool,
    pub chains: bool,
}

impl InspectPaneFlags {
    pub const ALL: Self = Self {
        reasoning: true,
        documents: true,
        chains: true,
    };

    #[must_use]
    pub fn allows(self, tab: ActiveTab) -> bool {
        match tab {
            ActiveTab::Reasoning => self.reasoning,
            ActiveTab::Documents => self.documents,
            ActiveTab::Chains => self.chains,
            _ => true,
        }
    }
}

impl Default for InspectPaneFlags {
    fn default() -> Self {
        Self::ALL
    }
}

/// Panes to show. Inspect drops panes whose RPC is method-not-found.
#[must_use]
pub fn visible_panes(group: TabGroup, inspect: InspectPaneFlags) -> Vec<ActiveTab> {
    panes_for(group)
        .iter()
        .copied()
        .filter(|tab| inspect.allows(*tab))
        .collect()
}

/// Open `target` unless that primary group is already active.
#[must_use]
pub fn select_primary(current: ActiveTab, target: ActiveTab) -> ActiveTab {
    if is_primary_active(target, current) {
        current
    } else {
        target
    }
}

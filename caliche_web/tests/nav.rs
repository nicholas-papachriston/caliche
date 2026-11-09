//! Primary tab grouping tests.

use caliche_web::components::app_shell::ActiveTab;
use caliche_web::components::nav::{
    InspectPaneFlags, TabGroup, is_primary_active, pane_label, panes_for, select_primary,
    tab_group, visible_panes,
};

#[test]
fn groups_coordination_surfaces_under_agents() {
    assert_eq!(tab_group(ActiveTab::Agents), TabGroup::Agents);
    assert_eq!(tab_group(ActiveTab::Orchestration), TabGroup::Agents);
    assert_eq!(tab_group(ActiveTab::Workflow), TabGroup::Agents);
}

#[test]
fn groups_inspect_surfaces_together() {
    assert_eq!(tab_group(ActiveTab::Metrics), TabGroup::Inspect);
    assert_eq!(tab_group(ActiveTab::Reasoning), TabGroup::Inspect);
    assert_eq!(tab_group(ActiveTab::Documents), TabGroup::Inspect);
    assert_eq!(tab_group(ActiveTab::Chains), TabGroup::Inspect);
}

#[test]
fn primary_agents_tab_stays_active_on_inner_panes() {
    assert!(is_primary_active(
        ActiveTab::Agents,
        ActiveTab::Orchestration
    ));
    assert!(is_primary_active(ActiveTab::Agents, ActiveTab::Workflow));
    assert!(!is_primary_active(ActiveTab::Agents, ActiveTab::Chat));
    assert!(is_primary_active(ActiveTab::Metrics, ActiveTab::Documents));
    assert!(!is_primary_active(ActiveTab::Chat, ActiveTab::History));
}

#[test]
fn pane_strips_cover_grouped_tabs() {
    assert_eq!(panes_for(TabGroup::Agents).len(), 3);
    assert_eq!(panes_for(TabGroup::Inspect).len(), 4);
    assert!(panes_for(TabGroup::Work).is_empty());
    assert_eq!(pane_label(ActiveTab::Agents), "fleet");
    let hidden = InspectPaneFlags {
        reasoning: false,
        documents: false,
        chains: false,
    };
    assert_eq!(
        visible_panes(TabGroup::Inspect, hidden),
        vec![ActiveTab::Metrics]
    );
    assert_eq!(
        visible_panes(TabGroup::Inspect, InspectPaneFlags::ALL).len(),
        4
    );
    assert_eq!(
        select_primary(ActiveTab::Orchestration, ActiveTab::Agents),
        ActiveTab::Orchestration
    );
    assert_eq!(
        select_primary(ActiveTab::Chat, ActiveTab::Agents),
        ActiveTab::Agents
    );
}

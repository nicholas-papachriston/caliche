//! Fleet view — unified view of managed endpoints and running sub-agent tasks.

mod broadcast;
mod broadcast_run;
mod endpoint_launch;
mod endpoint_row;
mod endpoints_section;
mod harness_form;
mod harness_row;
mod harnesses_section;
mod helpers;
mod spawn_form;
mod stats_bar;
mod tab_label;
mod task_row;
mod tasks_section;
mod types;
mod view;

pub use tab_label::fleet_tab_label;
pub use types::{FleetViewProps, SpawnRequest};
pub use view::FleetView;

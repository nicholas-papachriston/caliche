//! UI components module.

pub mod app_shell;
pub mod chains_view;
pub mod chat_input;
pub mod chat_view;
pub mod chrome;
pub mod documents_view;
pub mod file_browser;
pub mod fleet_view;
pub mod header;
pub mod history_view;
pub mod message;
pub mod message_list;
pub mod metrics_view;
pub mod nav;
pub mod orchestration_coordinator;
pub mod orchestration_view;
pub mod permissions_view;
pub mod plugins_view;
pub mod reasoning_view;
pub mod session_panel;
pub mod settings_view;
pub mod status_bar;
pub mod tool_call;
pub mod tunnel_view;
pub mod workflow_view;

pub use chains_view::ChainsView;
pub use chat_view::ChatView;
pub use documents_view::DocumentsView;
#[allow(unused_imports)]
pub use file_browser::{AttachedFile, FileBrowser};
pub use fleet_view::{FleetView, SpawnRequest};
pub use header::{Header, SimpleHeader};
pub use history_view::HistoryView;
pub use metrics_view::MetricsView;
pub use orchestration_view::OrchestrationView;
pub use plugins_view::PluginsView;
pub use reasoning_view::ReasoningView;
pub use session_panel::SessionPanel;
pub use settings_view::SettingsView;
pub use status_bar::StatusBar;
pub use workflow_view::WorkflowView;

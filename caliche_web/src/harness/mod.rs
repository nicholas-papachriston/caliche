//! Generic harness catalog and draft validation for the Agents tab.
//!
//! Presets fill a standard form so operators can add Claude Code, Codex,
//! Cursor, OpenCode, Pi, Hermes, or a custom CLI without a separate manual.

mod catalog;
mod draft;
mod placement;

pub use catalog::{HARNESS_PRESETS, preset_by_id, toggle_csv_tag, unique_capabilities};
pub use draft::HarnessDraft;
pub use placement::PromptPlacementKind;

#[cfg(any(test, feature = "test-utils"))]
#[allow(unused_imports)]
pub use catalog::{FORBIDDEN_SHELLS, HarnessPreset, command_is_forbidden_shell};
#[cfg(any(test, feature = "test-utils"))]
#[allow(unused_imports)]
pub use draft::{CreateHarnessInput, parse_arg_lines, parse_csv_list};

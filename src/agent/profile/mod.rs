//! Named agent profiles: Markdown files with YAML frontmatter.
//!
//! Profiles live in `.caliche/agents/` (project) or `~/.config/caliche/agents/`
//! and are keyed by filename stem (e.g. `reviewer.md` -> `reviewer`).

mod loader;
mod profile;

pub use loader::{
    AGENTS_DIR_NAME, ProfileDir, default_search_path, load_profile_file, load_profile_from_str,
    load_profiles_from_search_path,
};
pub use profile::{AgentProfile, ProfilePermission};

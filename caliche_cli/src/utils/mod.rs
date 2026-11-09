//! Shared utilities used across the CLI.

mod path;
mod temp_file;

pub use path::find_git_root;
pub use temp_file::{create_private_temp_file, temp_file_path};

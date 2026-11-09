//! Styles module - Monospace Web design inspired by https://owickstrom.github.io/the-monospace-web/

mod chat;
mod forms;
mod layout;
mod reset;
mod responsive;
mod tabs;
mod tokens;

use std::sync::OnceLock;

static BASE_STYLES_ONCE: OnceLock<&'static str> = OnceLock::new();

/// Full stylesheet composed from section modules.
pub fn base_styles() -> &'static str {
    BASE_STYLES_ONCE.get_or_init(|| {
        let combined = [
            tokens::STYLES,
            reset::STYLES,
            layout::SHELL,
            chat::MESSAGES,
            forms::STYLES,
            layout::SCROLLBAR,
            chat::EXTENDED,
            tabs::STYLES,
            responsive::STYLES,
        ]
        .concat();
        Box::leak(combined.into_boxed_str())
    })
}

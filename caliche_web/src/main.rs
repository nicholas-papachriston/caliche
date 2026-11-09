//! Caliche Cross-Platform Application
//!
//! A Dioxus-based UI for Caliche supporting Desktop, Web, and Mobile.

use dioxus::prelude::*;

mod api;
mod app;
mod catalog_select;
mod commands;
mod components;
mod context;
mod harness;
mod hooks;
mod state;
mod styles;

use app::App;
use styles::base_styles;

fn main() {
    // Platform-specific initialization
    init_tracing();

    // Launch the application
    tracing::info!("Launching Dioxus app...");
    dioxus::launch(AppWithStyles);
}

/// Initialize tracing based on platform
fn init_tracing() {
    #[cfg(target_arch = "wasm32")]
    {
        // Web initialization - set panic hook first
        console_error_panic_hook::set_once();

        // Initialize tracing for browser console with info level (less verbose)
        tracing_wasm::set_as_global_default_with_config(
            tracing_wasm::WASMLayerConfigBuilder::default()
                .set_max_level(tracing::Level::INFO)
                .build(),
        );

        tracing::info!("Starting Caliche (Web)");
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "tracing-subscriber"))]
    {
        // Desktop/Mobile initialization with tracing_subscriber
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .init();

        #[cfg(feature = "desktop")]
        tracing::info!("Starting Caliche (Desktop)");

        #[cfg(feature = "mobile")]
        tracing::info!("Starting Caliche (Mobile)");
    }

    #[cfg(all(not(target_arch = "wasm32"), not(feature = "tracing-subscriber")))]
    {
        // Fallback - no tracing setup
        // Tracing macros will be no-ops
    }
}

/// Wrapper component that includes styles
#[component]
fn AppWithStyles() -> Element {
    rsx! {
        // Inject base styles
        style { "{base_styles()}" }

        // Main app
        App {}
    }
}

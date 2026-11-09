//! Tunnel notice. The JSON-RPC client has no tunnel methods, so the settings
//! tab does not offer a form that always fails.

use dioxus::prelude::*;

/// Props for the tunnel notice.
#[derive(Props, Clone, PartialEq)]
pub struct TunnelViewProps {
    /// API address the operator is connected to.
    pub api_base_url: String,
}

/// Honest copy in place of the Cloudflare tunnel form.
#[component]
pub fn TunnelView(props: TunnelViewProps) -> Element {
    rsx! {
        div { style: "color: var(--muted-color); font-size: var(--font-size-sm); max-width: 72ch; line-height: 1.4;",
            "Tunnels are not available in JSON-RPC-only mode. This console talks to {props.api_base_url} and cannot start a Cloudflare tunnel from here."
        }
    }
}

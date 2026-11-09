mod cli;
mod config;
pub mod security;
mod shared;
mod vault;

pub(crate) use cli::*;
pub(crate) use config::*;
pub use security::{
    is_credential_route, is_credential_route_accessible, require_loopback_for_credential_routes,
};
pub(crate) use vault::*;

use axum::routing::{get, post};

pub(crate) fn add_cli_parity_routes(router: axum::Router) -> axum::Router {
    router
        .route("/api/cli/completions/{shell}", get(completions_handler))
        .route("/api/cli/man", get(man_page_handler))
        .route("/api/cli/man/all", get(man_pages_handler))
        .route("/api/cli/lsp/status", get(lsp_status_handler))
        .route("/api/cli/lsp/languages", get(lsp_languages_handler))
        .route("/api/cli/lsp/install", post(lsp_install_handler))
        .route("/api/cli/agents", get(agent_profiles_handler))
        .route("/api/cli/agents/{name}", get(agent_profile_handler))
        .route("/api/cli/doctor", get(doctor_handler))
        .route("/api/cli/config", get(config_show_handler))
        .route("/api/cli/config/schema", get(config_schema_handler))
        .route("/api/cli/config/path", get(config_path_handler))
        .route("/api/cli/config/providers", get(config_providers_handler))
        .route(
            "/api/cli/config/provider",
            post(config_set_provider_handler),
        )
        .route(
            "/api/cli/config/provider/{provider}",
            get(config_get_provider_handler).delete(config_remove_provider_handler),
        )
        .route(
            "/api/cli/config/default-provider",
            get(config_default_provider_handler).put(config_set_default_provider_handler),
        )
        .route(
            "/api/cli/config/default-model",
            post(config_set_default_model_handler),
        )
        .route(
            "/api/cli/config/gateway",
            get(config_gateway_credentials_handler),
        )
        .route(
            "/api/cli/config/gateway/{name}",
            get(config_get_gateway_credential_handler)
                .delete(config_remove_gateway_credential_handler),
        )
        .route(
            "/api/cli/config/gateway",
            post(config_set_gateway_credential_handler),
        )
        .route("/api/cli/config/export/{shell}", get(config_export_handler))
        .route("/api/cli/vault/status", get(vault_status_handler))
        .route("/api/cli/vault/scopes", get(vault_scopes_handler))
        .route("/api/cli/vault/credentials", get(vault_credentials_handler))
        .route("/api/cli/vault/link", post(vault_link_handler))
        .route("/api/cli/vault/unlink", post(vault_unlink_handler))
        .route("/api/cli/vault/import", post(vault_import_handler))
        .route("/api/cli/vault/promote", post(vault_promote_handler))
}

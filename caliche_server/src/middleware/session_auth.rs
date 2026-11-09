//! Session ownership checks to prevent IDOR on `/api/agent/sessions/*`.

use crate::adapter::{ApiAdapter, SessionOwner};
use crate::middleware::context::UserContext;
use crate::middleware::jwt_auth::{AUTH_REQUIRED_MESSAGE, JwtAuthState};
use crate::security::SecurityPolicy;
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

const SESSION_ACCESS_DENIED: &str = "session access denied";
const SESSION_NOT_FOUND: &str = "session not found";

/// Extract the session id segment from a nested sessions route path.
pub fn session_id_from_path(path: &str) -> Option<String> {
    let normalized = path.trim_end_matches('/');
    let prefix = "/api/agent/sessions";
    if normalized == prefix {
        return None;
    }
    let rest = normalized.strip_prefix(&format!("{prefix}/"))?;
    let segment = rest.split('/').next()?;
    if segment.is_empty() {
        None
    } else {
        Some(segment.to_string())
    }
}

/// Verify the caller may access `session_id` when session auth is enforced.
pub async fn authorize_session_access(
    adapter: &ApiAdapter,
    session_id: &str,
    user: &UserContext,
    policy: &SecurityPolicy,
) -> Result<(), (StatusCode, String)> {
    if !policy.session_auth_required() {
        return Ok(());
    }

    let owner = adapter
        .session_owner(session_id)
        .await
        .ok_or_else(|| (StatusCode::NOT_FOUND, SESSION_NOT_FOUND.to_string()))?;

    verify_session_ownership(&owner, user)
}

/// Enforce that `user` matches recorded session owner metadata (IDOR guard).
pub fn verify_session_ownership(
    owner: &SessionOwner,
    user: &UserContext,
) -> Result<(), (StatusCode, String)> {
    let Some(expected_user) = owner.user_id.as_deref() else {
        return Err((StatusCode::FORBIDDEN, SESSION_ACCESS_DENIED.to_string()));
    };

    if expected_user != user.user_id {
        return Err((StatusCode::FORBIDDEN, SESSION_ACCESS_DENIED.to_string()));
    }

    if let (Some(session_tenant), Some(user_tenant)) = (&owner.tenant_id, &user.tenant_id)
        && session_tenant != user_tenant
    {
        return Err((StatusCode::FORBIDDEN, SESSION_ACCESS_DENIED.to_string()));
    }

    Ok(())
}

/// Convenience wrapper for handlers that already hold an [`Arc<ApiAdapter>`].
pub async fn authorize_session_arc(
    adapter: &Arc<ApiAdapter>,
    session_id: &str,
    user: &UserContext,
    policy: &SecurityPolicy,
) -> Result<(), (StatusCode, String)> {
    authorize_session_access(adapter.as_ref(), session_id, user, policy).await
}

/// Enforce session ownership when auth is enabled; no-op in development mode.
pub async fn maybe_authorize_session_optional(
    adapter: &Arc<ApiAdapter>,
    session_id: &str,
    user: Option<&UserContext>,
) -> Result<(), (StatusCode, String)> {
    if !adapter.security().session_auth_required() {
        return Ok(());
    }
    let user = user.ok_or_else(|| (StatusCode::UNAUTHORIZED, AUTH_REQUIRED_MESSAGE.to_string()))?;
    authorize_session_arc(adapter, session_id, user, adapter.security()).await
}

/// Route-layer middleware: authorize session-scoped HTTP routes once per request.
pub async fn session_route_auth_middleware(
    State(adapter): State<Arc<ApiAdapter>>,
    mut request: Request,
    next: Next,
) -> Response {
    let policy = adapter.security();
    if !policy.session_auth_required() {
        return next.run(request).await;
    }

    let path = request.uri().path();
    let Some(session_id) = session_id_from_path(path) else {
        return next.run(request).await;
    };

    let Some(auth_state) = request.extensions().get::<JwtAuthState>().cloned() else {
        return (StatusCode::UNAUTHORIZED, AUTH_REQUIRED_MESSAGE).into_response();
    };

    let user = match auth_state.extract_user_context(request.headers()).await {
        Ok(Some(user)) => user,
        Ok(None) | Err(_) => {
            return (StatusCode::UNAUTHORIZED, AUTH_REQUIRED_MESSAGE).into_response();
        }
    };

    if let Err((status, msg)) = authorize_session_arc(&adapter, &session_id, &user, policy).await {
        return (status, msg).into_response();
    }

    request.extensions_mut().insert(user);
    next.run(request).await
}

/// Owner metadata captured when a session is created under enforced auth.
pub fn owner_from_user(user: &UserContext) -> SessionOwner {
    SessionOwner {
        user_id: Some(user.user_id.clone()),
        tenant_id: user.tenant_id.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::session_id_from_path;

    #[test]
    fn collection_path_has_no_session_id() {
        assert!(session_id_from_path("/api/agent/sessions").is_none());
        assert!(session_id_from_path("/api/agent/sessions/").is_none());
    }

    #[test]
    fn cancel_stop_and_turn_share_session_id_extraction() {
        assert_eq!(
            session_id_from_path("/api/agent/sessions/abc/turn").as_deref(),
            Some("abc")
        );
        assert_eq!(
            session_id_from_path("/api/agent/sessions/abc/cancel").as_deref(),
            Some("abc")
        );
        assert_eq!(
            session_id_from_path("/api/agent/sessions/abc/stop").as_deref(),
            Some("abc")
        );
    }
}

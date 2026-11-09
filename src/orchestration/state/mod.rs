use anyhow::Result;
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use serde_json::{Value, json};
use tokio::sync::RwLock;

use std::sync::Arc;

use crate::utils::json_merge::{JsonMergeMode, merge_json_value};

pub mod acl;

pub use acl::{
    AGENT_NAMESPACE_ROOT, StateWriteDecision, StateWritePolicy, agent_namespace, namespaced_path,
    sanitize_agent_id,
};

fn ensure_object_segment<'a>(
    map: &'a mut serde_json::Map<String, Value>,
    key: &str,
) -> Result<&'a mut serde_json::Map<String, Value>> {
    if !map.get(key).is_some_and(|v| v.is_object()) {
        map.insert(key.to_string(), json!({}));
    }
    map.get_mut(key)
        .ok_or_else(|| anyhow::anyhow!("Failed to get segment '{}'", key))?
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("Segment '{}' is not an object", key))
}

/// Inner storage for [`SharedState`], shared across clones via `Arc`.
///
/// The state is partitioned by top-level key (`entries`) instead of being one
/// `RwLock<Value>` over the whole document: concurrent `get`/`set`/`merge` calls that touch
/// different top-level keys (e.g. `"/customer/..."` vs `"/order/..."`) run against different
/// `DashMap` shards and don't contend with each other, whereas a single whole-document lock
/// would serialize every write against every other read/write regardless of which branch of
/// the tree was touched.
///
/// `scalar_root` covers the narrow edge case where the logical root is not a JSON object
/// (constructed via `from_value` with a scalar/array, or explicitly replaced via
/// `set("/", scalar)`); while it holds `Some(_)`, `entries` is unused and the whole value
/// lives there instead.
struct SharedStateInner {
    scalar_root: RwLock<Option<Value>>,
    entries: DashMap<String, Value>,
    /// Per-agent write rules. Only the `*_as` write methods consult it, so an unattributed
    /// write keeps the previous unrestricted behavior.
    write_policy: StateWritePolicy,
}

impl SharedStateInner {
    fn reconstruct_object(&self) -> Value {
        let map: serde_json::Map<String, Value> = self
            .entries
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();
        Value::Object(map)
    }

    /// Replace the whole root, choosing object-partitioned or scalar storage based on
    /// `value`'s shape.
    async fn replace_root(&self, value: Value) {
        match value {
            Value::Object(map) => {
                let mut scalar = self.scalar_root.write().await;
                *scalar = None;
                self.entries.clear();
                for (key, val) in map {
                    self.entries.insert(key, val);
                }
            }
            other => {
                let mut scalar = self.scalar_root.write().await;
                self.entries.clear();
                *scalar = Some(other);
            }
        }
    }

    /// If the root is currently a non-object scalar/array, coerce it to an empty object so
    /// nested `set` calls have somewhere to write — matching the pre-existing behavior of
    /// `SharedState::set`, which reset a non-object root to `{}` before writing a nested path.
    async fn ensure_object_root(&self) {
        let mut scalar = self.scalar_root.write().await;
        if scalar.is_some() {
            *scalar = None;
            self.entries.clear();
        }
    }
}

/// Thread-safe shared state container for agents
/// Allows multiple agents to share structured application state
#[derive(Clone)]
pub struct SharedState {
    inner: Arc<SharedStateInner>,
}

impl std::fmt::Debug for SharedState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedState")
            .field("inner", &"Arc<SharedStateInner>")
            .finish()
    }
}

impl SharedState {
    /// Create a new empty shared state
    pub fn new() -> Self {
        Self::with_write_policy(StateWritePolicy::allow_all())
    }

    /// Create empty shared state governed by a per-agent write policy.
    ///
    /// The policy applies to [`Self::set_as`], [`Self::merge_as`], and the namespace helpers.
    /// The unattributed `set`/`merge`/`clear` methods stay unrestricted, because they are the
    /// orchestrator's own writes rather than an agent's.
    pub fn with_write_policy(write_policy: StateWritePolicy) -> Self {
        Self {
            inner: Arc::new(SharedStateInner {
                scalar_root: RwLock::new(None),
                entries: DashMap::new(),
                write_policy,
            }),
        }
    }

    /// Create shared state from an initial value
    pub fn from_value(initial: Value) -> Self {
        let (scalar_root, entries) = match initial {
            Value::Object(map) => {
                let entries = DashMap::new();
                for (key, val) in map {
                    entries.insert(key, val);
                }
                (None, entries)
            }
            other => (Some(other), DashMap::new()),
        };
        Self {
            inner: Arc::new(SharedStateInner {
                scalar_root: RwLock::new(scalar_root),
                entries,
                write_policy: StateWritePolicy::allow_all(),
            }),
        }
    }

    /// Get a value from the state using a JSON pointer path (e.g., "/order_id" or "/customer/name")
    pub async fn get(&self, path: &str) -> Result<Option<Value>> {
        {
            let scalar = self.inner.scalar_root.read().await;
            if let Some(root) = scalar.as_ref() {
                return if path.is_empty() || path == "/" {
                    Ok(Some(root.clone()))
                } else {
                    Err(anyhow::anyhow!("State root must be an object"))
                };
            }
        }

        if path.is_empty() || path == "/" {
            return Ok(Some(self.inner.reconstruct_object()));
        }

        let path = path.strip_prefix('/').unwrap_or(path);
        let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

        let Some((top_key, rest)) = segments.split_first() else {
            return Ok(None);
        };

        let Some(top_value) = self.inner.entries.get(*top_key) else {
            return Ok(None);
        };

        if rest.is_empty() {
            return Ok(Some(top_value.clone()));
        }

        let mut current = match top_value.as_object() {
            Some(obj) => obj,
            None => return Ok(None),
        };

        for (i, segment) in rest.iter().enumerate() {
            if i == rest.len() - 1 {
                return Ok(current.get(*segment).cloned());
            }
            match current.get(*segment) {
                Some(Value::Object(obj)) => current = obj,
                _ => return Ok(None),
            }
        }

        Ok(None)
    }

    /// Set a value at a JSON pointer path
    pub async fn set(&self, path: &str, value: Value) -> Result<()> {
        if path.is_empty() || path == "/" {
            self.inner.replace_root(value).await;
            return Ok(());
        }

        let path = path.strip_prefix('/').unwrap_or(path);
        let segments: Vec<String> = path
            .split('/')
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();

        if segments.is_empty() {
            self.inner.replace_root(value).await;
            return Ok(());
        }

        self.inner.ensure_object_root().await;

        let top_key = segments[0].clone();

        if segments.len() == 1 {
            self.inner.entries.insert(top_key, value);
            return Ok(());
        }

        let mut top_entry = self
            .inner
            .entries
            .entry(top_key)
            .or_insert_with(|| json!({}));
        if !top_entry.is_object() {
            *top_entry = json!({});
        }

        let mut current = top_entry
            .as_object_mut()
            .ok_or_else(|| anyhow::anyhow!("State must be an object to set nested paths"))?;

        for segment in segments.iter().skip(1).take(segments.len() - 2) {
            current = ensure_object_segment(current, segment)?;
        }

        let final_segment = segments
            .last()
            .ok_or_else(|| anyhow::anyhow!("Segments list is empty"))?;
        current.insert(final_segment.clone(), value);

        Ok(())
    }

    /// Merge `value` into the root state using a deep object merge.
    ///
    /// Object fields are merged recursively. Arrays and scalars at conflicting paths are
    /// replaced by the incoming value. If the root state is not an object, it is replaced
    /// entirely by `value`. When `value` is itself an object, only the top-level keys it
    /// mentions are touched (each independently, via its own `DashMap` entry), so a merge
    /// into one part of the tree does not contend with concurrent `get`/`set`/`merge` calls
    /// touching unrelated top-level keys.
    pub async fn merge(&self, value: Value) -> Result<()> {
        let Value::Object(incoming) = value else {
            self.inner.replace_root(value).await;
            return Ok(());
        };

        self.inner.ensure_object_root().await;

        for (key, incoming_value) in incoming {
            match self.inner.entries.entry(key) {
                Entry::Occupied(mut occupied) => {
                    merge_json_value(occupied.get_mut(), incoming_value, JsonMergeMode::Deep);
                }
                Entry::Vacant(vacant) => {
                    vacant.insert(incoming_value);
                }
            }
        }

        Ok(())
    }

    /// Get the entire state as a clone
    pub async fn get_all(&self) -> Value {
        {
            let scalar = self.inner.scalar_root.read().await;
            if let Some(root) = scalar.as_ref() {
                return root.clone();
            }
        }
        self.inner.reconstruct_object()
    }

    /// Clear the state
    pub async fn clear(&self) {
        let mut scalar = self.inner.scalar_root.write().await;
        *scalar = None;
        self.inner.entries.clear();
    }

    /// The write policy in force.
    pub fn write_policy(&self) -> &StateWritePolicy {
        &self.inner.write_policy
    }

    /// Whether `agent_id` may write `path`.
    pub fn can_write(&self, agent_id: &str, path: &str) -> bool {
        self.inner
            .write_policy
            .evaluate(agent_id, path)
            .is_allowed()
    }

    /// Set a value on behalf of `agent_id`, subject to the write policy.
    pub async fn set_as(&self, agent_id: &str, path: &str, value: Value) -> Result<()> {
        self.inner.write_policy.authorize(agent_id, path)?;
        self.set(path, value).await
    }

    /// Merge a value on behalf of `agent_id`, subject to the write policy.
    ///
    /// Each top-level key of `value` is authorized independently, so a merge cannot slip a
    /// write past the policy by bundling several branches into one call. The check runs over
    /// every key before anything is written, which keeps a refused merge from applying part
    /// of its payload.
    pub async fn merge_as(&self, agent_id: &str, value: Value) -> Result<()> {
        let Value::Object(ref incoming) = value else {
            // A non-object merge replaces the whole root, so it needs root authority.
            self.inner.write_policy.authorize(agent_id, "/")?;
            return self.merge(value).await;
        };
        for key in incoming.keys() {
            self.inner
                .write_policy
                .authorize(agent_id, &format!("/{key}"))?;
        }
        self.merge(value).await
    }

    /// Set a value inside the agent's private namespace branch (`/agents/<agent_id>/...`).
    pub async fn set_in_namespace(
        &self,
        agent_id: &str,
        relative_path: &str,
        value: Value,
    ) -> Result<()> {
        let path = namespaced_path(agent_id, relative_path);
        self.set_as(agent_id, &path, value).await
    }

    /// Read a value from the agent's private namespace branch.
    pub async fn get_in_namespace(
        &self,
        agent_id: &str,
        relative_path: &str,
    ) -> Result<Option<Value>> {
        let path = namespaced_path(agent_id, relative_path);
        self.get(&path).await
    }
}

impl Default for SharedState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_get_set() {
        let state = SharedState::new();

        state
            .set("/order_id", json!("12345"))
            .await
            .expect("Should set order_id");
        let value = state.get("/order_id").await.expect("Should get order_id");
        assert_eq!(value, Some(json!("12345")));
    }

    #[tokio::test]
    async fn test_nested_path() {
        let state = SharedState::new();

        state
            .set("/customer/name", json!("John"))
            .await
            .expect("Should set customer name");
        state
            .set("/customer/email", json!("john@example.com"))
            .await
            .expect("Should set customer email");

        let name = state
            .get("/customer/name")
            .await
            .expect("Should get customer name");
        assert_eq!(name, Some(json!("John")));

        let email = state
            .get("/customer/email")
            .await
            .expect("Should get customer email");
        assert_eq!(email, Some(json!("john@example.com")));
    }

    #[tokio::test]
    async fn test_merge() {
        let state = SharedState::new();

        state
            .set("/customer/name", json!("John"))
            .await
            .expect("Should set customer name");
        state
            .merge(json!({"customer": {"email": "john@example.com"}}))
            .await
            .expect("Should merge customer data");

        let name = state
            .get("/customer/name")
            .await
            .expect("Should get customer name");
        assert_eq!(name, Some(json!("John")));

        let email = state
            .get("/customer/email")
            .await
            .expect("Should get customer email");
        assert_eq!(email, Some(json!("john@example.com")));
    }

    #[tokio::test]
    async fn test_merge_deeply_nested_objects() {
        let state = SharedState::from_value(json!({"a": {"b": {"x": 1}}}));
        state
            .merge(json!({"a": {"b": {"y": 2}}}))
            .await
            .expect("Should merge nested objects");

        let x = state.get("/a/b/x").await.expect("Should get /a/b/x");
        assert_eq!(x, Some(json!(1)));

        let y = state.get("/a/b/y").await.expect("Should get /a/b/y");
        assert_eq!(y, Some(json!(2)));
    }

    #[tokio::test]
    async fn test_merge_replaces_scalar_with_object() {
        let state = SharedState::from_value(json!({"a": 1}));
        state
            .merge(json!({"a": {"b": 2}}))
            .await
            .expect("Should replace scalar with merged object");

        let a = state.get("/a").await.expect("Should get /a");
        assert_eq!(a, Some(json!({"b": 2})));

        let b = state.get("/a/b").await.expect("Should get /a/b");
        assert_eq!(b, Some(json!(2)));
    }

    #[tokio::test]
    async fn test_scalar_root_get_and_get_all() {
        let state = SharedState::from_value(json!(42));
        assert_eq!(state.get_all().await, json!(42));
        assert_eq!(state.get("/").await.expect("root get"), Some(json!(42)));
        assert!(state.get("/anything").await.is_err());
    }

    #[tokio::test]
    async fn test_set_root_scalar_then_nested_coerces_to_object() {
        let state = SharedState::from_value(json!([1, 2, 3]));
        assert_eq!(state.get_all().await, json!([1, 2, 3]));

        state
            .set("/foo", json!("bar"))
            .await
            .expect("nested set should coerce non-object root to {}");
        assert_eq!(state.get_all().await, json!({"foo": "bar"}));
    }

    #[tokio::test]
    async fn test_clear() {
        let state = SharedState::from_value(json!({"a": 1, "b": 2}));
        state.clear().await;
        assert_eq!(state.get_all().await, json!({}));
    }

    #[tokio::test]
    async fn test_namespaced_writes_are_isolated_per_agent() {
        let policy = StateWritePolicy::deny_by_default()
            .with_private_namespace("reviewer")
            .with_private_namespace("planner");
        let state = SharedState::with_write_policy(policy);

        state
            .set_in_namespace("reviewer", "findings", json!(["bug"]))
            .await
            .expect("reviewer writes its own namespace");
        let findings = state
            .get_in_namespace("reviewer", "findings")
            .await
            .expect("read reviewer findings");
        assert_eq!(findings, Some(json!(["bug"])));

        let err = state
            .set_as("reviewer", "/agents/planner/plan", json!("hijack"))
            .await
            .expect_err("reviewer must not write the planner namespace");
        assert!(err.to_string().contains("may not write shared state path"));
    }

    #[tokio::test]
    async fn test_merge_as_authorizes_every_top_level_key() {
        let policy = StateWritePolicy::deny_by_default().allow("planner", "/plan*");
        let state = SharedState::with_write_policy(policy);

        state
            .merge_as("planner", json!({"plan": {"step": 1}}))
            .await
            .expect("planner may merge into /plan");

        let err = state
            .merge_as(
                "planner",
                json!({"plan": {"step": 2}, "secrets": {"k": "v"}}),
            )
            .await
            .expect_err("bundled unauthorized key must refuse the whole merge");
        assert!(err.to_string().contains("/secrets"));
        assert_eq!(state.get("/secrets").await.expect("read secrets"), None);
        // The refused merge must not have applied its authorized half either.
        assert_eq!(
            state.get("/plan/step").await.expect("read plan step"),
            Some(json!(1))
        );
    }

    #[tokio::test]
    async fn test_unattributed_writes_stay_unrestricted() {
        let policy = StateWritePolicy::deny_by_default();
        let state = SharedState::with_write_policy(policy);

        state
            .set("/anything", json!(1))
            .await
            .expect("orchestrator writes bypass the agent policy");
        assert!(!state.can_write("reviewer", "/anything"));
    }

    #[tokio::test]
    async fn test_concurrent_writes_to_different_top_level_keys() {
        let state = SharedState::new();
        let a = state.clone();
        let b = state.clone();

        let (r1, r2) = tokio::join!(
            async move { a.set("/alpha/value", json!(1)).await },
            async move { b.set("/beta/value", json!(2)).await },
        );
        r1.expect("set alpha");
        r2.expect("set beta");

        assert_eq!(state.get("/alpha/value").await.unwrap(), Some(json!(1)));
        assert_eq!(state.get("/beta/value").await.unwrap(), Some(json!(2)));
    }
}

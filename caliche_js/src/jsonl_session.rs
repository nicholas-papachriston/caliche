//! JavaScript bindings for pi-inspired JSONL session trees.
#![expect(
    clippy::needless_pass_by_value,
    reason = "NAPI must own values converted from JavaScript arguments at the ABI boundary"
)]

use crate::get_runtime;
use caliche::agent::session::{
    JsonlSessionEntry, JsonlSessionHeader, JsonlSessionStore, JsonlSessionTree,
    new_entry_id as rust_new_entry_id, parse_jsonl_session as rust_parse_jsonl_session,
};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde_json::Value;

/// Session header (first JSONL line).
#[napi(js_name = "JsonlSessionHeader")]
#[derive(Clone)]
pub struct JsJsonlSessionHeader {
    pub(crate) inner: JsonlSessionHeader,
}

#[napi]
impl JsJsonlSessionHeader {
    #[napi(constructor, catch_unwind)]
    pub fn new(id: String, cwd: Option<String>) -> Self {
        Self {
            inner: JsonlSessionHeader::new(id, cwd),
        }
    }

    #[napi(getter, catch_unwind)]
    pub fn id(&self) -> String {
        self.inner.id.clone()
    }

    #[napi(getter, catch_unwind)]
    pub const fn version(&self) -> u32 {
        self.inner.version
    }

    #[napi(getter, catch_unwind)]
    pub fn cwd(&self) -> Option<String> {
        self.inner.cwd.clone()
    }

    #[napi(getter, catch_unwind)]
    pub fn parent_session(&self) -> Option<String> {
        self.inner.parent_session.clone()
    }

    #[napi(catch_unwind)]
    pub fn set_parent_session(&mut self, parent_session: Option<String>) {
        self.inner.parent_session = parent_session;
    }

    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn to_object(&self) -> napi::Result<Value> {
        serde_json::to_value(&self.inner)
            .map_err(|e| Error::from_reason(format!("Failed to serialize header: {e}")))
    }
}

/// One tree-linked session entry.
#[napi(js_name = "JsonlSessionEntry")]
#[derive(Clone)]
pub struct JsJsonlSessionEntry {
    pub(crate) inner: JsonlSessionEntry,
}

#[napi]
impl JsJsonlSessionEntry {
    #[napi(factory, catch_unwind)]
    pub fn message(id: String, message: Value, parent_id: Option<String>) -> Self {
        Self {
            inner: JsonlSessionEntry::message(id, parent_id, message),
        }
    }

    #[napi(factory, catch_unwind)]
    pub fn branch_summary(
        id: String,
        from_id: String,
        summary: String,
        parent_id: Option<String>,
    ) -> Self {
        Self {
            inner: JsonlSessionEntry::BranchSummary {
                id,
                parent_id,
                timestamp: chrono::Utc::now(),
                summary,
                from_id,
            },
        }
    }

    #[napi(factory, catch_unwind)]
    pub fn compaction_summary(
        id: String,
        summary: String,
        parent_id: Option<String>,
        tokens_before: Option<u32>,
    ) -> Self {
        Self {
            inner: JsonlSessionEntry::CompactionSummary {
                id,
                parent_id,
                timestamp: chrono::Utc::now(),
                summary,
                tokens_before: tokens_before.map(|v| v as usize),
            },
        }
    }

    #[napi(factory, catch_unwind)]
    pub fn model_change(
        id: String,
        provider: String,
        model_id: String,
        parent_id: Option<String>,
    ) -> Self {
        Self {
            inner: JsonlSessionEntry::ModelChange {
                id,
                parent_id,
                timestamp: chrono::Utc::now(),
                provider,
                model_id,
            },
        }
    }

    #[napi(getter, catch_unwind)]
    pub fn id(&self) -> String {
        self.inner.id().to_string()
    }

    #[napi(getter, catch_unwind)]
    pub fn parent_id(&self) -> Option<String> {
        self.inner.parent_id().map(str::to_string)
    }

    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn to_object(&self) -> napi::Result<Value> {
        serde_json::to_value(&self.inner)
            .map_err(|e| Error::from_reason(format!("Failed to serialize entry: {e}")))
    }
}

/// In-memory JSONL session tree.
#[napi(js_name = "JsonlSessionTree")]
pub struct JsJsonlSessionTree {
    inner: JsonlSessionTree,
}

#[napi]
impl JsJsonlSessionTree {
    #[napi(constructor, catch_unwind)]
    pub fn new(header: &JsJsonlSessionHeader) -> Self {
        Self {
            inner: JsonlSessionTree::new(header.inner.clone()),
        }
    }

    #[napi(getter, catch_unwind)]
    pub fn header(&self) -> JsJsonlSessionHeader {
        JsJsonlSessionHeader {
            inner: self.inner.header.clone(),
        }
    }

    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn append(&mut self, entry: &JsJsonlSessionEntry) -> napi::Result<()> {
        self.inner
            .append(entry.inner.clone())
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    #[napi(catch_unwind)]
    pub fn leaves(&self) -> Vec<String> {
        self.inner
            .leaves()
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn branch_path(&self, leaf_id: String) -> napi::Result<Vec<Value>> {
        let path = self
            .inner
            .branch_path(&leaf_id)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        path.into_iter()
            .map(|entry| {
                serde_json::to_value(entry)
                    .map_err(|e| Error::from_reason(format!("Failed to serialize entry: {e}")))
            })
            .collect()
    }

    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn restore_branch_messages(&self, leaf_id: String) -> napi::Result<Vec<Value>> {
        let messages = self
            .inner
            .restore_branch_messages(&leaf_id)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        messages
            .into_iter()
            .map(|message| {
                serde_json::to_value(message)
                    .map_err(|e| Error::from_reason(format!("Failed to serialize message: {e}")))
            })
            .collect()
    }

    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn branch_from(
        &mut self,
        from_id: String,
        branch_entry_id: String,
        summary: String,
    ) -> napi::Result<String> {
        self.inner
            .branch_from(&from_id, branch_entry_id, summary)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    #[napi(catch_unwind)]
    pub fn entry_count(&self) -> napi::Result<u32> {
        u32::try_from(self.inner.entries.len())
            .map_err(|_| Error::from_reason("session entry count exceeds the JavaScript API limit"))
    }

    /// Return a copy of all entries (root-order append sequence).
    #[napi(catch_unwind)]
    pub fn entries(&self) -> Vec<JsJsonlSessionEntry> {
        self.inner
            .entries
            .iter()
            .cloned()
            .map(|inner| JsJsonlSessionEntry { inner })
            .collect()
    }
}

/// Append-only JSONL session store on disk.
#[napi(js_name = "JsonlSessionStore")]
pub struct JsJsonlSessionStore {
    inner: JsonlSessionStore,
}

#[napi]
impl JsJsonlSessionStore {
    #[napi(constructor, catch_unwind)]
    pub fn new(path: String) -> Self {
        Self {
            inner: JsonlSessionStore::new(path),
        }
    }

    #[napi(getter, catch_unwind)]
    pub fn path(&self) -> String {
        self.inner.path().display().to_string()
    }

    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn create(&self, header: &JsJsonlSessionHeader) -> napi::Result<()> {
        let rt = get_runtime()?;
        rt.block_on(self.inner.create(&header.inner))
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    #[napi(catch_unwind)]
    pub fn append_entry(&self, entry: &JsJsonlSessionEntry) -> napi::Result<()> {
        let rt = get_runtime()?;
        rt.block_on(self.inner.append_entry(&entry.inner))
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn load(&self) -> napi::Result<JsJsonlSessionTree> {
        let rt = get_runtime()?;
        let tree = rt
            .block_on(self.inner.load())
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(JsJsonlSessionTree { inner: tree })
    }
}

/// Parse a full JSONL session document.
#[napi(catch_unwind)]
pub fn parse_jsonl_session(text: String) -> napi::Result<JsJsonlSessionTree> {
    let tree = rust_parse_jsonl_session(&text).map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(JsJsonlSessionTree { inner: tree })
}

/// Generate a short hex entry id.
#[napi(catch_unwind)]
pub fn new_entry_id() -> String {
    rust_new_entry_id()
}

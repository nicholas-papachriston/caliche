//! JavaScript bindings for file mutation style (`fs` write/edit vs `apply_patch`).
#![expect(
    clippy::needless_pass_by_value,
    reason = "NAPI must own values converted from JavaScript arguments at the ABI boundary"
)]
#![expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "NAPI instance methods require a borrowed self receiver"
)]

use caliche::agent::tools::MutationStyle;
use napi_derive::napi;

/// How file mutations are exposed to the model.
///
/// `EditWrite` and `ApplyPatch` are mutually exclusive.
#[napi(js_name = "MutationStyle")]
#[derive(Clone, Copy)]
pub struct JsMutationStyle {
    pub(crate) inner: MutationStyle,
}

#[napi]
impl JsMutationStyle {
    #[napi(constructor, catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn new(style: String) -> napi::Result<Self> {
        let inner = match style.to_lowercase().as_str() {
            "editwrite" | "edit_write" | "edit-write" => MutationStyle::EditWrite,
            "applypatch" | "apply_patch" | "apply-patch" => MutationStyle::ApplyPatch,
            _ => {
                return Err(napi::Error::from_reason(format!(
                    "Invalid mutation style: {style}. Must be one of: edit_write, apply_patch"
                )));
            }
        };
        Ok(Self { inner })
    }

    #[napi(factory, catch_unwind)]
    pub const fn edit_write() -> Self {
        Self {
            inner: MutationStyle::EditWrite,
        }
    }

    #[napi(factory, catch_unwind)]
    pub const fn apply_patch() -> Self {
        Self {
            inner: MutationStyle::ApplyPatch,
        }
    }

    #[napi(catch_unwind)]
    #[allow(clippy::inherent_to_string)]
    pub const fn to_string(&self) -> &'static str {
        match self.inner {
            MutationStyle::EditWrite => "EditWrite",
            MutationStyle::ApplyPatch => "ApplyPatch",
        }
    }
}

impl From<JsMutationStyle> for MutationStyle {
    fn from(value: JsMutationStyle) -> Self {
        value.inner
    }
}

impl From<MutationStyle> for JsMutationStyle {
    fn from(inner: MutationStyle) -> Self {
        Self { inner }
    }
}

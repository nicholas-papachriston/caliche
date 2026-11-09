//! Credential class carried by model-use actions.
//!
//! Records the class. The floor denies [`CredentialClass::VendorSubscription`].
//! The provider factory is not wired to this check. There is no in-app
//! `ChatGPT` or Claude subscription login.

use serde::{Deserialize, Serialize};

/// How a model call is authenticated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CredentialClass {
    /// Operator-supplied provider API key.
    ByoApiKey,
    /// Local model with no vendor account.
    LocalModel,
    /// Vendor subscription / device-login credential.
    ///
    /// The floor denies `ModelUse` of this class. Caliche does not read
    /// `~/.codex/auth.json` or Claude credential stores.
    VendorSubscription,
}

//! Policy sources contribute rules at one layer. They are not deciders.
//!
//! A future Cedar or Rego adapter can implement this trait and feed the
//! deployment layer. It cannot sit above deployment, cannot supply the floor,
//! and cannot implement [`crate::Decide`].

use crate::error::PolicyError;
use crate::posture::Layer;
use crate::rule::{Rule, RuleSet};

/// Loads a [`RuleSet`] for exactly one layer.
pub trait PolicySource {
    /// Layer these rules belong to. [`Layer::Floor`] is rejected.
    fn layer(&self) -> Layer;

    /// Load rules.
    ///
    /// # Errors
    ///
    /// Returns a [`PolicyError`] when the source cannot read its rules.
    fn load(&self) -> Result<RuleSet, PolicyError>;
}

/// Ingest a source.
///
/// # Errors
///
/// Returns [`PolicyError::SourceCannotSupplyFloor`] when the source claims
/// the floor, and [`PolicyError::SourceLayerMismatch`] when a rule's layer
/// disagrees with [`PolicySource::layer`].
pub fn ingest_source(source: &impl PolicySource) -> Result<RuleSet, PolicyError> {
    let layer = source.layer();
    if layer == Layer::Floor {
        return Err(PolicyError::SourceCannotSupplyFloor);
    }
    let set = source.load()?;
    if set.rules.iter().any(|rule| rule.layer != layer) {
        return Err(PolicyError::SourceLayerMismatch);
    }
    Ok(set)
}

/// Return the rules from [`ingest_source`].
///
/// # Errors
///
/// Returns the same errors as [`ingest_source`].
pub fn source_rules(source: &impl PolicySource) -> Result<Vec<Rule>, PolicyError> {
    Ok(ingest_source(source)?.rules)
}

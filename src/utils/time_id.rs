//! Time-ordered identifiers.
//!
//! Sessions, checkpoints, tool actions, and thinking segments share UUID v7
//! (`Uuid::now_v7`). The timestamp is in the high bits, so lexical order matches
//! creation order for ids minted on one clock.

use uuid::Uuid;

/// Mint a time-sortable UUID v7.
///
/// This is the same generator as [`crate::data::persistence::generate_session_id`].
#[must_use]
pub fn generate_time_id() -> Uuid {
    Uuid::now_v7()
}

#[cfg(test)]
mod tests {
    use super::generate_time_id;
    use uuid::Version;

    #[test]
    fn time_ids_are_uuid_v7_and_sort_by_creation() {
        let first = generate_time_id();
        let second = generate_time_id();
        assert_eq!(first.get_version(), Some(Version::SortRand));
        assert_eq!(second.get_version(), Some(Version::SortRand));
        assert!(first.as_bytes() <= second.as_bytes());
        assert_ne!(first, second);
    }
}

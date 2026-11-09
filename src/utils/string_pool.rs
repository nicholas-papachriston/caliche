//! Cheap `Arc<str>` construction for repeated tool-argument strings.

use std::sync::Arc;

/// Allocate an `Arc<str>` for `s`.
#[inline]
pub fn intern(s: &str) -> Arc<str> {
    Arc::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_preserves_content() {
        let user = intern("user");
        assert_eq!(user.as_ref(), "user");
    }

    #[test]
    fn intern_unknown_string() {
        let custom = intern("custom_string");
        assert_eq!(custom.as_ref(), "custom_string");
    }
}

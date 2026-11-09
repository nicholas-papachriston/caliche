use beef::lean::Cow;
use std::cell::RefCell;

/// Owned key string (lean `beef::Cow` for compact storage vs `String`).
pub type KeyCow = Cow<'static, str>;

/// Thread-local key builder for efficient string operations without allocations
pub struct KeyBuilder {
    buf: String,
}

impl KeyBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            buf: String::with_capacity(128),
        }
    }

    /// Build a composite key in the format "namespace:id"
    /// Returns a reference to the internal buffer to avoid allocation
    pub fn composite_key(&mut self, namespace: Option<&str>, id: &str) -> &str {
        self.buf.clear();
        self.buf.push_str(namespace.unwrap_or("default"));
        self.buf.push(':');
        self.buf.push_str(id);
        &self.buf
    }

    /// Build a key with a prefix in the format "prefix/namespace/id.ext"
    pub fn prefixed_key(
        &mut self,
        prefix: &str,
        namespace: Option<&str>,
        id: &str,
        ext: &str,
    ) -> &str {
        self.buf.clear();
        self.buf.push_str(prefix);
        self.buf.push_str(namespace.unwrap_or("default"));
        self.buf.push('/');
        self.buf.push_str(id);
        self.buf.push_str(ext);
        &self.buf
    }

    /// Build a Redis-style key in the format "prefixnamespace:id"
    pub fn redis_key(&mut self, prefix: &str, namespace: &str, id: &str) -> &str {
        self.buf.clear();
        self.buf.push_str(prefix);
        self.buf.push_str(namespace);
        self.buf.push(':');
        self.buf.push_str(id);
        &self.buf
    }
}

impl Default for KeyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

thread_local! {
    static KEY_BUILDER: RefCell<KeyBuilder> = RefCell::new(KeyBuilder::new());
}

/// Build a composite key using thread-local storage for zero-allocation
/// of the intermediate buffer; returns an owned [`KeyCow`].
#[must_use]
pub fn build_composite_key(namespace: Option<&str>, id: &str) -> KeyCow {
    KEY_BUILDER.with(|kb| {
        let mut builder = kb.borrow_mut();
        let key = builder.composite_key(namespace, id);
        Cow::from(String::from(key))
    })
}

/// Build a prefixed key using thread-local storage
#[must_use]
pub fn build_prefixed_key(prefix: &str, namespace: Option<&str>, id: &str, ext: &str) -> KeyCow {
    KEY_BUILDER.with(|kb| {
        let mut builder = kb.borrow_mut();
        let key = builder.prefixed_key(prefix, namespace, id, ext);
        Cow::from(String::from(key))
    })
}

/// Build a Redis-style key using thread-local storage
#[must_use]
pub fn build_redis_key(prefix: &str, namespace: &str, id: &str) -> KeyCow {
    KEY_BUILDER.with(|kb| {
        let mut builder = kb.borrow_mut();
        let key = builder.redis_key(prefix, namespace, id);
        Cow::from(String::from(key))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composite_key_with_namespace() {
        let key = build_composite_key(Some("test"), "id123");
        assert_eq!(key.as_ref(), "test:id123");
    }

    #[test]
    fn test_composite_key_without_namespace() {
        let key = build_composite_key(None, "id456");
        assert_eq!(key.as_ref(), "default:id456");
    }

    #[test]
    fn test_prefixed_key() {
        let key = build_prefixed_key("prefix/", Some("ns"), "id", ".bin");
        assert_eq!(key.as_ref(), "prefix/ns/id.bin");
    }

    #[test]
    fn test_redis_key() {
        let key = build_redis_key("emb:", "namespace", "id123");
        assert_eq!(key.as_ref(), "emb:namespace:id123");
    }

    #[test]
    fn test_builder_reuse() {
        let mut builder = KeyBuilder::new();
        let key1 = builder.composite_key(Some("ns1"), "id1");
        assert_eq!(key1, "ns1:id1");
        let key2 = builder.composite_key(Some("ns2"), "id2");
        assert_eq!(key2, "ns2:id2");
    }
}

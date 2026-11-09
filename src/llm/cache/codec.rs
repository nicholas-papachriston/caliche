//! Versioned binary encoding for [`crate::llm::cache::response::CachedResponse`] (LLM response cache).
//!
//! Payloads are a fixed magic prefix plus an rkyv body (v1 only).

use crate::data::persistence::rkyv_envelope::{
    LLM_CACHE_MAGIC_V1, prepend_magic, strip_magic_prefix,
};
use crate::llm::cache::response::CachedResponse;
use anyhow::Context;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};

#[derive(Archive, RkyvSerialize, RkyvDeserialize, Debug, Clone)]
#[rkyv(derive(Debug))]
struct CachedResponseStoredV1 {
    content: String,
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    embedding_tokens: Option<u32>,
    total_tokens: Option<u32>,
    context_window: Option<u32>,
    metadata_json: String,
}

impl TryFrom<&CachedResponse> for CachedResponseStoredV1 {
    type Error = anyhow::Error;

    fn try_from(c: &CachedResponse) -> Result<Self, Self::Error> {
        let metadata_json =
            serde_json::to_string(&c.metadata).context("serialize cache metadata")?;
        Ok(Self {
            content: c.content.clone(),
            prompt_tokens: c.token_usage.prompt_tokens,
            completion_tokens: c.token_usage.completion_tokens,
            embedding_tokens: c.token_usage.embedding_tokens,
            total_tokens: c.token_usage.total_tokens,
            context_window: c.token_usage.context_window,
            metadata_json,
        })
    }
}

impl TryFrom<CachedResponseStoredV1> for CachedResponse {
    type Error = anyhow::Error;

    fn try_from(s: CachedResponseStoredV1) -> Result<Self, Self::Error> {
        let metadata =
            serde_json::from_str(&s.metadata_json).context("deserialize cache metadata")?;
        Ok(Self {
            content: s.content,
            token_usage: crate::messages::TokenUsage {
                prompt_tokens: s.prompt_tokens,
                completion_tokens: s.completion_tokens,
                embedding_tokens: s.embedding_tokens,
                total_tokens: s.total_tokens,
                context_window: s.context_window,
                cache_creation_tokens: None,
                cache_read_tokens: None,
                last_prompt_tokens: None,
            },
            metadata,
        })
    }
}

pub fn encode_cached_response(response: &CachedResponse) -> anyhow::Result<Vec<u8>> {
    let stored = CachedResponseStoredV1::try_from(response)?;
    let body = rkyv::to_bytes::<rkyv::rancor::Error>(&stored)
        .map(rkyv::util::AlignedVec::into_vec)
        .map_err(|e| anyhow::anyhow!("rkyv serialize LLM cache entry: {e}"))?;
    Ok(prepend_magic(LLM_CACHE_MAGIC_V1, &body))
}

pub fn decode_cached_response(bytes: &[u8]) -> anyhow::Result<CachedResponse> {
    let body = strip_magic_prefix(LLM_CACHE_MAGIC_V1, bytes, "LLM cache entry")?;
    let stored: CachedResponseStoredV1 = rkyv::from_bytes::<_, rkyv::rancor::Error>(body)
        .map_err(|e| anyhow::anyhow!("rkyv deserialize LLM cache entry: {e}"))?;
    CachedResponse::try_from(stored)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::cache::response::CachedResponse;
    use crate::messages::TokenUsage;

    #[test]
    fn cached_response_roundtrip() {
        let c = CachedResponse {
            content: "hello".into(),
            token_usage: TokenUsage::from_counts(Some(1), Some(2), None),
            metadata: serde_json::json!({"x": true}),
        };
        let bytes = encode_cached_response(&c).expect("encode");
        assert!(bytes.starts_with(crate::data::persistence::rkyv_envelope::LLM_CACHE_MAGIC_V1));
        let got = decode_cached_response(&bytes).expect("decode");
        assert_eq!(got.content, c.content);
        assert_eq!(got.metadata, c.metadata);
        assert_eq!(got.token_usage.prompt_tokens, Some(1));
    }

    #[test]
    fn cached_response_roundtrips_token_boundaries() {
        let cached = CachedResponse {
            content: String::new(),
            token_usage: TokenUsage {
                prompt_tokens: Some(u32::MAX),
                completion_tokens: Some(u32::MAX),
                embedding_tokens: Some(u32::MAX),
                total_tokens: Some(u32::MAX),
                context_window: Some(u32::MAX),
                cache_creation_tokens: None,
                cache_read_tokens: None,
                last_prompt_tokens: None,
            },
            metadata: serde_json::json!({"nested": [null, true, u64::MAX]}),
        };

        let decoded = decode_cached_response(&encode_cached_response(&cached).expect("encode"))
            .expect("decode");
        assert_eq!(decoded.token_usage, cached.token_usage);
        assert_eq!(decoded.metadata, cached.metadata);
    }

    #[test]
    fn decode_rejects_bad_magic() {
        let err = decode_cached_response(b"xxxx").expect_err("bad");
        assert!(
            err.to_string().contains("cache") || err.to_string().contains("magic"),
            "{err}"
        );
    }
}

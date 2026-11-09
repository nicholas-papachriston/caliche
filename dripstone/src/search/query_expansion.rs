//! Query expansion for full-text search (keyword extraction, FTS-ready queries).

use std::collections::HashSet;

/// Configuration for query expansion.
#[derive(Debug, Clone, Default)]
pub struct QueryExpansionConfig {
    /// Whether expansion is enabled.
    pub enabled: bool,
    /// Use expansion automatically when no embedder is available.
    pub auto_fallback: bool,
    /// Extra stop words beyond built-in sets.
    pub extra_stop_words: Vec<String>,
}

/// Optional LLM-assisted query expansion (for use when an LLM is available).
#[async_trait::async_trait]
pub trait LlmQueryExpander: Send + Sync {
    async fn expand(
        &self,
        query: &str,
    ) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>>;
}

fn stop_words_en() -> HashSet<&'static str> {
    [
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is",
        "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with", "the", "this",
        "but", "they", "have", "had", "what", "when", "where", "which", "who", "why", "how", "all",
        "each", "every", "both", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "just", "can", "should", "now",
        "into", "out", "up", "down", "over", "under", "again", "then", "once", "here", "there",
        "any", "if", "or", "because", "until", "while", "during", "before", "after", "above",
        "below",
    ]
    .into_iter()
    .collect()
}

fn stop_words_zh() -> HashSet<&'static str> {
    [
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也",
        "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那",
        "里", "什么", "我们", "可以", "这个", "他们", "她", "他", "它", "为", "与", "及", "而",
        "或", "等", "但", "如果", "因为", "所以", "然后", "现在", "已经", "还是", "再", "又", "还",
        "才",
    ]
    .into_iter()
    .collect()
}

fn is_valid_keyword(token: &str) -> bool {
    if token.len() < 2 {
        return false;
    }
    if token.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    if token.chars().all(|c| !c.is_alphanumeric()) {
        return false;
    }
    true
}

/// Tokenize text: CJK as unigrams+bigrams, Latin on whitespace/punctuation.
fn tokenize_multilingual(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut latin = String::new();
    let mut cjk = String::new();

    for c in text.chars() {
        if c.is_whitespace() || c.is_ascii_punctuation() {
            if !latin.is_empty() {
                out.push(std::mem::take(&mut latin));
            }
            if cjk.chars().count() >= 2 {
                for i in 0..=cjk.chars().count().saturating_sub(2) {
                    let s: String = cjk.chars().skip(i).take(2).collect();
                    if !s.is_empty() {
                        out.push(s);
                    }
                }
                cjk.clear();
            } else if !cjk.is_empty() {
                out.push(std::mem::take(&mut cjk));
            }
        } else if is_cjk(c) {
            if !latin.is_empty() {
                out.push(std::mem::take(&mut latin));
            }
            cjk.push(c);
        } else {
            if cjk.chars().count() >= 2 {
                for i in 0..=cjk.chars().count().saturating_sub(2) {
                    let s: String = cjk.chars().skip(i).take(2).collect();
                    if !s.is_empty() {
                        out.push(s);
                    }
                }
                cjk.clear();
            } else if !cjk.is_empty() {
                out.push(std::mem::take(&mut cjk));
            }
            latin.push(c);
        }
    }
    if !latin.is_empty() {
        out.push(latin);
    }
    if cjk.chars().count() >= 2 {
        for i in 0..=cjk.chars().count().saturating_sub(2) {
            let s: String = cjk.chars().skip(i).take(2).collect();
            if !s.is_empty() {
                out.push(s);
            }
        }
    } else if !cjk.is_empty() {
        out.push(cjk);
    }
    out
}

const fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}'   // CJK Unified Ideographs
        | '\u{3040}'..='\u{309F}' // Hiragana
        | '\u{30A0}'..='\u{30FF}' // Katakana
        | '\u{AC00}'..='\u{D7AF}' // Hangul
    )
}

/// Extract meaningful keywords from a conversational query.
#[must_use]
pub fn extract_keywords(query: &str) -> Vec<String> {
    let stop_en = stop_words_en();
    let stop_zh = stop_words_zh();
    let tokens = tokenize_multilingual(query);
    tokens
        .into_iter()
        .filter(|t| {
            is_valid_keyword(t) && !stop_en.contains(t.as_str()) && !stop_zh.contains(t.as_str())
        })
        .collect()
}

/// Result of expanding a query for FTS.
#[derive(Debug, Clone)]
pub struct ExpandedQuery {
    pub original: String,
    pub keywords: Vec<String>,
    /// FTS-ready query string (terms joined with OR).
    pub fts_query: String,
}

/// Expand a query for FTS: original terms plus extracted keywords, joined with OR.
#[must_use]
pub fn expand_query_for_fts(query: &str) -> ExpandedQuery {
    let keywords = extract_keywords(query);
    let mut terms: Vec<String> = Vec::new();
    let q_trim = query.trim();
    if !q_trim.is_empty() && is_valid_keyword(q_trim) {
        terms.push(q_trim.to_string());
    }
    for k in &keywords {
        if !terms.contains(k) {
            terms.push(k.clone());
        }
    }
    let fts_query = terms.join(" OR ");
    ExpandedQuery {
        original: query.to_string(),
        keywords,
        fts_query: if fts_query.is_empty() {
            query.to_string()
        } else {
            fts_query
        },
    }
}

//! Text tokenization utilities for sparse vector generation.

use std::collections::HashSet;

#[cfg(feature = "rust-stemmers")]
use rust_stemmers::{Algorithm, Stemmer};

/// Trait for tokenization strategies.
pub trait Tokenizer: Send + Sync {
    /// Tokenizes input text into a vector of tokens.
    fn tokenize(&self, text: &str) -> Vec<String>;
}

/// Simple word-based tokenizer with optional stemming and stopword removal.
pub struct SimpleTokenizer {
    lowercase: bool,
    remove_punctuation: bool,
    #[cfg(feature = "rust-stemmers")]
    stemmer: Option<Stemmer>,
    #[cfg(feature = "rust-stemmers")]
    stemmer_algorithm: Option<Algorithm>,
    stopwords: HashSet<String>,
}

impl std::fmt::Debug for SimpleTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimpleTokenizer")
            .field("lowercase", &self.lowercase)
            .field("remove_punctuation", &self.remove_punctuation)
            .field("stopwords_len", &self.stopwords.len())
            .finish_non_exhaustive()
    }
}

impl Clone for SimpleTokenizer {
    fn clone(&self) -> Self {
        let cloned = Self::new()
            .with_lowercase(self.lowercase)
            .with_punctuation_removal(self.remove_punctuation)
            .with_stopwords(self.stopwords.iter().cloned());
        #[cfg(feature = "rust-stemmers")]
        let cloned = if let Some(algorithm) = self.stemmer_algorithm {
            cloned.with_stemming(algorithm)
        } else {
            cloned
        };
        cloned
    }
}

impl SimpleTokenizer {
    /// Creates a new simple tokenizer with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            lowercase: true,
            remove_punctuation: true,
            #[cfg(feature = "rust-stemmers")]
            stemmer: None,
            #[cfg(feature = "rust-stemmers")]
            stemmer_algorithm: None,
            stopwords: HashSet::new(),
        }
    }

    /// Enables or disables lowercase conversion.
    #[must_use]
    pub const fn with_lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }

    /// Enables or disables punctuation removal.
    #[must_use]
    pub const fn with_punctuation_removal(mut self, remove: bool) -> Self {
        self.remove_punctuation = remove;
        self
    }

    /// Adds stemming with the specified language algorithm.
    #[cfg(feature = "rust-stemmers")]
    #[must_use]
    pub fn with_stemming(mut self, algorithm: Algorithm) -> Self {
        self.stemmer = Some(Stemmer::create(algorithm));
        self.stemmer_algorithm = Some(algorithm);
        self
    }

    /// Sets a custom stopword list.
    #[must_use]
    pub fn with_stopwords<I, S>(mut self, stopwords: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.stopwords = stopwords
            .into_iter()
            .map(std::convert::Into::into)
            .collect();
        self
    }

    /// Uses common English stopwords.
    #[must_use]
    pub fn with_english_stopwords(mut self) -> Self {
        self.stopwords = ENGLISH_STOPWORDS.iter().map(|&s| s.to_string()).collect();
        self
    }

    fn process_token(&self, token: &str) -> Option<String> {
        let mut processed = token.to_string();

        if self.remove_punctuation {
            processed = processed
                .chars()
                .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                .collect();
        }

        if self.lowercase {
            processed = processed.to_lowercase();
        }

        if processed.trim().is_empty() {
            return None;
        }

        if self.stopwords.contains(&processed) {
            return None;
        }

        #[cfg(feature = "rust-stemmers")]
        if let Some(ref stemmer) = self.stemmer {
            processed = stemmer.stem(&processed).to_string();
        }

        Some(processed)
    }
}

impl Default for SimpleTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for SimpleTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .filter_map(|token| self.process_token(token))
            .collect()
    }
}

/// Common English stopwords.
const ENGLISH_STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tokenizer() {
        let tokenizer = SimpleTokenizer::new();
        let tokens = tokenizer.tokenize("Hello World! This is a test.");

        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
    }

    #[test]
    fn test_stopwords() {
        let tokenizer = SimpleTokenizer::new().with_english_stopwords();
        let tokens = tokenizer.tokenize("This is a test");

        assert!(!tokens.contains(&"this".to_string()));
        assert!(!tokens.contains(&"is".to_string()));
        assert!(!tokens.contains(&"a".to_string()));
        assert!(tokens.contains(&"test".to_string()));
    }

    #[test]
    fn test_punctuation_removal() {
        let tokenizer = SimpleTokenizer::new().with_punctuation_removal(true);
        let tokens = tokenizer.tokenize("Hello, World!");

        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[cfg(feature = "rust-stemmers")]
    #[test]
    fn test_stemming() {
        let tokenizer = SimpleTokenizer::new().with_stemming(Algorithm::English);

        let tokens = tokenizer.tokenize("running runs ran");
        // All should stem to "run" or similar
        assert_eq!(tokens.len(), 3);
    }
}

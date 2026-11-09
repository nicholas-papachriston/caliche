//! Knowledge store management for CLI.
//!
//! Provides persistent cross-session semantic memory.

use anyhow::Result;
use caliche::agent::memory::{ApproximateTokenCounter, KnowledgeEntry, KnowledgeStore};
use caliche::llm::EmbeddingProvider;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Global knowledge store instance (wrapped in Arc<RwLock> for sharing)
pub type SharedKnowledgeStore = Arc<RwLock<KnowledgeStore>>;

/// Create or load a knowledge store from the service.
///
/// Returns None if the service doesn't have an embedding provider.
pub fn create_knowledge_store(
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
) -> Result<Option<SharedKnowledgeStore>> {
    let embedding_provider = match embedding_provider {
        Some(ep) => ep,
        None => return Ok(None),
    };

    let token_counter: Arc<dyn caliche::agent::memory::TokenCounter> =
        Arc::new(ApproximateTokenCounter::default());

    let store = KnowledgeStore::new(embedding_provider, token_counter, 1000);
    Ok(Some(Arc::new(RwLock::new(store))))
}

/// Store a fact in the knowledge store.
pub async fn remember(
    store: &SharedKnowledgeStore,
    content: String,
    category: Option<String>,
    importance: Option<f32>,
) -> Result<String> {
    let mut store_guard = store.write().await;

    // Simple categorization
    let category = category.unwrap_or_else(|| {
        let lower = content.to_lowercase();
        if lower.contains("user") || lower.contains("person") || lower.contains("name") {
            "person".to_string()
        } else if lower.contains("company") || lower.contains("organization") {
            "organization".to_string()
        } else if lower.contains("location") || lower.contains("place") || lower.contains("city") {
            "location".to_string()
        } else if lower.contains("preference")
            || lower.contains("like")
            || lower.contains("dislike")
        {
            "preference".to_string()
        } else {
            "general".to_string()
        }
    });

    // Simple importance estimation
    let importance = importance.unwrap_or_else(|| {
        let mut imp: f32 = 0.5;
        let lower = content.to_lowercase();
        if lower.contains("important") || lower.contains("critical") || lower.contains("key") {
            imp += 0.3;
        }
        if lower.contains("remember") || lower.contains("note") {
            imp += 0.2;
        }
        if content.len() > 100 {
            imp += 0.1;
        }
        imp.clamp(0.0, 1.0)
    });

    // Use the public learn_from_message method
    use caliche::messages::Message;
    let message = Message::user(&content);
    store_guard
        .learn_from_message(&message, category, importance, None)
        .await
}

/// Search the knowledge store for relevant information.
pub async fn recall(
    store: &SharedKnowledgeStore,
    query: &str,
    top_k: usize,
) -> Result<Vec<(KnowledgeEntry, f32)>> {
    let store_guard = store.read().await;
    let results = store_guard.search(query, top_k, 0.5).await?;

    // Convert to owned entries
    let owned_results: Vec<(KnowledgeEntry, f32)> = results
        .into_iter()
        .map(|(entry, score)| (entry.clone(), score))
        .collect();

    Ok(owned_results)
}

/// Get summary of stored knowledge.
pub async fn knowledge_summary(store: &SharedKnowledgeStore) -> String {
    let store_guard = store.read().await;
    let entries = store_guard.all_entries();

    if entries.is_empty() {
        return "No knowledge stored.".to_string();
    }

    let mut categories: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    for entry in entries {
        *categories.entry(entry.category.clone()).or_insert(0) += 1;
    }

    let mut lines = vec![format!("Total entries: {}", entries.len())];
    lines.push(String::new());
    lines.push("By category:".to_string());

    let mut cat_vec: Vec<_> = categories.into_iter().collect();
    cat_vec.sort_by(|a, b| b.1.cmp(&a.1));

    for (category, count) in cat_vec {
        lines.push(format!("  {}: {}", category, count));
    }

    lines.join("\n")
}

/// Remove a knowledge entry by ID.
pub async fn forget(store: &SharedKnowledgeStore, id: &str) -> Result<bool> {
    let mut store_guard = store.write().await;
    let removed = store_guard.remove(id).is_some();
    Ok(removed)
}

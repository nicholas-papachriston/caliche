//! Example demonstrating advanced search: MMR diversity reranking and temporal decay.
//!
//! Run with:
//! ```bash
//! cargo run -p dripstone --example advanced_search_example --features="memory-backend,rag,tokio"
//! ```

macro_rules! outputln {
    ($($arg:tt)*) => {{
        let mut output = std::io::stdout().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use dripstone::Dripstone;
use dripstone::backend::memory::MemoryBackend;
use dripstone::search::{MmrConfig, MmrSimilarity};
use dripstone::traits::{Backend, Embedder};
use std::sync::Arc;

#[cfg(feature = "search-advanced")]
use dripstone::search::TemporalDecayConfig;

#[derive(Debug)]
struct DemoEmbedder {
    dim: usize,
}

impl DemoEmbedder {
    const fn new(dim: usize) -> Self {
        Self { dim }
    }

    fn embed_sync(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];
        for (i, b) in text.bytes().enumerate() {
            v[i % self.dim] = (v[i % self.dim] + f32::from(b) / 256.0) % 1.0;
        }
        v
    }
}

#[async_trait::async_trait]
impl Embedder for DemoEmbedder {
    async fn embed(
        &self,
        texts: &[&str],
    ) -> Result<Vec<Vec<f32>>, dripstone::errors::DripstoneError> {
        Ok(texts.iter().map(|t| self.embed_sync(t)).collect())
    }

    async fn embed_query(
        &self,
        query: &str,
    ) -> Result<Vec<f32>, dripstone::errors::DripstoneError> {
        Ok(self.embed_sync(query))
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn name(&self) -> &'static str {
        "demo"
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    outputln!("=== Dripstone Advanced Search (MMR + Temporal Decay) ===\n");

    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
    let embedder = DemoEmbedder::new(128);
    let vf = Dripstone::new(backend, embedder);

    // Ingest sample documents (similar content to show MMR diversity)
    let docs = [
        "Machine learning is a subset of AI that uses data to learn patterns.",
        "Deep learning uses neural networks with many layers for ML tasks.",
        "Neural networks are the foundation of deep learning and modern AI.",
        "Natural language processing uses ML to understand human language.",
    ];
    for (i, doc) in docs.iter().enumerate() {
        let doc_id = format!("doc-{i}");
        vf.ingest(&doc_id, doc).await?;
    }
    outputln!("Indexed {} documents.\n", docs.len());

    let query = "neural networks and deep learning";
    outputln!("Query: \"{query}\"\n");

    // 1. Default search (no MMR)
    outputln!("--- Default search (top 4) ---");
    let default_results = vf.search(query).limit(4).await?;
    for (i, r) in default_results.iter().enumerate() {
        outputln!(
            "  {}. [Score: {:.4}] {}",
            i + 1,
            r.score,
            r.chunk.content.chars().take(60).collect::<String>()
        );
    }

    // 2. With MMR diversity reranking (Jaccard)
    outputln!("\n--- With MMR diversity (lambda=0.7, Jaccard) ---");
    let mmr_config = MmrConfig {
        enabled: true,
        lambda: 0.7,
        similarity: MmrSimilarity::Jaccard,
    };
    let mmr_results = vf.search(query).limit(4).mmr(mmr_config).await?;
    for (i, r) in mmr_results.iter().enumerate() {
        outputln!(
            "  {}. [Score: {:.4}] {}",
            i + 1,
            r.score,
            r.chunk.content.chars().take(60).collect::<String>()
        );
    }

    #[cfg(feature = "search-advanced")]
    {
        outputln!("\n--- With temporal decay (search-advanced) ---");
        let decay_config = TemporalDecayConfig {
            enabled: true,
            half_life_days: 30.0,
            timestamp_key: "created_at".to_string(),
            path_date_pattern: None,
            evergreen_paths: vec!["MEMORY.md".into()],
        };
        let _decay_results = vf
            .search(query)
            .limit(4)
            .temporal_decay(decay_config)
            .await?;
        outputln!("  (Temporal decay applied when chunk metadata has created_at)");
    }

    outputln!("\nDone.");
    Ok(())
}

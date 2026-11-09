//! Five-line open, add, and search against the default local store.
//!
//! ```text
//! cargo run --example quickstart
//! ```

use dripstone::Dripstone;

#[tokio::main]
async fn main() -> Result<(), dripstone::DripstoneError> {
    let dir = std::env::temp_dir().join(format!("dripstone-quickstart-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let store = Dripstone::open(&dir)?;
    store.add("alpha", "penguins swim in cold water").await?;
    let hits = store.search("penguins").limit(1).await?;
    assert_eq!(hits[0].chunk.doc_id, "alpha");
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

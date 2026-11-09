//! Concurrency and thread safety tests

use caliche::templates::TemplateRegistry;
use std::sync::Arc;
use tokio::task;

#[tokio::test]
async fn test_concurrent_template_access() {
    let registry = Arc::new({
        let mut r = TemplateRegistry::new();
        r.add_template_dir("templates");
        r.load_templates().await.expect("Failed to load templates");
        r
    });

    // Concurrent reads should be safe
    let handles: Vec<_> = (0..10)
        .map(|_| {
            let reg = registry.clone();
            task::spawn(async move { reg.list().len() })
        })
        .collect();

    let results: Vec<_> = futures_util::future::join_all(handles).await;
    let counts: Vec<_> = results
        .into_iter()
        .map(|r| r.expect("Task failed"))
        .collect();

    // All should return same count
    let first_count = counts[0];
    assert!(counts.iter().all(|&c| c == first_count));
}

#[tokio::test]
async fn test_concurrent_search() {
    let registry = Arc::new({
        let mut r = TemplateRegistry::new();
        r.add_template_dir("templates");
        r.load_templates().await.expect("Failed to load templates");
        r
    });

    use caliche::templates::TemplateSearchFilters;

    // Concurrent searches
    let handles: Vec<_> = (0..20)
        .map(|i| {
            let reg = registry.clone();
            task::spawn(async move {
                let filters = TemplateSearchFilters {
                    query: Some(format!("test{}", i)),
                    ..Default::default()
                };
                reg.search(&filters).len()
            })
        })
        .collect();

    let results: Vec<_> = futures_util::future::join_all(handles).await;
    // All should complete without panicking
    for result in results {
        assert!(result.is_ok());
    }
}

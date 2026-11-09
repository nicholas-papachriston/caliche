//! Performance and stress tests

use caliche::templates::TemplateRegistry;

#[tokio::test]
#[ignore = "performance budget; not a correctness test"]
async fn test_registry_performance() {
    let mut registry = TemplateRegistry::new();
    registry.add_template_dir("templates");

    // Measure load time
    let start = std::time::Instant::now();
    registry
        .load_templates()
        .await
        .expect("Failed to load templates");
    let load_time = start.elapsed();

    // Should load reasonably fast (< 1 second for typical template count)
    assert!(load_time.as_secs() < 1 || registry.count() > 100);
}

#[tokio::test]
#[ignore = "performance budget; not a correctness test"]
async fn test_search_performance() {
    let mut registry = TemplateRegistry::new();
    registry.add_template_dir("templates");
    registry
        .load_templates()
        .await
        .expect("Failed to load templates");

    use caliche::templates::TemplateSearchFilters;

    // Multiple searches should be fast
    let start = std::time::Instant::now();
    for _ in 0..100 {
        let filters = TemplateSearchFilters {
            query: Some("test".to_string()),
            ..Default::default()
        };
        let _ = registry.search(&filters);
    }
    let search_time = start.elapsed();

    // 100 searches should complete in reasonable time
    assert!(search_time.as_millis() < 1000);
}

#[tokio::test]
#[ignore = "performance budget; not a correctness test"]
async fn test_large_template_list() {
    let mut registry = TemplateRegistry::new();
    registry.add_template_dir("templates");
    registry
        .load_templates()
        .await
        .expect("Failed to load templates");

    // Getting all templates should be fast
    let start = std::time::Instant::now();
    let all = registry.list();
    let list_time = start.elapsed();

    assert!(list_time.as_millis() < 100);
    assert_eq!(all.len(), registry.count());
}

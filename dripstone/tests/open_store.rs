//! Behavior of `Dripstone::open`: quickstart hit, hybrid default, dense escape.

use dripstone::{Backend, Chunk, Dripstone, SearchMode, SearchResult};

fn doc_ids(results: &[SearchResult]) -> Vec<&str> {
    results
        .iter()
        .map(|result| result.chunk.doc_id.as_str())
        .collect()
}

#[tokio::test]
async fn quickstart_returns_the_penguin_note() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Dripstone::open(dir.path()).expect("open");
    store
        .add("alpha", "penguins swim in cold water")
        .await
        .expect("add alpha");
    store
        .add("beta", "rust compilers translate source")
        .await
        .expect("add beta");
    let hits = store.search("penguins").limit(1).await.expect("search");
    assert_eq!(hits[0].chunk.doc_id, "alpha");
}

#[tokio::test]
async fn open_store_uses_hybrid_and_dense_only_changes_order() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Dripstone::open(dir.path()).expect("open");
    // Dense order is alpha, bravo, charlie.
    // BM25 keeps alpha then charlie; bravo shares no query term.
    // Reciprocal-rank fusion therefore orders alpha, charlie, bravo.
    store.add("alpha", "aaa aaxa").await.expect("add alpha");
    store.add("bravo", "byb aca cec").await.expect("add bravo");
    store
        .add("charlie", "aaxa aaxa aaxa aaxa aaxa aaxa aaxa aaxa")
        .await
        .expect("add charlie");

    let hybrid = store.search("aaa aaxa").limit(3).await.expect("hybrid");
    let dense = store
        .search("aaa aaxa")
        .dense_only()
        .limit(3)
        .await
        .expect("dense");

    assert_eq!(doc_ids(&hybrid), ["alpha", "charlie", "bravo"]);
    assert_eq!(doc_ids(&dense), ["alpha", "bravo", "charlie"]);
    assert_eq!(
        store.config().search.default_mode,
        SearchMode::HybridDenseFullText
    );
    assert_eq!(store.config().search.hybrid_full_text_fusion.rrf_k, 60.0);
}

#[tokio::test]
async fn stopword_note_does_not_hide_a_real_hit() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Dripstone::open(dir.path()).expect("open");
    store
        .add("empty", "to be or not to be")
        .await
        .expect("add empty");
    store
        .add("alpha", "penguins swim in cold water")
        .await
        .expect("add alpha");

    let hits = store
        .search("penguins")
        .dense_only()
        .limit(2)
        .await
        .expect("search");
    assert_eq!(hits[0].chunk.doc_id, "alpha");
    let empty = hits
        .iter()
        .find(|hit| hit.chunk.doc_id == "empty")
        .expect("zero-norm note stays in the result list");
    assert_eq!(empty.score, 0.0);
    store.search("the").limit(2).await.expect("stopword query");
}

#[tokio::test]
async fn reopen_loads_the_persisted_note() {
    let dir = tempfile::tempdir().expect("tempdir");
    {
        let store = Dripstone::open(dir.path()).expect("open");
        store
            .add("alpha", "penguins swim in cold water")
            .await
            .expect("add");
    }
    let store = Dripstone::open(dir.path()).expect("reopen");
    let hits = store.search("penguins").limit(1).await.expect("search");
    assert_eq!(hits[0].chunk.doc_id, "alpha");
}

#[tokio::test]
async fn two_open_handles_do_not_silently_drop_adds() {
    let dir = tempfile::tempdir().expect("tempdir");
    let first = Dripstone::open(dir.path()).expect("first open");
    match Dripstone::open(dir.path()) {
        Err(error) => {
            let message = error.to_string();
            assert!(
                message.contains("already open"),
                "second open should say the folder is locked: {message}"
            );
            first
                .add("alpha", "penguins swim in cold water")
                .await
                .expect("add through the holding handle");
            drop(first);
            let store = Dripstone::open(dir.path()).expect("open after release");
            let hits = store.search("penguins").limit(1).await.expect("search");
            assert_eq!(hits[0].chunk.doc_id, "alpha");
        }
        Ok(second) => {
            first
                .add("alpha", "penguins swim in cold water")
                .await
                .expect("add alpha");
            second
                .add("beta", "rust compilers translate source")
                .await
                .expect("add beta");
            drop(first);
            drop(second);
            let store = Dripstone::open(dir.path()).expect("reopen");
            let penguin = store.search("penguins").limit(1).await.expect("penguin");
            let rust = store.search("compilers").limit(1).await.expect("compilers");
            assert_eq!(penguin[0].chunk.doc_id, "alpha");
            assert_eq!(rust[0].chunk.doc_id, "beta");
        }
    }
}

#[tokio::test]
async fn nan_multi_vector_does_not_roll_back_later_adds() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Dripstone::open(dir.path()).expect("open");
    store
        .add("alpha", "penguins swim in cold water")
        .await
        .expect("add alpha");
    let bad = raw_chunk("bad", vec![0.2; dripstone::local::LOCAL_EMBED_DIMENSION])
        .with_multi_vectors(dripstone::multi_vectors_from_owned(vec![vec![f32::NAN; 4]]));
    let error = store
        .backend()
        .upsert(std::slice::from_ref(&bad))
        .await
        .expect_err("nan multi-vector");
    assert!(
        error.to_string().contains("finite"),
        "unexpected error: {error}"
    );
    assert_eq!(store.list_docs().await, ["alpha"]);
    store
        .add("charlie", "penguins rest on the ice")
        .await
        .expect("add charlie");
    store
        .add("delta", "penguins dive for fish")
        .await
        .expect("add delta");
    drop(store);

    let store = Dripstone::open(dir.path()).expect("reopen");
    assert_eq!(store.list_docs().await, ["alpha", "charlie", "delta"]);
}

#[tokio::test]
async fn upsert_rejects_a_non_finite_vector() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Dripstone::open(dir.path()).expect("open");
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let chunk = raw_chunk("bad", vec![bad; dripstone::local::LOCAL_EMBED_DIMENSION]);
        let error = store
            .backend()
            .upsert(std::slice::from_ref(&chunk))
            .await
            .expect_err("non-finite vector");
        assert!(
            error.to_string().contains("finite"),
            "unexpected error: {error}"
        );
        let listed = store.backend().list(None, None).await.expect("list");
        assert!(
            listed.iter().all(|stored| stored.doc_id != "bad"),
            "rejected vector must not stay in the store"
        );
    }
    drop(store);
    Dripstone::open(dir.path()).expect("reopen");
}

#[tokio::test]
async fn upsert_rejects_a_wrong_dimension() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Dripstone::open(dir.path()).expect("open");
    let chunk = raw_chunk("bad", vec![1.0; 3]);
    let error = store
        .backend()
        .upsert(std::slice::from_ref(&chunk))
        .await
        .expect_err("wrong dimension");
    assert!(
        matches!(
            error,
            dripstone::DripstoneError::InvalidVectorLength {
                expected: dripstone::local::LOCAL_EMBED_DIMENSION,
                actual: 3
            }
        ),
        "unexpected error: {error}"
    );
    let listed = store.backend().list(None, None).await.expect("list");
    assert!(listed.iter().all(|stored| stored.doc_id != "bad"));
    drop(store);
    Dripstone::open(dir.path()).expect("reopen");
}

#[tokio::test]
async fn empty_store_json_reopens_from_the_backup() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Dripstone::open(dir.path()).expect("open");
    store
        .add("alpha", "penguins swim in cold water")
        .await
        .expect("add alpha");
    store
        .add("beta", "rust compilers translate source")
        .await
        .expect("add beta");
    drop(store);

    let primary = dir.path().join("store.json");
    std::fs::write(&primary, b"").expect("truncate snapshot");
    let (opened, warning) = capture_warn(|| Dripstone::open(dir.path()));
    let store = opened.expect("recover");
    assert!(
        warning.contains("quarantine"),
        "unreadable snapshot should warn: {warning}"
    );
    let quarantined = corrupt_files(dir.path());
    assert_eq!(quarantined.len(), 1, "expected one quarantine file");
    assert_eq!(
        std::fs::read(&quarantined[0]).expect("quarantine bytes"),
        b""
    );
    let hits = store.search("penguins").limit(1).await.expect("search");
    assert_eq!(hits[0].chunk.doc_id, "alpha");
    assert_eq!(
        store.list_docs().await,
        ["alpha"],
        "the backup is one write behind, so beta is not restored"
    );
    assert!(
        warning.contains("one write behind"),
        "recovery should say the backup is behind: {warning}"
    );
    drop(store);
    assert_eq!(
        std::fs::read(&quarantined[0]).expect("quarantine bytes stay"),
        b"",
        "a later open must not overwrite the quarantined snapshot"
    );
    let store = Dripstone::open(dir.path()).expect("reopen restored snapshot");
    assert_eq!(store.list_docs().await, ["alpha"]);
}

#[tokio::test]
async fn unsupported_snapshot_does_not_load_or_replace_the_backup() {
    for (field, value) in [
        ("version", serde_json::json!(2)),
        ("embedder", serde_json::json!("other-embedder")),
        ("dimension", serde_json::json!(8)),
    ] {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let store = Dripstone::open(dir.path()).expect("open");
            store.add("one", "one note").await.expect("one");
            store.add("two", "two note").await.expect("two");
            store.add("three", "three note").await.expect("three");
        }
        let primary = dir.path().join("store.json");
        let mut snapshot: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&primary).expect("read")).expect("json");
        snapshot[field] = value;
        let rewritten = serde_json::to_vec_pretty(&snapshot).expect("encode");
        std::fs::write(&primary, &rewritten).expect("write");
        let backup_before = std::fs::read(dir.path().join("store.json.bak")).expect("bak");

        let error = Dripstone::open(dir.path()).expect_err("unsupported snapshot");
        let message = error.to_string();
        assert!(message.contains("not supported"), "{field}: {message}");
        assert_eq!(std::fs::read(&primary).expect("primary stays"), rewritten);
        assert!(corrupt_files(dir.path()).is_empty());
        assert_eq!(
            std::fs::read(dir.path().join("store.json.bak")).expect("bak stays"),
            backup_before
        );
        let docs = snapshot["chunks"]
            .as_array()
            .expect("chunks")
            .iter()
            .filter_map(|chunk| chunk["doc_id"].as_str())
            .collect::<Vec<_>>();
        assert!(docs.contains(&"three"), "{docs:?}");
        assert_eq!(docs.len(), 3);
    }
}

#[tokio::test]
async fn a_newer_version_with_a_new_chunk_shape_is_unsupported_not_quarantined() {
    let dir = tempfile::tempdir().expect("tempdir");
    {
        let store = Dripstone::open(dir.path()).expect("open");
        store.add("one", "one note").await.expect("one");
        store.add("two", "two note").await.expect("two");
    }
    let primary = dir.path().join("store.json");
    let mut snapshot: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&primary).expect("read")).expect("json");
    snapshot["version"] = serde_json::json!(2);
    snapshot["chunks"] = serde_json::json!([{ "segments": [{ "text": "two note" }] }]);
    let rewritten = serde_json::to_vec_pretty(&snapshot).expect("encode");
    std::fs::write(&primary, &rewritten).expect("write");
    let backup_before = std::fs::read(dir.path().join("store.json.bak")).expect("bak");

    let error = Dripstone::open(dir.path()).expect_err("newer snapshot");
    assert_eq!(
        error.to_string(),
        "backend misconfiguration: local store snapshot version is not supported"
    );
    assert_eq!(std::fs::read(&primary).expect("primary stays"), rewritten);
    assert_eq!(
        std::fs::read(dir.path().join("store.json.bak")).expect("bak stays"),
        backup_before
    );
    assert!(corrupt_files(dir.path()).is_empty());
}

#[tokio::test]
async fn unreadable_snapshot_and_backup_fail_loudly() {
    let dir = tempfile::tempdir().expect("tempdir");
    {
        let store = Dripstone::open(dir.path()).expect("open");
        store.add("one", "one note").await.expect("one");
    }
    std::fs::write(dir.path().join("store.json"), b"not json").expect("break primary");
    std::fs::write(dir.path().join("store.json.bak"), b"not backup").expect("break bak");
    let (opened, warning) = capture_warn(|| Dripstone::open(dir.path()));
    let error = opened.expect_err("both snapshots bad");
    let message = error.to_string();
    assert!(
        message.contains("unreadable"),
        "failure should say the snapshot is unreadable: {message}"
    );
    assert!(warning.contains("quarantine"), "{warning}");
    let quarantined = corrupt_files(dir.path());
    assert_eq!(quarantined.len(), 1);
    assert_eq!(
        std::fs::read(&quarantined[0]).expect("quarantine"),
        b"not json"
    );
    assert_eq!(
        std::fs::read(dir.path().join("store.json.bak")).expect("bak"),
        b"not backup"
    );
}

#[tokio::test]
async fn quarantine_without_a_backup_blocks_the_next_open() {
    let dir = tempfile::tempdir().expect("tempdir");
    let primary = dir.path().join("store.json");
    std::fs::write(&primary, b"not json").expect("break primary");
    let (opened, warning) = capture_warn(|| Dripstone::open(dir.path()));
    let error = opened.expect_err("unreadable primary");
    assert!(
        error.to_string().contains("unreadable"),
        "first open should fail closed: {error}"
    );
    assert!(warning.contains("quarantine"), "{warning}");
    let quarantined = corrupt_files(dir.path());
    assert_eq!(quarantined.len(), 1);
    let quarantined_bytes = std::fs::read(&quarantined[0]).expect("quarantine");
    assert_eq!(quarantined_bytes, b"not json");
    assert!(!primary.exists());

    let error = Dripstone::open(dir.path()).expect_err("quarantine remains");
    let message = error.to_string();
    assert!(
        message.contains("quarantine"),
        "a later open must refuse the leftover quarantine: {message}"
    );
    assert!(!primary.exists(), "open must not create an empty snapshot");
    assert_eq!(
        std::fs::read(&quarantined[0]).expect("quarantine stays"),
        quarantined_bytes
    );
}

#[tokio::test]
async fn valid_primary_is_not_replaced_by_an_older_backup() {
    let dir = tempfile::tempdir().expect("tempdir");
    {
        let store = Dripstone::open(dir.path()).expect("open");
        store.add("one", "one note").await.expect("one");
        store.add("two", "two note").await.expect("two");
        store.add("three", "three note").await.expect("three");
    }
    let store = Dripstone::open(dir.path()).expect("reopen");
    let hits = store.list_docs().await;
    assert_eq!(hits, ["one", "three", "two"]);
}

#[cfg(unix)]
#[tokio::test]
async fn open_read_only_directory_reports_that_it_is_not_writable() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("locked");
    std::fs::create_dir(&path).expect("dir");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o555)).expect("mode");
    let error = Dripstone::open(&path).expect_err("read only");
    assert!(
        error.to_string().contains("not writable"),
        "read-only directory should be explicit: {error}"
    );
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).expect("restore");
}

#[tokio::test]
async fn upsert_rejects_a_mixed_batch_without_keeping_the_good_chunk() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Dripstone::open(dir.path()).expect("open");
    let good = raw_chunk("good", vec![0.25; dripstone::local::LOCAL_EMBED_DIMENSION]);
    let bad = raw_chunk(
        "bad",
        vec![f32::NAN; dripstone::local::LOCAL_EMBED_DIMENSION],
    );
    for batch in [vec![good.clone(), bad.clone()], vec![bad, good]] {
        let error = store.backend().upsert(&batch).await.expect_err("mixed");
        assert!(error.to_string().contains("finite"), "{error}");
        let listed = store.backend().list(None, None).await.expect("list");
        assert!(
            listed.is_empty(),
            "a rejected batch must not keep the valid chunk"
        );
    }
}

#[tokio::test]
async fn upsert_rejects_a_finite_vector_whose_squared_norm_overflows() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Dripstone::open(dir.path()).expect("open");
    let mut vector = vec![0.0; dripstone::local::LOCAL_EMBED_DIMENSION];
    vector[0] = 1e30;
    let chunk = raw_chunk("huge", vector);
    let error = store
        .backend()
        .upsert(std::slice::from_ref(&chunk))
        .await
        .expect_err("overflowing norm");
    assert!(
        error.to_string().contains("squared norm"),
        "unexpected error: {error}"
    );
    let listed = store.backend().list(None, None).await.expect("list");
    assert!(listed.is_empty());
    drop(store);
    let store = Dripstone::open(dir.path()).expect("reopen");
    let hits = store.search("huge").await.expect("search still runs");
    assert!(hits.is_empty());
}

fn raw_chunk(doc_id: &str, vector: Vec<f32>) -> Chunk {
    Chunk::new(
        doc_id,
        0,
        "text",
        vector,
        0,
        4,
        1,
        serde_json::json!({}),
        None,
    )
}

trait ListDocs {
    async fn list_docs(&self) -> Vec<String>;
}

impl ListDocs for Dripstone<dripstone::OpenBackend, dripstone::LocalEmbedder> {
    #[allow(
        clippy::expect_used,
        reason = "integration-test helper panics with the same context as the tests"
    )]
    async fn list_docs(&self) -> Vec<String> {
        let mut chunks = self.backend().list(None, None).await.expect("list");
        chunks.sort_by(|left, right| left.doc_id.cmp(&right.doc_id));
        chunks.into_iter().map(|chunk| chunk.doc_id).collect()
    }
}

#[allow(
    clippy::expect_used,
    reason = "integration-test helper panics with the same context as the tests"
)]
fn corrupt_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .expect("read dir")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("store.json.corrupt-"))
        })
        .collect();
    paths.sort();
    paths
}

#[allow(
    clippy::expect_used,
    reason = "integration-test helper panics with the same context as the tests"
)]
fn capture_warn<T>(body: impl FnOnce() -> T) -> (T, String) {
    let seen = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let subscriber = WarnCatch(std::sync::Arc::clone(&seen));
    let result = tracing::subscriber::with_default(subscriber, body);
    let warning = seen.lock().expect("warning lock").clone();
    (result, warning)
}

struct WarnCatch(std::sync::Arc<std::sync::Mutex<String>>);

impl tracing::Subscriber for WarnCatch {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        *metadata.level() <= tracing::Level::WARN
    }

    fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }

    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}

    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}

    #[allow(
        clippy::expect_used,
        reason = "integration-test helper panics with the same context as the tests"
    )]
    fn event(&self, event: &tracing::Event<'_>) {
        let mut visitor = MessageVisitor(String::new());
        event.record(&mut visitor);
        if !visitor.0.is_empty() {
            self.0.lock().expect("warning lock").push_str(&visitor.0);
        }
    }

    fn enter(&self, _span: &tracing::span::Id) {}

    fn exit(&self, _span: &tracing::span::Id) {}
}

struct MessageVisitor(String);

impl tracing::field::Visit for MessageVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            use std::fmt::Write;
            let _ = write!(self.0, "{value:?}");
        }
    }
}

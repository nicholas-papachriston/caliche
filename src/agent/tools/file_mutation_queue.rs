//! Serialize file mutations that target the same path (pi-style).
//!
//! Parallel tool calls may edit different files concurrently; same-path write/edit/delete
//! operations are queued so they do not race.

use std::collections::HashMap;
use std::future::Future;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use tokio::sync::Mutex as AsyncMutex;

type PathLock = Arc<AsyncMutex<()>>;

fn global_queue() -> &'static Mutex<HashMap<PathBuf, PathLock>> {
    static QUEUE: OnceLock<Mutex<HashMap<PathBuf, PathLock>>> = OnceLock::new();
    QUEUE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn canonicalize_key(path: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf())
    };
    collapse_dot_components(&absolute)
}

fn collapse_dot_components(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                let _ = out.pop();
            }
            other => out.push(other),
        }
    }
    out
}

fn lock_for_path(path: &Path) -> (PathBuf, PathLock) {
    let key = canonicalize_key(path);
    let mut guard = global_queue()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let lock = Arc::clone(
        guard
            .entry(key.clone())
            .or_insert_with(|| Arc::new(AsyncMutex::new(()))),
    );
    (key, lock)
}

fn evict_if_idle(key: &Path, lock: &PathLock) {
    // Map entry + this stack frame. Anything higher means another caller holds or waits.
    if Arc::strong_count(lock) > 2 {
        return;
    }
    let mut guard = global_queue()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if guard
        .get(key)
        .is_some_and(|existing| Arc::strong_count(existing) <= 2)
    {
        guard.remove(key);
    }
}

/// Run `fut` while holding the per-path mutation lock.
pub async fn with_path_lock<T, F, Fut>(path: &Path, fut: F) -> T
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = T>,
{
    let (key, lock) = lock_for_path(path);
    let output = {
        let _guard = lock.lock().await;
        fut().await
    };
    evict_if_idle(&key, &lock);
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[tokio::test]
    async fn same_path_mutations_serialize() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("file.txt");
        std::fs::write(&path, "start").expect("write");
        let counter = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));

        let c1 = Arc::clone(&counter);
        let m1 = Arc::clone(&max_seen);
        let p1 = path.clone();
        let t1 = tokio::spawn(async move {
            with_path_lock(&p1, || async {
                let now = c1.fetch_add(1, Ordering::SeqCst) + 1;
                m1.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(30)).await;
                c1.fetch_sub(1, Ordering::SeqCst);
            })
            .await;
        });

        let c2 = Arc::clone(&counter);
        let m2 = Arc::clone(&max_seen);
        let p2 = path.clone();
        let t2 = tokio::spawn(async move {
            with_path_lock(&p2, || async {
                let now = c2.fetch_add(1, Ordering::SeqCst) + 1;
                m2.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(30)).await;
                c2.fetch_sub(1, Ordering::SeqCst);
            })
            .await;
        });

        t1.await.expect("t1");
        t2.await.expect("t2");
        assert_eq!(max_seen.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn idle_path_locks_are_evicted() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("file.txt");
        std::fs::write(&path, "start").expect("write");
        with_path_lock(&path, || async {}).await;
        let key = canonicalize_key(&path);
        let present = global_queue()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(&key);
        assert!(!present, "idle mutation lock should not stay in the map");
    }

    #[tokio::test]
    async fn dot_components_share_the_same_path_lock() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("file.txt");
        std::fs::write(&path, "start").expect("write");
        let dotted = dir.path().join(".").join("file.txt");
        assert_eq!(canonicalize_key(&path), canonicalize_key(&dotted));

        let counter = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));
        let c1 = Arc::clone(&counter);
        let m1 = Arc::clone(&max_seen);
        let p1 = path.clone();
        let t1 = tokio::spawn(async move {
            with_path_lock(&p1, || async {
                let now = c1.fetch_add(1, Ordering::SeqCst) + 1;
                m1.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(30)).await;
                c1.fetch_sub(1, Ordering::SeqCst);
            })
            .await;
        });
        let c2 = Arc::clone(&counter);
        let m2 = Arc::clone(&max_seen);
        let t2 = tokio::spawn(async move {
            with_path_lock(&dotted, || async {
                let now = c2.fetch_add(1, Ordering::SeqCst) + 1;
                m2.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(30)).await;
                c2.fetch_sub(1, Ordering::SeqCst);
            })
            .await;
        });
        t1.await.expect("t1");
        t2.await.expect("t2");
        assert_eq!(max_seen.load(Ordering::SeqCst), 1);
    }
}

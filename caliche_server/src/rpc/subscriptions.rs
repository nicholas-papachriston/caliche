use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;

#[derive(Clone, Default)]
pub struct SubscriptionManager {
    tasks: Arc<RwLock<HashMap<String, JoinHandle<()>>>>,
}

impl SubscriptionManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn insert(&self, key: String, task: JoinHandle<()>) {
        let mut guard = self.tasks.write().await;
        if let Some(existing) = guard.insert(key, task) {
            existing.abort();
        }
    }

    pub async fn remove(&self, key: &str) {
        let mut guard = self.tasks.write().await;
        if let Some(task) = guard.remove(key) {
            task.abort();
        }
    }

    pub async fn clear(&self) {
        let mut guard = self.tasks.write().await;
        for (_, task) in guard.drain() {
            task.abort();
        }
    }
}

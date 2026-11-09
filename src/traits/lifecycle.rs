//! Resource lifecycle management abstraction.
//!
//! This module provides a unified interface for managing resource initialization,
//! health checks, and cleanup. Implement the `Resource` trait to add custom
//! resource management for databases, HTTP clients, or other resources.

use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;

#[cfg(feature = "sqlx")]
use sqlx::{PgPool, postgres::PgPoolOptions};

/// Health status of a resource
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// Resource is healthy and ready
    Healthy,
    /// Resource is degraded but still functional
    Degraded,
    /// Resource is unhealthy and should not be used
    Unhealthy,
}

/// Handle to a resource with thread-safe access
pub type ResourceHandle = Arc<RwLock<dyn Resource>>;

/// Collection of resource handles with thread-safe access
pub type ResourceList = Arc<RwLock<Vec<ResourceHandle>>>;

/// Trait for resources that need lifecycle management
#[async_trait]
pub trait Resource: Send + Sync + std::fmt::Debug {
    /// Initialize the resource
    ///
    /// # Returns
    /// * `Ok(())` - Successfully initialized
    /// * `Err` - Failed to initialize
    async fn initialize(&mut self) -> Result<()>;

    /// Check the health of the resource
    ///
    /// # Returns
    /// * `Ok(HealthStatus)` - Current health status
    /// * `Err` - Error checking health
    async fn health_check(&self) -> Result<HealthStatus>;

    /// Cleanup and shutdown the resource
    ///
    /// # Returns
    /// * `Ok(())` - Successfully cleaned up
    /// * `Err` - Error during cleanup
    async fn cleanup(&mut self) -> Result<()>;

    /// Get the name of this resource (for debugging/logging)
    fn resource_name(&self) -> &str;
}

/// Trait for managing multiple resources
#[async_trait]
pub trait ResourceManager: Send + Sync + std::fmt::Debug {
    /// Register a resource for lifecycle management
    ///
    /// # Arguments
    /// * `resource` - Resource to register
    ///
    /// # Returns
    /// * `Ok(())` - Successfully registered
    /// * `Err` - Failed to register
    async fn register(&self, resource: ResourceHandle) -> Result<()>;

    /// Shutdown all registered resources
    ///
    /// # Returns
    /// * `Ok(())` - Successfully shut down all resources
    /// * `Err` - Error shutting down (may have partially completed)
    async fn shutdown_all(&self) -> Result<()>;

    /// Get health status of all resources
    ///
    /// # Returns
    /// Map of resource names to their health status
    async fn health_check_all(&self) -> Result<std::collections::HashMap<String, HealthStatus>>;
}

/// Default resource manager implementation
#[derive(Debug)]
pub struct DefaultResourceManager {
    resources: ResourceList,
}

impl DefaultResourceManager {
    /// Create a new default resource manager
    pub fn new() -> Self {
        Self {
            resources: Arc::new(RwLock::new(Vec::new())),
        }
    }
}

impl Default for DefaultResourceManager {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ResourceManager for DefaultResourceManager {
    async fn register(&self, resource: ResourceHandle) -> Result<()> {
        let mut resources = self.resources.write().await;

        // Initialize the resource before registering
        {
            let mut res = resource.write().await;
            res.initialize().await?;
        }

        resources.push(resource);
        Ok(())
    }

    async fn shutdown_all(&self) -> Result<()> {
        let mut resources = self.resources.write().await;
        let mut errors = Vec::new();

        // Cleanup all resources in reverse order
        for resource in resources.iter().rev() {
            let mut res = resource.write().await;
            if let Err(e) = res.cleanup().await {
                errors.push(anyhow::anyhow!(
                    "Failed to cleanup resource '{}': {}",
                    res.resource_name(),
                    e
                ));
            }
        }

        resources.clear();

        if errors.is_empty() {
            Ok(())
        } else {
            Err(anyhow::anyhow!("Errors during shutdown: {:?}", errors))
        }
    }

    async fn health_check_all(&self) -> Result<std::collections::HashMap<String, HealthStatus>> {
        let resources = self.resources.read().await;
        let mut statuses = std::collections::HashMap::new();

        for resource in resources.iter() {
            let res = resource.read().await;
            match res.health_check().await {
                Ok(status) => {
                    statuses.insert(res.resource_name().to_string(), status);
                }
                Err(e) => {
                    tracing::warn!(
                        "Health check failed for resource '{}': {}",
                        res.resource_name(),
                        e
                    );
                    statuses.insert(res.resource_name().to_string(), HealthStatus::Unhealthy);
                }
            }
        }

        Ok(statuses)
    }
}

/// PostgreSQL resource wrapper
#[cfg(feature = "sqlx")]
#[derive(Debug)]
pub struct PostgresResource {
    database_url: String,
    pool: Option<PgPool>,
    name: String,
}

#[cfg(feature = "sqlx")]
impl PostgresResource {
    /// Create a new PostgreSQL resource
    ///
    /// # Arguments
    /// * `database_url` - PostgreSQL connection URL
    /// * `name` - Resource name for identification
    pub fn new(database_url: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            database_url: database_url.into(),
            pool: None,
            name: name.into(),
        }
    }
}

#[cfg(feature = "sqlx")]
#[async_trait]
impl Resource for PostgresResource {
    async fn initialize(&mut self) -> Result<()> {
        if self.pool.is_none() {
            let pool = PgPoolOptions::new()
                .max_connections(10)
                .connect(&self.database_url)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to connect to PostgreSQL: {}", e))?;
            self.pool = Some(pool);
        }
        Ok(())
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        match &self.pool {
            Some(pool) => {
                sqlx::query("SELECT 1")
                    .execute(pool)
                    .await
                    .map_err(|e| anyhow::anyhow!("PostgreSQL health check failed: {}", e))?;
                Ok(HealthStatus::Healthy)
            }
            None => Ok(HealthStatus::Unhealthy),
        }
    }

    async fn cleanup(&mut self) -> Result<()> {
        if let Some(pool) = self.pool.take() {
            pool.close().await;
        }
        Ok(())
    }

    fn resource_name(&self) -> &str {
        &self.name
    }
}

/// Redis resource wrapper
#[cfg(feature = "redis")]
#[derive(Debug)]
pub struct RedisResource {
    redis_url: String,
    pool: Option<deadpool_redis::Pool>,
    name: String,
}

#[cfg(feature = "redis")]
impl RedisResource {
    /// Create a new Redis resource
    ///
    /// # Arguments
    /// * `redis_url` - Redis connection URL
    /// * `name` - Resource name for identification
    pub fn new(redis_url: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            redis_url: redis_url.into(),
            pool: None,
            name: name.into(),
        }
    }
}

#[cfg(feature = "redis")]
#[async_trait]
impl Resource for RedisResource {
    async fn initialize(&mut self) -> Result<()> {
        if self.pool.is_none() {
            let cfg = deadpool_redis::Config::from_url(self.redis_url.clone());
            let pool = cfg
                .create_pool(Some(deadpool_redis::Runtime::Tokio1))
                .map_err(|e| anyhow::anyhow!("Failed to create Redis pool: {}", e))?;
            self.pool = Some(pool);
        }
        Ok(())
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        match &self.pool {
            Some(pool) => {
                use deadpool_redis::redis::AsyncCommands;
                let mut conn = pool
                    .get()
                    .await
                    .map_err(|e| anyhow::anyhow!("Redis health check failed: {}", e))?;
                let _: String = conn
                    .ping()
                    .await
                    .map_err(|e| anyhow::anyhow!("Redis PING failed: {}", e))?;
                Ok(HealthStatus::Healthy)
            }
            None => Ok(HealthStatus::Unhealthy),
        }
    }

    async fn cleanup(&mut self) -> Result<()> {
        // Pool cleanup happens automatically on drop
        self.pool = None;
        Ok(())
    }

    fn resource_name(&self) -> &str {
        &self.name
    }
}

/// No-op resource for testing
#[derive(Debug, Clone)]
pub struct NoOpResource {
    name: String,
    initialized: bool,
}

impl NoOpResource {
    /// Create a new no-op resource
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            initialized: false,
        }
    }
}

#[async_trait]
impl Resource for NoOpResource {
    async fn initialize(&mut self) -> Result<()> {
        self.initialized = true;
        Ok(())
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        if self.initialized {
            Ok(HealthStatus::Healthy)
        } else {
            Ok(HealthStatus::Unhealthy)
        }
    }

    async fn cleanup(&mut self) -> Result<()> {
        self.initialized = false;
        Ok(())
    }

    fn resource_name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_no_op_resource() {
        let mut resource = NoOpResource::new("test".to_string());

        assert_eq!(
            resource
                .health_check()
                .await
                .expect("health_check should succeed"),
            HealthStatus::Unhealthy
        );

        resource
            .initialize()
            .await
            .expect("initialize should succeed");
        assert_eq!(
            resource
                .health_check()
                .await
                .expect("health_check should succeed"),
            HealthStatus::Healthy
        );

        resource.cleanup().await.expect("cleanup should succeed");
        assert_eq!(
            resource
                .health_check()
                .await
                .expect("health_check should succeed"),
            HealthStatus::Unhealthy
        );
    }

    #[tokio::test]
    async fn test_resource_manager() {
        let manager = DefaultResourceManager::new();

        let resource1 = Arc::new(RwLock::new(NoOpResource::new("resource1".to_string())));
        let resource2 = Arc::new(RwLock::new(NoOpResource::new("resource2".to_string())));

        manager
            .register(resource1.clone())
            .await
            .expect("register should succeed");
        manager
            .register(resource2.clone())
            .await
            .expect("register should succeed");

        let statuses = manager
            .health_check_all()
            .await
            .expect("health_check_all should succeed");
        assert_eq!(statuses.len(), 2);
        assert_eq!(statuses.get("resource1"), Some(&HealthStatus::Healthy));
        assert_eq!(statuses.get("resource2"), Some(&HealthStatus::Healthy));

        manager
            .shutdown_all()
            .await
            .expect("shutdown_all should succeed");

        // After shutdown, resources should still be accessible but may be cleaned up
        let statuses = manager
            .health_check_all()
            .await
            .expect("health_check_all should succeed");
        assert_eq!(statuses.len(), 0);
    }
}

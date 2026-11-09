use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub status: HealthStatus,
    pub checks: Vec<ComponentHealth>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub name: String,
    pub status: HealthStatus,
    pub message: Option<String>,
    pub response_time_ms: u64,
}

pub struct HealthChecker {
    start_time: Instant,
}

impl HealthChecker {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
        }
    }

    pub fn liveness(&self) -> HealthCheck {
        HealthCheck {
            status: HealthStatus::Healthy,
            checks: vec![],
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn readiness(&self) -> HealthCheck {
        let mut checks = Vec::new();

        // Check if service has been running for at least 5 seconds
        let uptime = self.start_time.elapsed();
        if uptime < Duration::from_secs(5) {
            checks.push(ComponentHealth {
                name: "startup".to_string(),
                status: HealthStatus::Degraded,
                message: Some(format!("Still initializing ({}s)", uptime.as_secs())),
                response_time_ms: 0,
            });
        }

        let overall_status = if checks.iter().any(|c| c.status == HealthStatus::Unhealthy) {
            HealthStatus::Unhealthy
        } else if checks.iter().any(|c| c.status == HealthStatus::Degraded) {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };

        HealthCheck {
            status: overall_status,
            checks,
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn startup(&self) -> HealthCheck {
        self.readiness()
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

//! Request-level spend controls (tokens/cost budgets per tenant).

use anyhow::Result;
use async_trait::async_trait;
use chrono::Utc;
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Error when spend budget is exceeded.
#[derive(Debug, Clone)]
pub struct BudgetExceeded {
    pub tenant_id: String,
    pub limit_type: BudgetLimitType,
    pub current: u64,
    pub limit: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetLimitType {
    TokensPerDay,
    CostPerDay,
}

impl std::fmt::Display for BudgetExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Budget exceeded for tenant {}: {} (current={}, limit={})",
            self.tenant_id,
            match self.limit_type {
                BudgetLimitType::TokensPerDay => "tokens/day",
                BudgetLimitType::CostPerDay => "cost/day",
            },
            self.current,
            self.limit
        )
    }
}

impl std::error::Error for BudgetExceeded {}

/// Tracks and enforces per-tenant spend budgets.
#[async_trait]
pub trait SpendTracker: Send + Sync {
    /// Check if tenant can consume estimated tokens. Returns Err if over budget.
    async fn check_budget_tokens(
        &self,
        tenant_id: &str,
        estimated_tokens: u64,
    ) -> Result<(), BudgetExceeded>;

    /// Check if tenant can consume estimated cost (in cents or smallest unit).
    async fn check_budget_cost(
        &self,
        tenant_id: &str,
        estimated_cost: u64,
    ) -> Result<(), BudgetExceeded>;

    /// Record actual token usage after request completes.
    async fn record_tokens(&self, tenant_id: &str, tokens: u64);

    /// Record actual cost after request completes.
    async fn record_cost(&self, tenant_id: &str, cost: u64);
}

/// Per-tenant usage for a day.
struct DayUsage {
    tokens: AtomicU64,
    cost: AtomicU64,
}

/// In-memory spend tracker with daily reset.
#[derive(Clone)]
pub struct InMemorySpendTracker {
    usage: DashMap<String, std::sync::Arc<DayUsage>>,
    tokens_per_day_limit: Option<u64>,
    cost_per_day_limit: Option<u64>,
    /// Current day (YYYY-MM-DD) for reset detection
    current_day: std::sync::Arc<parking_lot::RwLock<String>>,
}

impl InMemorySpendTracker {
    /// Create tracker with optional daily limits.
    pub fn new(tokens_per_day_limit: Option<u64>, cost_per_day_limit: Option<u64>) -> Self {
        Self {
            usage: DashMap::new(),
            tokens_per_day_limit,
            cost_per_day_limit,
            current_day: std::sync::Arc::new(parking_lot::RwLock::new(
                Utc::now().format("%Y-%m-%d").to_string(),
            )),
        }
    }

    /// Create from environment:
    /// - CALICHE_SPEND_BUDGET_PER_TENANT_TOKENS
    /// - CALICHE_SPEND_BUDGET_PER_TENANT_COST
    pub fn from_env() -> Option<Self> {
        let tokens = std::env::var("CALICHE_SPEND_BUDGET_PER_TENANT_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok());
        let cost = std::env::var("CALICHE_SPEND_BUDGET_PER_TENANT_COST")
            .ok()
            .and_then(|s| s.parse().ok());
        if tokens.is_some() || cost.is_some() {
            Some(Self::new(tokens, cost))
        } else {
            None
        }
    }

    fn today() -> String {
        Utc::now().format("%Y-%m-%d").to_string()
    }

    fn maybe_reset(&self, _tenant_id: &str) {
        let today = Self::today();
        {
            let mut day_guard = self.current_day.write();
            if *day_guard != today {
                *day_guard = today;
                self.usage.clear();
            }
        }
    }

    fn get_or_create(&self, tenant_id: &str) -> std::sync::Arc<DayUsage> {
        self.maybe_reset(tenant_id);
        let entry = self.usage.entry(tenant_id.to_string()).or_insert_with(|| {
            std::sync::Arc::new(DayUsage {
                tokens: AtomicU64::new(0),
                cost: AtomicU64::new(0),
            })
        });
        std::sync::Arc::clone(&*entry)
    }
}

#[async_trait]
impl SpendTracker for InMemorySpendTracker {
    async fn check_budget_tokens(
        &self,
        tenant_id: &str,
        estimated_tokens: u64,
    ) -> Result<(), BudgetExceeded> {
        let Some(limit) = self.tokens_per_day_limit else {
            return Ok(());
        };
        let usage = self.get_or_create(tenant_id);
        let current = usage.tokens.load(Ordering::Relaxed);
        if current + estimated_tokens > limit {
            return Err(BudgetExceeded {
                tenant_id: tenant_id.to_string(),
                limit_type: BudgetLimitType::TokensPerDay,
                current,
                limit,
            });
        }
        Ok(())
    }

    async fn check_budget_cost(
        &self,
        tenant_id: &str,
        estimated_cost: u64,
    ) -> Result<(), BudgetExceeded> {
        let Some(limit) = self.cost_per_day_limit else {
            return Ok(());
        };
        let usage = self.get_or_create(tenant_id);
        let current = usage.cost.load(Ordering::Relaxed);
        if current + estimated_cost > limit {
            return Err(BudgetExceeded {
                tenant_id: tenant_id.to_string(),
                limit_type: BudgetLimitType::CostPerDay,
                current,
                limit,
            });
        }
        Ok(())
    }

    async fn record_tokens(&self, tenant_id: &str, tokens: u64) {
        let usage = self.get_or_create(tenant_id);
        usage.tokens.fetch_add(tokens, Ordering::Relaxed);
    }

    async fn record_cost(&self, tenant_id: &str, cost: u64) {
        let usage = self.get_or_create(tenant_id);
        usage.cost.fetch_add(cost, Ordering::Relaxed);
    }
}

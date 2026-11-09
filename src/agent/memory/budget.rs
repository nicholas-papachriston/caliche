//! Context budget allocation across partitions of the context window.

use serde::{Deserialize, Serialize};

/// Allocation strategy for context budget partitioning.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// Fixed percentages for each component
    Fixed {
        system_pct: f32,
        history_pct: f32,
        rag_pct: f32,
        tools_pct: f32,
    },
    /// Dynamic allocation based on query complexity
    Dynamic {
        min_system: usize,
        min_history: usize,
        min_rag: usize,
        reserve_tools: usize,
    },
    /// Adaptive allocation that learns from usage patterns
    Adaptive {
        base_system_pct: f32,
        base_history_pct: f32,
        base_rag_pct: f32,
        base_tools_pct: f32,
    },
}

impl Default for AllocationStrategy {
    fn default() -> Self {
        Self::Fixed {
            system_pct: 0.10,  // 10% for system prompt
            history_pct: 0.50, // 50% for conversation history
            rag_pct: 0.30,     // 30% for RAG context
            tools_pct: 0.10,   // 10% reserved for tool outputs
        }
    }
}

/// Context budget allocation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextBudget {
    /// Total available tokens
    pub total_tokens: usize,
    /// Tokens allocated for system prompt
    pub system_tokens: usize,
    /// Tokens allocated for conversation history
    pub history_tokens: usize,
    /// Tokens allocated for RAG/retrieved context
    pub rag_tokens: usize,
    /// Tokens reserved for tool outputs
    pub tools_tokens: usize,
    /// Remaining tokens after allocation
    pub remaining_tokens: usize,
}

impl ContextBudget {
    /// Create a new budget with fixed allocation.
    #[must_use]
    pub fn new_fixed(
        total_tokens: usize,
        system_pct: f32,
        history_pct: f32,
        rag_pct: f32,
        tools_pct: f32,
    ) -> Self {
        let system_tokens = (total_tokens as f32 * system_pct).floor() as usize;
        let history_tokens = (total_tokens as f32 * history_pct).floor() as usize;
        let rag_tokens = (total_tokens as f32 * rag_pct).floor() as usize;
        let tools_tokens = (total_tokens as f32 * tools_pct).floor() as usize;
        let allocated = system_tokens + history_tokens + rag_tokens + tools_tokens;
        let remaining_tokens = total_tokens.saturating_sub(allocated);

        Self {
            total_tokens,
            system_tokens,
            history_tokens,
            rag_tokens,
            tools_tokens,
            remaining_tokens,
        }
    }

    /// Create a budget with dynamic allocation based on query complexity.
    #[must_use]
    pub fn new_dynamic(
        total_tokens: usize,
        query_complexity: QueryComplexity,
        min_system: usize,
        min_history: usize,
        min_rag: usize,
        reserve_tools: usize,
    ) -> Self {
        let base_allocated = min_system + min_history + min_rag + reserve_tools;
        let available = total_tokens.saturating_sub(base_allocated);

        // Allocate remaining tokens based on query complexity
        let (history_multiplier, rag_multiplier) = match query_complexity {
            QueryComplexity::Simple => (0.6, 0.2),
            QueryComplexity::Moderate => (0.5, 0.4),
            QueryComplexity::Complex => (0.3, 0.6),
            QueryComplexity::VeryComplex => (0.2, 0.7),
        };

        let history_extra = (available as f32 * history_multiplier).floor() as usize;
        let rag_extra = (available as f32 * rag_multiplier).floor() as usize;

        let system_tokens = min_system;
        let history_tokens = min_history + history_extra;
        let rag_tokens = min_rag + rag_extra;
        let tools_tokens = reserve_tools;
        let remaining_tokens = total_tokens
            .saturating_sub(system_tokens)
            .saturating_sub(history_tokens)
            .saturating_sub(rag_tokens)
            .saturating_sub(tools_tokens);

        Self {
            total_tokens,
            system_tokens,
            history_tokens,
            rag_tokens,
            tools_tokens,
            remaining_tokens,
        }
    }

    /// Check if a component has exceeded its budget.
    #[must_use]
    pub const fn is_over_budget(&self, component: BudgetComponent, used: usize) -> bool {
        let budget = match component {
            BudgetComponent::System => self.system_tokens,
            BudgetComponent::History => self.history_tokens,
            BudgetComponent::Rag => self.rag_tokens,
            BudgetComponent::Tools => self.tools_tokens,
        };
        used > budget
    }

    /// Get remaining budget for a component.
    #[must_use]
    pub const fn remaining_for(&self, component: BudgetComponent, used: usize) -> usize {
        let budget = match component {
            BudgetComponent::System => self.system_tokens,
            BudgetComponent::History => self.history_tokens,
            BudgetComponent::Rag => self.rag_tokens,
            BudgetComponent::Tools => self.tools_tokens,
        };
        budget.saturating_sub(used)
    }
}

/// Query complexity estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryComplexity {
    Simple,
    Moderate,
    Complex,
    VeryComplex,
}

impl QueryComplexity {
    /// Estimate complexity from query text.
    #[must_use]
    pub fn from_query(query: &str) -> Self {
        let word_count = query.split_whitespace().count();
        let has_question = query.contains('?');
        let has_multiple_questions = query.matches('?').count() > 1;
        let has_commands =
            query.contains("list") || query.contains("compare") || query.contains("analyze");

        match (
            word_count,
            has_question,
            has_multiple_questions,
            has_commands,
        ) {
            // Complex queries first (they override simple word count)
            (16..=30, _, true, _) | (_, _, _, true) => Self::Complex,
            // Questions with moderate word count
            (6..=15, true, false, false) => Self::Moderate,
            // Short questions should be Moderate, not Simple
            (0..=5, true, false, false) => Self::Moderate,
            // Very short non-questions
            (0..=5, _, _, _) => Self::Simple,
            // Longer queries without questions
            (6..=15, false, _, _) => Self::Moderate,
            _ => Self::VeryComplex,
        }
    }
}

/// Budget component identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetComponent {
    System,
    History,
    Rag,
    Tools,
}

/// Context budget allocator.
#[derive(Clone)]
pub struct ContextBudgetAllocator {
    strategy: AllocationStrategy,
    max_tokens: usize,
}

impl ContextBudgetAllocator {
    /// Create a new allocator with default strategy.
    #[must_use]
    pub fn new(max_tokens: usize) -> Self {
        Self {
            strategy: AllocationStrategy::default(),
            max_tokens,
        }
    }

    /// Create an allocator with custom strategy.
    #[must_use]
    pub const fn with_strategy(max_tokens: usize, strategy: AllocationStrategy) -> Self {
        Self {
            strategy,
            max_tokens,
        }
    }

    /// Allocate budget based on the configured strategy.
    pub fn allocate(&self, query: Option<&str>) -> ContextBudget {
        match self.strategy {
            AllocationStrategy::Fixed {
                system_pct,
                history_pct,
                rag_pct,
                tools_pct,
            } => ContextBudget::new_fixed(
                self.max_tokens,
                system_pct,
                history_pct,
                rag_pct,
                tools_pct,
            ),
            AllocationStrategy::Dynamic {
                min_system,
                min_history,
                min_rag,
                reserve_tools,
            } => {
                let complexity =
                    query.map_or(QueryComplexity::Moderate, QueryComplexity::from_query);
                ContextBudget::new_dynamic(
                    self.max_tokens,
                    complexity,
                    min_system,
                    min_history,
                    min_rag,
                    reserve_tools,
                )
            }
            AllocationStrategy::Adaptive {
                base_system_pct,
                base_history_pct,
                base_rag_pct,
                base_tools_pct,
            } => {
                // For now, adaptive uses fixed percentages
                // Future: could track usage patterns and adjust
                ContextBudget::new_fixed(
                    self.max_tokens,
                    base_system_pct,
                    base_history_pct,
                    base_rag_pct,
                    base_tools_pct,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_allocation() {
        let budget = ContextBudget::new_fixed(10000, 0.10, 0.50, 0.30, 0.10);
        assert_eq!(budget.system_tokens, 1000);
        assert_eq!(budget.history_tokens, 5000);
        assert_eq!(budget.rag_tokens, 3000);
        assert_eq!(budget.tools_tokens, 1000);
    }

    #[test]
    fn test_dynamic_allocation() {
        let budget =
            ContextBudget::new_dynamic(10000, QueryComplexity::Complex, 500, 1000, 500, 500);
        assert!(budget.history_tokens >= 1000);
        assert!(budget.rag_tokens >= 500);
    }

    #[test]
    fn test_query_complexity() {
        assert_eq!(
            QueryComplexity::from_query("Hello"),
            QueryComplexity::Simple
        );
        assert_eq!(
            QueryComplexity::from_query("What is the weather?"),
            QueryComplexity::Moderate
        );
        assert_eq!(
            QueryComplexity::from_query(
                "Compare and analyze the differences between these two approaches?"
            ),
            QueryComplexity::Complex
        );
    }
}

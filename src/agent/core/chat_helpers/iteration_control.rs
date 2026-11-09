#[derive(Debug, Clone)]
pub struct IterationLimiter {
    limit: usize,
    remaining: usize,
}

impl IterationLimiter {
    pub(crate) const fn new(limit: usize) -> Self {
        Self {
            limit,
            remaining: limit,
        }
    }

    pub(crate) const fn limit(&self) -> usize {
        self.limit
    }

    /// Returns `true` if the caller should stop due to reaching the limit.
    /// Mirrors the common pattern:
    /// - if remaining == 0: stop
    /// - else decrement remaining and continue
    pub(crate) const fn tick(&mut self) -> bool {
        if self.limit == usize::MAX {
            return false;
        }

        if self.remaining == 0 {
            return true;
        }

        self.remaining = self.remaining.saturating_sub(1);
        false
    }
}

/// System nudge injected near the hard iteration cap so the model can finish.
pub const SOFT_WRAP_UP_MESSAGE: &str = "You have used many tool call rounds. Synthesize the information you have gathered so far and provide your response. Only make additional tool calls if absolutely necessary to answer the question.";

/// Trailing hint appended when the hard iteration cap is hit.
pub const ITERATION_LIMIT_CONTINUE_HINT: &str = "You can say \"continue\" to keep going.";

/// 1-based iteration at which to inject [`SOFT_WRAP_UP_MESSAGE`].
///
/// The wrap-up tracks the configured cap (80% of `max_iterations`) instead of a
/// fixed 8-round cutoff so long REPL/coding tasks can keep working.
#[must_use]
pub const fn soft_wrap_iteration(max_iterations: usize) -> Option<usize> {
    if max_iterations == 0 || max_iterations == usize::MAX {
        return None;
    }
    let threshold = max_iterations.saturating_mul(4) / 5;
    if threshold == 0 || threshold >= max_iterations {
        None
    } else {
        Some(threshold)
    }
}

/// True when this 1-based iteration should inject the wrap-up nudge.
#[must_use]
pub const fn should_inject_soft_wrap(iteration: usize, max_iterations: usize) -> bool {
    match soft_wrap_iteration(max_iterations) {
        Some(threshold) => threshold == iteration,
        None => false,
    }
}

/// User-visible suffix for a turn that stopped on the iteration cap.
#[must_use]
pub fn iteration_limit_reached_suffix(limit: usize) -> String {
    format!("[Iteration limit ({limit}) reached. {ITERATION_LIMIT_CONTINUE_HINT}]")
}

#[cfg(test)]
mod tests {
    use super::{
        ITERATION_LIMIT_CONTINUE_HINT, iteration_limit_reached_suffix, should_inject_soft_wrap,
        soft_wrap_iteration,
    };

    #[test]
    fn wrap_up_tracks_eighty_percent_of_a_large_budget() {
        assert_eq!(soft_wrap_iteration(100), Some(80));
        assert_eq!(soft_wrap_iteration(50), Some(40));
        assert!(should_inject_soft_wrap(80, 100));
        assert!(!should_inject_soft_wrap(8, 100));
    }

    #[test]
    fn wrap_up_skips_unlimited_and_tiny_budgets() {
        assert_eq!(soft_wrap_iteration(0), None);
        assert_eq!(soft_wrap_iteration(1), None);
        assert_eq!(soft_wrap_iteration(usize::MAX), None);
        assert_eq!(soft_wrap_iteration(8), Some(6));
    }

    #[test]
    fn iteration_limit_suffix_tells_the_user_to_continue() {
        let suffix = iteration_limit_reached_suffix(100);
        assert!(suffix.contains("100"));
        assert!(suffix.contains(ITERATION_LIMIT_CONTINUE_HINT));
    }
}

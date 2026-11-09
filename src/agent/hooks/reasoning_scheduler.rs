//! Reasoning budget scheduler (sandwich / uniform / ramp / custom).
//!
//! Sets a shared [`ReasoningEffortSlot`] before each LLM call so providers can
//! map abstract effort onto native parameters (`reasoning_effort`, thinking
//! budgets, etc.).

use super::lifecycle::{HookContext, HookResult, LifecycleHook, PreLlmContext};
use crate::messages::ReasoningEffort;
use async_trait::async_trait;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Shared slot the agent reads when building [`crate::messages::CompletionRequest`].
#[derive(Debug, Clone, Default)]
pub struct ReasoningEffortSlot {
    inner: Arc<Mutex<Option<ReasoningEffort>>>,
}

impl ReasoningEffortSlot {
    /// Create an empty slot.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Store the effort for the next LLM call (or clear it).
    pub fn set(&self, effort: Option<ReasoningEffort>) {
        *self.inner.lock() = effort;
    }

    /// Current effort hint, if any.
    #[must_use]
    pub fn current(&self) -> Option<ReasoningEffort> {
        *self.inner.lock()
    }
}

/// Built-in reasoning schedule presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum ReasoningPreset {
    /// High → low → high (default research-backed pattern).
    #[default]
    Sandwich,
    /// Constant high.
    UniformHigh,
    /// Constant low.
    UniformLow,
    /// Low → medium → high.
    RampUp,
    /// Explicit custom phase lengths / efforts.
    Custom,
}

/// Configuration for [`ReasoningScheduler`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningScheduleConfig {
    /// When false, the scheduler is a no-op.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Preset pattern (custom overrides apply when `preset == Custom`).
    #[serde(default)]
    pub preset: ReasoningPreset,
    /// Number of initial turns treated as planning.
    #[serde(default = "default_planning_turns")]
    pub planning_turns: usize,
    /// Effort during planning turns.
    #[serde(default = "default_high")]
    pub planning_effort: ReasoningEffort,
    /// Effort during the middle implementation phase.
    #[serde(default = "default_low")]
    pub impl_effort: ReasoningEffort,
    /// Number of final turns treated as verification.
    #[serde(default = "default_verification_turns")]
    pub verification_turns: usize,
    /// Effort during verification turns.
    #[serde(default = "default_high")]
    pub verification_effort: ReasoningEffort,
}

const fn default_true() -> bool {
    true
}
const fn default_planning_turns() -> usize {
    2
}
const fn default_verification_turns() -> usize {
    2
}
const fn default_high() -> ReasoningEffort {
    ReasoningEffort::High
}
const fn default_low() -> ReasoningEffort {
    ReasoningEffort::Low
}

impl Default for ReasoningScheduleConfig {
    fn default() -> Self {
        Self::preset(ReasoningPreset::Sandwich)
    }
}

impl ReasoningScheduleConfig {
    /// Config for a named preset.
    #[must_use]
    pub fn preset(preset: ReasoningPreset) -> Self {
        let mut cfg = Self {
            enabled: true,
            preset,
            planning_turns: 2,
            planning_effort: ReasoningEffort::High,
            impl_effort: ReasoningEffort::Low,
            verification_turns: 2,
            verification_effort: ReasoningEffort::High,
        };
        cfg.apply_preset_efforts();
        cfg
    }

    /// Apply preset effort values when `preset != Custom`.
    ///
    /// Turn counts and `enabled` are left unchanged so TOML can override phase lengths
    /// while still selecting a named effort sandwich/ramp.
    #[must_use]
    pub fn normalize(mut self) -> Self {
        self.apply_preset_efforts();
        self
    }

    const fn apply_preset_efforts(&mut self) {
        match self.preset {
            ReasoningPreset::Custom => {}
            ReasoningPreset::Sandwich => {
                self.planning_effort = ReasoningEffort::High;
                self.impl_effort = ReasoningEffort::Low;
                self.verification_effort = ReasoningEffort::High;
            }
            ReasoningPreset::UniformHigh => {
                self.planning_effort = ReasoningEffort::High;
                self.impl_effort = ReasoningEffort::High;
                self.verification_effort = ReasoningEffort::High;
            }
            ReasoningPreset::UniformLow => {
                self.planning_effort = ReasoningEffort::Low;
                self.impl_effort = ReasoningEffort::Low;
                self.verification_effort = ReasoningEffort::Low;
            }
            ReasoningPreset::RampUp => {
                self.planning_effort = ReasoningEffort::Low;
                self.impl_effort = ReasoningEffort::Medium;
                self.verification_effort = ReasoningEffort::High;
            }
        }
    }

    /// Resolve effort for `turn_count` (1-based) given a step budget.
    #[must_use]
    pub fn effort_for_turn(&self, turn_count: usize, max_iterations: usize) -> ReasoningEffort {
        let turn = turn_count.max(1);
        let planning_end = self.planning_turns.max(1);
        if turn <= planning_end {
            return self.planning_effort;
        }

        let verification_start = max_iterations
            .saturating_sub(self.verification_turns.saturating_sub(1))
            .max(planning_end + 1);
        if turn >= verification_start {
            return self.verification_effort;
        }

        self.impl_effort
    }
}

/// Lifecycle hook that updates [`ReasoningEffortSlot`] each pre-LLM call.
#[derive(Debug)]
pub struct ReasoningScheduler {
    config: ReasoningScheduleConfig,
    slot: Arc<ReasoningEffortSlot>,
    max_iterations: usize,
}

impl ReasoningScheduler {
    /// Create a scheduler bound to a shared effort slot.
    #[must_use]
    pub fn new(
        config: ReasoningScheduleConfig,
        slot: Arc<ReasoningEffortSlot>,
        max_iterations: usize,
    ) -> Self {
        Self {
            config: config.normalize(),
            slot,
            max_iterations: max_iterations.max(1),
        }
    }
}

#[async_trait]
impl LifecycleHook for ReasoningScheduler {
    async fn on_pre_tool_use(&self, _ctx: &HookContext<'_>) -> HookResult {
        HookResult::allow()
    }

    async fn on_pre_llm_call(&self, ctx: &PreLlmContext<'_>) -> Option<String> {
        if !self.config.enabled {
            self.slot.set(None);
            return None;
        }
        let effort = self
            .config
            .effort_for_turn(ctx.turn_count, self.max_iterations);
        self.slot.set(Some(effort));
        Some(format!(
            "[Reasoning effort: {}]\nCalibrate deliberation depth for this turn: low favors concise execution, medium balances speed and verification, and high favors deeper planning or final checks.",
            effort.as_str()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sandwich_phases_map_high_low_high() {
        let cfg = ReasoningScheduleConfig::preset(ReasoningPreset::Sandwich);
        assert_eq!(cfg.effort_for_turn(1, 10), ReasoningEffort::High);
        assert_eq!(cfg.effort_for_turn(2, 10), ReasoningEffort::High);
        assert_eq!(cfg.effort_for_turn(5, 10), ReasoningEffort::Low);
        assert_eq!(cfg.effort_for_turn(9, 10), ReasoningEffort::High);
        assert_eq!(cfg.effort_for_turn(10, 10), ReasoningEffort::High);
    }

    #[test]
    fn ramp_up_phases_map_low_medium_high() {
        let cfg = ReasoningScheduleConfig::preset(ReasoningPreset::RampUp);
        assert_eq!(cfg.effort_for_turn(1, 10), ReasoningEffort::Low);
        assert_eq!(cfg.effort_for_turn(5, 10), ReasoningEffort::Medium);
        assert_eq!(cfg.effort_for_turn(10, 10), ReasoningEffort::High);
    }

    #[test]
    fn uniform_high_is_constant() {
        let cfg = ReasoningScheduleConfig::preset(ReasoningPreset::UniformHigh);
        assert_eq!(cfg.effort_for_turn(1, 6), ReasoningEffort::High);
        assert_eq!(cfg.effort_for_turn(3, 6), ReasoningEffort::High);
        assert_eq!(cfg.effort_for_turn(6, 6), ReasoningEffort::High);
    }

    #[test]
    fn normalize_applies_named_preset_efforts() {
        let cfg = ReasoningScheduleConfig {
            preset: ReasoningPreset::UniformLow,
            planning_effort: ReasoningEffort::High,
            impl_effort: ReasoningEffort::High,
            verification_effort: ReasoningEffort::High,
            planning_turns: 4,
            ..ReasoningScheduleConfig::default()
        }
        .normalize();
        assert_eq!(cfg.planning_turns, 4);
        assert_eq!(cfg.planning_effort, ReasoningEffort::Low);
        assert_eq!(cfg.impl_effort, ReasoningEffort::Low);
        assert_eq!(cfg.verification_effort, ReasoningEffort::Low);
    }

    #[test]
    fn normalize_leaves_custom_efforts_unchanged() {
        let cfg = ReasoningScheduleConfig {
            preset: ReasoningPreset::Custom,
            planning_effort: ReasoningEffort::Medium,
            impl_effort: ReasoningEffort::High,
            verification_effort: ReasoningEffort::Low,
            planning_turns: 3,
            verification_turns: 1,
            enabled: true,
        }
        .normalize();
        assert_eq!(cfg.planning_effort, ReasoningEffort::Medium);
        assert_eq!(cfg.impl_effort, ReasoningEffort::High);
        assert_eq!(cfg.verification_effort, ReasoningEffort::Low);
        assert_eq!(cfg.effort_for_turn(1, 8), ReasoningEffort::Medium);
        assert_eq!(cfg.effort_for_turn(4, 8), ReasoningEffort::High);
        assert_eq!(cfg.effort_for_turn(8, 8), ReasoningEffort::Low);
    }

    #[test]
    fn slot_roundtrip() {
        let slot = ReasoningEffortSlot::new();
        assert_eq!(slot.current(), None);
        slot.set(Some(ReasoningEffort::Medium));
        assert_eq!(slot.current(), Some(ReasoningEffort::Medium));
        slot.set(None);
        assert_eq!(slot.current(), None);
    }

    #[test]
    fn effort_for_turn_zero_counts_as_first_turn() {
        let cfg = ReasoningScheduleConfig::preset(ReasoningPreset::Sandwich);
        assert_eq!(cfg.effort_for_turn(0, 10), ReasoningEffort::High);
        assert_eq!(cfg.effort_for_turn(0, 10), cfg.effort_for_turn(1, 10));
    }

    #[tokio::test]
    async fn disabled_scheduler_clears_slot_and_skips_injection() {
        let slot = Arc::new(ReasoningEffortSlot::new());
        slot.set(Some(ReasoningEffort::High));
        let scheduler = ReasoningScheduler::new(
            ReasoningScheduleConfig {
                enabled: false,
                ..ReasoningScheduleConfig::default()
            },
            Arc::clone(&slot),
            10,
        );
        let ctx = PreLlmContext {
            session_id: "s",
            turn_count: 1,
        };
        let injection = scheduler.on_pre_llm_call(&ctx).await;
        assert!(injection.is_none());
        assert_eq!(slot.current(), None);
    }

    #[tokio::test]
    async fn enabled_scheduler_sets_slot_for_turn() {
        let slot = Arc::new(ReasoningEffortSlot::new());
        let scheduler = ReasoningScheduler::new(
            ReasoningScheduleConfig::preset(ReasoningPreset::Sandwich),
            Arc::clone(&slot),
            10,
        );
        let ctx = PreLlmContext {
            session_id: "s",
            turn_count: 5,
        };
        let injection = scheduler
            .on_pre_llm_call(&ctx)
            .await
            .expect("enabled scheduler should inject");
        assert!(injection.contains("low"));
        assert_eq!(slot.current(), Some(ReasoningEffort::Low));
    }
}

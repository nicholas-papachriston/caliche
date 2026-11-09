//! Process-wide hotpath performance instrumentation.
//!
//! The fast path is a single `Relaxed` atomic load. When disabled there are no
//! clocks, locks, allocations, or tracing events.
//!
//! Defaults:
//! - debug builds (`cfg!(debug_assertions)`): enabled
//! - release builds: disabled
//!
//! Overrides (highest priority first):
//! - CLI `--no-perf` / `configure(false)`
//! - `CALICHE_PERF=0|1|true|false|on|off`
//! - CLI `--debug` (enables unless `--no-perf` or env disables)
//!
//! Events are emitted on the tracing target `caliche::perf` at `debug`.

use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

/// Environment toggle for hotpath measurement.
pub const ENV_VAR: &str = "CALICHE_PERF";
/// Tracing target for hotpath events.
pub const TARGET: &str = "caliche::perf";
const EVENT_CAP: usize = 256;

static ENABLED: AtomicBool = AtomicBool::new(cfg!(debug_assertions));
static CONFIGURED: AtomicBool = AtomicBool::new(false);
static EVENT_COUNT: AtomicU64 = AtomicU64::new(0);
/// Bumped by [`clear`]. A span records only into the generation it started in.
static GENERATION: AtomicU64 = AtomicU64::new(0);
static EVENTS: Mutex<VecDeque<PerfEvent>> = Mutex::new(VecDeque::new());

/// One recorded hotpath sample.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerfEvent {
    /// Stable hotpath name, e.g. `fs.grep` or `edit.apply`.
    pub name: &'static str,
    /// Elapsed microseconds.
    pub duration_us: u64,
    /// Files touched or scanned.
    pub file_count: u64,
    /// Matches, operations, or hits depending on the hotpath.
    pub match_count: u64,
    /// Bytes read or written when the hotpath reports them.
    pub byte_count: u64,
    /// Lines scanned when the hotpath reports them.
    pub line_count: u64,
}

/// Parse `CALICHE_PERF` values. Unknown strings yield `None`.
#[must_use]
pub fn parse_env(raw: &str) -> Option<bool> {
    let trimmed = raw.trim();
    if trimmed.eq_ignore_ascii_case("1")
        || trimmed.eq_ignore_ascii_case("true")
        || trimmed.eq_ignore_ascii_case("on")
        || trimmed.eq_ignore_ascii_case("yes")
    {
        return Some(true);
    }
    if trimmed.eq_ignore_ascii_case("0")
        || trimmed.eq_ignore_ascii_case("false")
        || trimmed.eq_ignore_ascii_case("off")
        || trimmed.eq_ignore_ascii_case("no")
    {
        return Some(false);
    }
    None
}

/// Resolve whether measurement should run for a CLI invocation.
#[must_use]
pub fn resolve(debug: bool, no_perf: bool) -> bool {
    resolve_with_env(debug, no_perf, std::env::var(ENV_VAR).ok().as_deref())
}

#[must_use]
fn resolve_with_env(debug: bool, no_perf: bool, env: Option<&str>) -> bool {
    if no_perf {
        return false;
    }
    if let Some(raw) = env
        && let Some(value) = parse_env(raw)
    {
        return value;
    }
    debug || cfg!(debug_assertions)
}

/// Set the process-wide gate. Call once at CLI startup.
pub fn configure(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
    CONFIGURED.store(true, Ordering::Relaxed);
}

/// Apply CLI flags (`--debug`, `--no-perf`) and `CALICHE_PERF`.
pub fn configure_from_cli(debug: bool, no_perf: bool) {
    configure(resolve(debug, no_perf));
}

fn hydrate_default() {
    if CONFIGURED.load(Ordering::Relaxed) {
        return;
    }
    let on = match std::env::var(ENV_VAR) {
        Ok(raw) => parse_env(&raw).unwrap_or(cfg!(debug_assertions)),
        Err(_) => cfg!(debug_assertions),
    };
    ENABLED.store(on, Ordering::Relaxed);
    CONFIGURED.store(true, Ordering::Relaxed);
}

/// Whether the current process should record hotpath spans.
#[inline]
#[must_use]
pub fn enabled() -> bool {
    if !CONFIGURED.load(Ordering::Relaxed) {
        hydrate_default();
    }
    ENABLED.load(Ordering::Relaxed)
}

/// In-flight hotpath sample. Records on drop when measurement is enabled.
#[derive(Debug)]
pub struct PerfSpan {
    name: &'static str,
    start: Instant,
    generation: u64,
    file_count: u64,
    match_count: u64,
    byte_count: u64,
    line_count: u64,
}

impl PerfSpan {
    /// Files scanned or mutated.
    pub const fn set_file_count(&mut self, count: u64) {
        self.file_count = count;
    }

    /// Matches, operations, or search hits.
    pub const fn set_match_count(&mut self, count: u64) {
        self.match_count = count;
    }

    /// Bytes read or written.
    pub const fn set_byte_count(&mut self, count: u64) {
        self.byte_count = count;
    }

    /// Lines scanned.
    pub const fn set_line_count(&mut self, count: u64) {
        self.line_count = count;
    }
}

impl Drop for PerfSpan {
    fn drop(&mut self) {
        let duration_us = u64::try_from(self.start.elapsed().as_micros()).unwrap_or(u64::MAX);
        let event = PerfEvent {
            name: self.name,
            duration_us,
            file_count: self.file_count,
            match_count: self.match_count,
            byte_count: self.byte_count,
            line_count: self.line_count,
        };
        {
            // Same lock as `clear`, so a span that started before a clear
            // cannot record into the cleared count.
            let mut events = EVENTS.lock();
            if GENERATION.load(Ordering::Relaxed) != self.generation {
                return;
            }
            EVENT_COUNT.fetch_add(1, Ordering::Relaxed);
            if events.len() >= EVENT_CAP {
                events.pop_front();
            }
            events.push_back(event.clone());
        }
        tracing::debug!(
            target: TARGET,
            hotpath = event.name,
            duration_us = event.duration_us,
            file_count = event.file_count,
            match_count = event.match_count,
            byte_count = event.byte_count,
            line_count = event.line_count,
            "hotpath"
        );
    }
}

/// Start a hotpath span. Returns `None` when measurement is off (no clock).
#[inline]
#[must_use]
pub fn span(name: &'static str) -> Option<PerfSpan> {
    if !enabled() {
        return None;
    }
    // Read the generation, then check the gate again. A `configure(false)`
    // made before a `clear` this load observes is then visible, so a span
    // cannot start in the new generation while measurement is off.
    let generation = GENERATION.load(Ordering::Acquire);
    if !ENABLED.load(Ordering::Relaxed) {
        return None;
    }
    Some(PerfSpan {
        name,
        start: Instant::now(),
        generation,
        file_count: 0,
        match_count: 0,
        byte_count: 0,
        line_count: 0,
    })
}

/// Number of events recorded since process start or [`clear`].
#[must_use]
pub fn event_count() -> u64 {
    EVENT_COUNT.load(Ordering::Relaxed)
}

/// Copy of the recent event ring (newest last).
#[must_use]
pub fn snapshot() -> Vec<PerfEvent> {
    EVENTS.lock().iter().cloned().collect()
}

/// Drop recorded events, and spans still in flight. Intended for tests.
pub fn clear() {
    let mut events = EVENTS.lock();
    events.clear();
    EVENT_COUNT.store(0, Ordering::Relaxed);
    GENERATION.fetch_add(1, Ordering::Release);
}

#[cfg(test)]
pub(crate) fn lock_for_test() -> parking_lot::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn parse_env_accepts_common_toggles() {
        assert_eq!(parse_env("1"), Some(true));
        assert_eq!(parse_env("TRUE"), Some(true));
        assert_eq!(parse_env("off"), Some(false));
        assert_eq!(parse_env("0"), Some(false));
        assert_eq!(parse_env("maybe"), None);
        assert_eq!(parse_env(""), None);
    }

    #[test]
    fn resolve_honors_no_perf_ahead_of_debug() {
        assert!(!resolve_with_env(true, true, None));
        assert!(resolve_with_env(true, false, None));
        assert!(!resolve_with_env(false, true, None));
        assert!(!resolve_with_env(true, false, Some("0")));
        assert!(resolve_with_env(false, false, Some("1")));
    }

    #[test]
    fn a_span_started_before_clear_does_not_record() {
        let _lock = lock_for_test();
        configure(true);
        let in_flight = span("test.in_flight");
        assert!(in_flight.is_some());
        configure(false);
        clear();
        drop(in_flight);
        assert_eq!(event_count(), 0);
        assert_eq!(snapshot(), Vec::new());
    }

    #[test]
    fn disabled_span_is_none_and_records_nothing() {
        let _lock = lock_for_test();
        configure(false);
        clear();
        let before = event_count();
        assert!(span("test.disabled").is_none());
        assert_eq!(event_count(), before);
    }

    #[test]
    fn enabled_span_records_counts() {
        let _lock = lock_for_test();
        configure(true);
        clear();
        {
            let mut sample = span("test.enabled").expect("span when enabled");
            sample.set_file_count(3);
            sample.set_match_count(7);
            sample.set_line_count(11);
            sample.set_byte_count(42);
        }
        let events = snapshot();
        let last = events.last().expect("recorded event");
        assert_eq!(last.name, "test.enabled");
        assert_eq!(last.file_count, 3);
        assert_eq!(last.match_count, 7);
        assert_eq!(last.line_count, 11);
        assert_eq!(last.byte_count, 42);
        configure(false);
        clear();
    }

    #[test]
    fn disabled_enabled_check_is_cheap() {
        let _lock = lock_for_test();
        configure(false);
        let started = Instant::now();
        for _ in 0..100_000 {
            assert!(!enabled());
            let _ = span("bench.disabled");
        }
        let elapsed = started.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "disabled hotpath gate took {elapsed:?} for 100k checks"
        );
    }
}

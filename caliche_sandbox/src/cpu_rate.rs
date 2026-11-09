//! Windows Job Object CPU-rate hard cap.
//!
//! [`crate::config::ResourceLimits::cpu_rate_hard_cap_percent`] defaults to
//! `None`. `None` does not set `JOBOBJECT_CPU_RATE_CONTROL_INFORMATION`, so
//! the job has no CPU-rate limit. A percentage from 1 through 100 becomes a
//! hard cap: `JOB_OBJECT_CPU_RATE_CONTROL_ENABLE` plus
//! `JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP`, with `CpuRate = percent * 100`
//! (hundredths of a percent of processor cycles, 10_000 = 100%).
//!
//! The percentage is of the whole machine, every logical processor, not of
//! one CPU. `CpuRate` 100 is the 1% ceiling. On a 16-CPU host that ceiling
//! measured about 0.1% of the machine and did not run over. The formula stays
//! `percent * 100`; a larger multiplier would raise the ceiling.

use anyhow::Result;

/// `JOB_OBJECT_CPU_RATE_CONTROL_ENABLE`.
pub(crate) const CPU_RATE_CONTROL_ENABLE: u32 = 0x1;
/// `JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP`.
pub(crate) const CPU_RATE_CONTROL_HARD_CAP: u32 = 0x4;

/// Fields copied into `JOBOBJECT_CPU_RATE_CONTROL_INFORMATION` on Windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CpuRateHardCap {
    /// `CPU_RATE_CONTROL_ENABLE | CPU_RATE_CONTROL_HARD_CAP`.
    pub(crate) control_flags: u32,
    /// Hundredths of a percent. 2_000 is 20%.
    pub(crate) cpu_rate: u32,
}

/// Map a config percentage to a hard-cap struct.
///
/// `None` means the caller must not call `SetInformationJobObject` for
/// `JobObjectCpuRateControlInformation`.
///
/// # Errors
///
/// Returns an error when `percent` is present and outside 1 through 100.
pub(crate) fn cpu_rate_hard_cap(percent: Option<u8>) -> Result<Option<CpuRateHardCap>> {
    match percent {
        None => Ok(None),
        Some(percent) if (1..=100).contains(&percent) => Ok(Some(CpuRateHardCap {
            control_flags: CPU_RATE_CONTROL_ENABLE | CPU_RATE_CONTROL_HARD_CAP,
            cpu_rate: u32::from(percent) * 100,
        })),
        Some(percent) => anyhow::bail!(
            "cpu_rate_hard_cap_percent must be 1 through 100; {percent} is not a Windows CPU hard cap"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{CPU_RATE_CONTROL_ENABLE, CPU_RATE_CONTROL_HARD_CAP, cpu_rate_hard_cap};

    #[test]
    fn absent_percentage_leaves_cpu_rate_control_off() {
        assert_eq!(cpu_rate_hard_cap(None).expect("none"), None);
    }

    #[test]
    fn twenty_percent_is_a_hard_cap_of_two_thousand() {
        let cap = cpu_rate_hard_cap(Some(20)).expect("20").expect("some");
        assert_eq!(
            cap.control_flags,
            CPU_RATE_CONTROL_ENABLE | CPU_RATE_CONTROL_HARD_CAP
        );
        assert_eq!(cap.control_flags, 0x5);
        assert_eq!(cap.cpu_rate, 2_000);
    }

    #[test]
    fn one_and_one_hundred_percent_are_the_inclusive_bounds() {
        let low = cpu_rate_hard_cap(Some(1)).expect("1").expect("some");
        let high = cpu_rate_hard_cap(Some(100)).expect("100").expect("some");
        assert_eq!((low.cpu_rate, high.cpu_rate), (100, 10_000));
    }

    #[test]
    fn zero_and_over_one_hundred_are_rejected() {
        let zero = cpu_rate_hard_cap(Some(0)).expect_err("0");
        let over = cpu_rate_hard_cap(Some(101)).expect_err("101");
        assert!(zero.to_string().contains("1 through 100"));
        assert!(over.to_string().contains("1 through 100"));
    }
}

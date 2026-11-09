fn main() {
    #[cfg(target_os = "linux")]
    linux::emit_test_cfgs();
}

#[cfg(target_os = "linux")]
mod linux {
    use landlock::{ABI, AccessFs, AccessNet, CompatLevel, Compatible, Ruleset, RulesetAttr};

    const REQUIRE_CGROUP_TESTS: &str = "CLS_REQUIRE_CGROUP_TESTS";

    pub fn emit_test_cfgs() {
        // Re-probe when the script or PATH changes. A missing Landlock ABI or
        // python3 must keep the TCP test ignored instead of a silent pass.
        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rerun-if-env-changed=PATH");
        // Required mode must re-probe, not reuse a cfg from an earlier build
        // that ran without access to cgroup v2.
        println!("cargo:rerun-if-env-changed={REQUIRE_CGROUP_TESTS}");
        println!("cargo::rustc-check-cfg=cfg(sandbox_landlock_tcp_available)");
        println!("cargo::rustc-check-cfg=cfg(sandbox_python3_available)");
        println!("cargo::rustc-check-cfg=cfg(sandbox_exec_cgroup_available)");

        let filesystem = Ruleset::default()
            .set_compatibility(CompatLevel::HardRequirement)
            .handle_access(AccessFs::from_read(ABI::V5))
            .and_then(|ruleset| ruleset.create())
            .is_ok();
        let network = Ruleset::default()
            .set_compatibility(CompatLevel::HardRequirement)
            .handle_access(AccessNet::BindTcp | AccessNet::ConnectTcp)
            .and_then(|ruleset| ruleset.create())
            .is_ok();
        if filesystem && network {
            println!("cargo:rustc-cfg=sandbox_landlock_tcp_available");
        }

        let python3 = std::process::Command::new("python3")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|status| status.success());
        if python3 {
            println!("cargo:rustc-cfg=sandbox_python3_available");
        }

        if exec_cgroup_available() {
            println!("cargo:rustc-cfg=sandbox_exec_cgroup_available");
        } else {
            println!(
                "cargo:warning=cgroup containment tests in tests/linux.rs are skipped: no writable cgroup v2 with cgroup.kill. Set {REQUIRE_CGROUP_TESTS}=1 to make the skip fail the test run."
            );
        }
    }

    /// True when this process may create a cgroup with `cgroup.kill` under its
    /// own cgroup and move processes into it, as the Linux backend does.
    fn exec_cgroup_available() -> bool {
        let Ok(membership) = std::fs::read_to_string("/proc/self/cgroup") else {
            return false;
        };
        let Some(relative) = membership.lines().find_map(|line| line.strip_prefix("0::")) else {
            return false;
        };
        let parent = std::path::Path::new("/sys/fs/cgroup").join(relative.trim_start_matches('/'));
        let writable = std::fs::OpenOptions::new()
            .write(true)
            .open(parent.join("cgroup.procs"))
            .is_ok();
        let probe = parent.join(format!("caliche-build-probe-{}", std::process::id()));
        if !writable || std::fs::create_dir(&probe).is_err() {
            return false;
        }
        let killable = probe.join("cgroup.kill").exists();
        let _ = std::fs::remove_dir(&probe);
        killable
    }
}

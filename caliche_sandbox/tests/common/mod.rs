// One Landlock skip gate for the sandbox tests.
//
// Integration binaries and the Linux backend unit tests include this file.
// The flag is exactly `1`. `0` and an unset variable skip when Landlock is
// missing.

pub const CLS_REQUIRE_LANDLOCK_TESTS: &str = "CLS_REQUIRE_LANDLOCK_TESTS";

fn value_is_required(value: Option<&str>) -> bool {
    matches!(value, Some("1"))
}

pub fn landlock_tests_required() -> bool {
    value_is_required(std::env::var(CLS_REQUIRE_LANDLOCK_TESTS).ok().as_deref())
}

/// Panic when the require flag is `1` and Landlock is missing.
///
/// Call this only after creation failed because Landlock is unavailable.
pub fn require_landlock() {
    assert!(
        !landlock_tests_required(),
        "{CLS_REQUIRE_LANDLOCK_TESTS}=1, but Landlock is not available"
    );
}

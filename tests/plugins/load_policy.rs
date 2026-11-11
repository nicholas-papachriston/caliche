use caliche::plugins::PluginLoadPolicy;
use serial_test::serial;

#[test]
fn default_is_warn() {
    assert_eq!(PluginLoadPolicy::default(), PluginLoadPolicy::Warn);
}

#[test]
#[serial]
fn from_env_defaults_to_warn_when_unset() {
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    env.remove("CALICHE_PLUGIN_LOAD_POLICY");
    assert_eq!(PluginLoadPolicy::from_env(), PluginLoadPolicy::Warn);
}

#[test]
#[serial]
fn from_env_parses_fail() {
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    env.set("CALICHE_PLUGIN_LOAD_POLICY", "fail");
    assert_eq!(PluginLoadPolicy::from_env(), PluginLoadPolicy::Fail);
    env.remove("CALICHE_PLUGIN_LOAD_POLICY");
}

#[test]
#[serial]
fn from_env_unknown_values_default_to_warn() {
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    env.set("CALICHE_PLUGIN_LOAD_POLICY", "invalid");
    assert_eq!(PluginLoadPolicy::from_env(), PluginLoadPolicy::Warn);
    env.remove("CALICHE_PLUGIN_LOAD_POLICY");
}

use caliche::plugins::PluginLoadPolicy;
use serial_test::serial;

#[test]
fn default_is_warn() {
    assert_eq!(PluginLoadPolicy::default(), PluginLoadPolicy::Warn);
}

#[test]
#[serial]
#[expect(
    unsafe_code,
    clippy::disallowed_methods,
    reason = "test sets and restores process env vars"
)]
fn from_env_defaults_to_warn_when_unset() {
    unsafe {
        std::env::remove_var("CALICHE_PLUGIN_LOAD_POLICY");
    }
    assert_eq!(PluginLoadPolicy::from_env(), PluginLoadPolicy::Warn);
}

#[test]
#[serial]
#[expect(
    unsafe_code,
    clippy::disallowed_methods,
    reason = "test sets and restores process env vars"
)]
fn from_env_parses_fail() {
    unsafe {
        std::env::set_var("CALICHE_PLUGIN_LOAD_POLICY", "fail");
    }
    assert_eq!(PluginLoadPolicy::from_env(), PluginLoadPolicy::Fail);
    unsafe {
        std::env::remove_var("CALICHE_PLUGIN_LOAD_POLICY");
    }
}

#[test]
#[serial]
#[expect(
    unsafe_code,
    clippy::disallowed_methods,
    reason = "test sets and restores process env vars"
)]
fn from_env_unknown_values_default_to_warn() {
    unsafe {
        std::env::set_var("CALICHE_PLUGIN_LOAD_POLICY", "invalid");
    }
    assert_eq!(PluginLoadPolicy::from_env(), PluginLoadPolicy::Warn);
    unsafe {
        std::env::remove_var("CALICHE_PLUGIN_LOAD_POLICY");
    }
}

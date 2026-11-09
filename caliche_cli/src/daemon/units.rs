//! Render OS unit files for the Caliche server daemon.

use std::path::PathBuf;

/// Spec describing how the user daemon should launch the server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DaemonUnitSpec {
    /// launchd label / systemd unit basename (without `.service`).
    pub label: String,
    /// Absolute path to the `caliche` binary.
    pub exe: PathBuf,
    /// HTTP listen port (`--port`).
    pub port: u16,
    /// Working directory for the daemon process.
    pub cwd: PathBuf,
    /// Optional log directory (macOS StandardOut/Err paths).
    pub log_dir: PathBuf,
}

impl DaemonUnitSpec {
    /// Systemd unit filename (`{label}.service`).
    #[must_use]
    pub fn systemd_unit_name(&self) -> String {
        format!("{}.service", self.label)
    }

    /// launchd plist filename (`{label}.plist`).
    #[must_use]
    pub fn launchd_plist_name(&self) -> String {
        format!("{}.plist", self.label)
    }
}

/// Render a launchd user agent plist.
#[must_use]
pub fn render_launchd_plist(spec: &DaemonUnitSpec) -> String {
    let exe = xml_escape(&spec.exe.display().to_string());
    let cwd = xml_escape(&spec.cwd.display().to_string());
    let label = xml_escape(&spec.label);
    let stdout = xml_escape(
        &spec
            .log_dir
            .join("caliche.daemon.out.log")
            .display()
            .to_string(),
    );
    let stderr = xml_escape(
        &spec
            .log_dir
            .join("caliche.daemon.err.log")
            .display()
            .to_string(),
    );
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>{label}</string>
	<key>ProgramArguments</key>
	<array>
		<string>{exe}</string>
		<string>--server</string>
		<string>--port</string>
		<string>{port}</string>
	</array>
	<key>WorkingDirectory</key>
	<string>{cwd}</string>
	<key>EnvironmentVariables</key>
	<dict>
		<key>{origin_env}</key>
		<string>daemon</string>
	</dict>
	<key>RunAtLoad</key>
	<true/>
	<key>KeepAlive</key>
	<true/>
	<key>StandardOutPath</key>
	<string>{stdout}</string>
	<key>StandardErrorPath</key>
	<string>{stderr}</string>
</dict>
</plist>
"#,
        port = spec.port,
        origin_env = crate::server_registry::SERVER_ORIGIN_ENV,
    )
}

/// Render a systemd user unit.
#[must_use]
pub fn render_systemd_user_unit(spec: &DaemonUnitSpec) -> String {
    let exe = spec.exe.display();
    let cwd = spec.cwd.display();
    format!(
        "[Unit]\n\
Description=Caliche HTTP server\n\
After=default.target\n\
\n\
[Service]\n\
Type=simple\n\
WorkingDirectory={cwd}\n\
Environment={origin_env}=daemon\n\
ExecStart={exe} --server --port {port}\n\
Restart=on-failure\n\
RestartSec=2\n\
\n\
[Install]\n\
WantedBy=default.target\n",
        port = spec.port,
        origin_env = crate::server_registry::SERVER_ORIGIN_ENV,
    )
}

fn xml_escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            other => out.push(other),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn sample_spec() -> DaemonUnitSpec {
        DaemonUnitSpec {
            label: "com.caliche.server".to_string(),
            exe: PathBuf::from("/usr/local/bin/caliche"),
            port: 3000,
            cwd: PathBuf::from("/Users/demo/project"),
            log_dir: PathBuf::from("/Users/demo/.caliche/logs"),
        }
    }

    #[test]
    fn launchd_plist_contains_server_argv_and_escaped_paths() {
        let mut spec = sample_spec();
        spec.cwd = PathBuf::from("/tmp/a&b");
        let plist = render_launchd_plist(&spec);
        assert!(plist.contains("<string>--server</string>"));
        assert!(plist.contains("<string>--port</string>"));
        assert!(plist.contains("<string>3000</string>"));
        assert!(plist.contains("/usr/local/bin/caliche"));
        assert!(plist.contains("/tmp/a&amp;b"));
        assert!(plist.contains("KeepAlive"));
        assert!(plist.contains(crate::server_registry::SERVER_ORIGIN_ENV));
        assert!(plist.contains("<string>daemon</string>"));
    }

    #[test]
    fn systemd_unit_contains_execstart_and_restart() {
        let unit = render_systemd_user_unit(&sample_spec());
        assert!(unit.contains("ExecStart=/usr/local/bin/caliche --server --port 3000"));
        assert!(unit.contains("WorkingDirectory=/Users/demo/project"));
        assert!(unit.contains(&format!(
            "Environment={}={}",
            crate::server_registry::SERVER_ORIGIN_ENV,
            "daemon"
        )));
        assert!(unit.contains("Restart=on-failure"));
        assert!(unit.contains("WantedBy=default.target"));
    }
}

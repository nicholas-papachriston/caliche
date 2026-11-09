//! Logging initialization for CLI.

use crate::commands::Args;
use std::path::PathBuf;
use std::sync::OnceLock;

/// Directives for `-q`. `sqlx` stays at error so statement DEBUG cannot flood a pipe.
pub const QUIET_LOG_DIRECTIVES: &str = "warn,sqlx=error,sqlx_core=error";

static LOGGING_INIT: OnceLock<()> = OnceLock::new();

/// Logging configuration.
#[derive(Debug, Clone)]
pub struct LogConfig {
    /// Log level (debug, info, warn, error).
    pub level: Option<String>,
    /// Whether interactive (REPL) mode - log to file to avoid terminal interference.
    pub use_interactive: bool,
    /// Whether debug mode is enabled.
    pub debug: bool,
    /// Custom log file path (if any).
    pub log_file: Option<PathBuf>,
    /// Quiet mode: stderr stays pipe-friendly and ignores `RUST_LOG` debug floods.
    pub quiet: bool,
}

impl LogConfig {
    /// Create log config from command line arguments.
    pub fn from_args(args: &Args) -> Self {
        if args.quiet {
            return Self {
                level: Some("warn".to_string()),
                use_interactive: false,
                debug: false,
                log_file: None,
                quiet: true,
            };
        }

        let level = if std::env::var("RUST_LOG").is_ok() {
            None // Let EnvFilter handle it
        } else if args.verbose || args.debug {
            Some("debug".to_string())
        } else {
            Some("info".to_string())
        };

        Self {
            level,
            use_interactive: args.message.is_none() && !args.server,
            debug: args.debug,
            log_file: None,
            quiet: false,
        }
    }

    /// Get the log directory.
    pub fn log_dir() -> PathBuf {
        dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("caliche")
            .join("logs")
    }

    /// Generate a timestamped log file name.
    pub fn timestamped_log_file(prefix: &str) -> PathBuf {
        let timestamp = chrono::Utc::now().format("%Y%m%d-%H%M%S");
        Self::log_dir().join(format!("{}-{}.log", prefix, timestamp))
    }
}

/// Initialize logging based on configuration.
///
/// When `use_interactive` is true and a log file is created successfully, returns
/// `Ok(Some(path))` so the caller can print the path via serialized output (e.g.
/// `with_stdout`) to avoid terminal interleaving. Returns `Ok(None)` otherwise.
pub fn init_logging(config: &LogConfig) -> anyhow::Result<Option<PathBuf>> {
    if LOGGING_INIT.get().is_some() {
        return Ok(None);
    }
    let path = install_logging(config)?;
    let _ = LOGGING_INIT.set(());
    Ok(path)
}

fn install_logging(config: &LogConfig) -> anyhow::Result<Option<PathBuf>> {
    use tracing_subscriber::EnvFilter;

    // Build filter. Quiet mode does not consult RUST_LOG, so sqlx DEBUG cannot
    // flood stdout or stderr on a piped command.
    let mut filter = if config.quiet {
        EnvFilter::new(QUIET_LOG_DIRECTIVES)
    } else {
        EnvFilter::from_default_env()
    };
    if !config.quiet
        && let Some(ref level) = config.level
    {
        if let Ok(directive) = format!("caliche={}", level).parse() {
            filter = filter.add_directive(directive);
        }
    }
    if !config.quiet
        && caliche::observability::perf::enabled()
        && let Ok(directive) = "caliche::perf=debug".parse()
    {
        filter = filter.add_directive(directive);
    }

    if config.use_interactive {
        // In interactive (REPL) mode, log to file to avoid terminal interference
        let log_dir = LogConfig::log_dir();
        let _ = std::fs::create_dir_all(&log_dir);

        let log_file = if config.debug {
            LogConfig::timestamped_log_file("debug")
        } else {
            LogConfig::timestamped_log_file("repl")
        };

        match open_private_log(&log_file) {
            Ok(file) => {
                tracing_subscriber::fmt()
                    .with_env_filter(filter)
                    .with_writer(std::sync::Mutex::new(file))
                    .with_target(false)
                    .init();
                // Return path for caller to print via with_stdout; avoids stderr/stdout interleaving
                return Ok(Some(log_file));
            }
            Err(e) => {
                eprintln!(
                    "Warning: Failed to create log file {}: {}. Logging to stderr.",
                    log_file.display(),
                    e
                );
                tracing_subscriber::fmt().with_env_filter(filter).init();
            }
        }
        return Ok(None);
    }

    // Non-interactive mode
    if config.debug {
        // Write to both file and stderr when --debug is set
        let log_dir = LogConfig::log_dir();
        let _ = std::fs::create_dir_all(&log_dir);
        let log_file = LogConfig::timestamped_log_file("debug");

        if let Ok(file) = open_private_log(&log_file) {
            use std::io::Write;
            struct DualWriter {
                file: std::sync::Mutex<std::fs::File>,
                stderr: std::io::Stderr,
            }
            impl Write for DualWriter {
                fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                    let mut file = self
                        .file
                        .lock()
                        .map_err(|_| std::io::Error::other("DualWriter file lock poisoned"))?;
                    file.write_all(buf)?;
                    let mut stderr = self.stderr.lock();
                    stderr.write_all(buf)?;
                    Ok(buf.len())
                }
                fn flush(&mut self) -> std::io::Result<()> {
                    let mut file = self
                        .file
                        .lock()
                        .map_err(|_| std::io::Error::other("DualWriter file lock poisoned"))?;
                    file.flush()?;
                    let mut stderr = self.stderr.lock();
                    stderr.flush()
                }
            }
            let dual_writer = DualWriter {
                file: std::sync::Mutex::new(file),
                stderr: std::io::stderr(),
            };
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_writer(std::sync::Mutex::new(dual_writer))
                .init();
            println!(
                "Debug logging enabled. Writing logs to: {}",
                log_file.display()
            );
        } else {
            eprintln!("Warning: Failed to create debug log file. Logging to stderr only.");
            tracing_subscriber::fmt().with_env_filter(filter).init();
        }
    } else {
        // Standard stderr logging
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }

    Ok(None)
}

fn open_private_log(path: &std::path::Path) -> std::io::Result<std::fs::File> {
    let mut options = std::fs::OpenOptions::new();
    options.create(true).write(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)
}

#[cfg(test)]
mod tests {
    use super::{LogConfig, QUIET_LOG_DIRECTIVES};
    use crate::commands::Args;
    use clap::Parser;

    #[test]
    fn quiet_flag_uses_pipe_friendly_filter() {
        let args = Args::parse_from(["caliche", "-q", "config", "path"]);
        let config = LogConfig::from_args(&args);
        assert!(config.quiet);
        assert!(!config.debug);
        assert!(!config.use_interactive);
        assert_eq!(config.level.as_deref(), Some("warn"));
        assert!(QUIET_LOG_DIRECTIVES.contains("sqlx=error"));
        assert!(QUIET_LOG_DIRECTIVES.contains("sqlx_core=error"));
        assert!(!QUIET_LOG_DIRECTIVES.contains("debug"));
        let _parsed = tracing_subscriber::EnvFilter::new(QUIET_LOG_DIRECTIVES);
    }

    #[test]
    fn repl_log_file_is_owner_only() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("repl.log");
        let _file = super::open_private_log(&path).expect("log file");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path)
                .expect("metadata")
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600, "repl log mode was {mode:o}");
        }
    }

    #[test]
    fn quiet_filter_does_not_follow_rust_log() {
        let mut env = crate::test_env::EnvGuard::acquire();
        env.set("RUST_LOG", "debug,sqlx=debug");
        let args = Args::parse_from(["caliche", "-q"]);
        let config = LogConfig::from_args(&args);
        assert!(config.quiet);
        assert_ne!(config.level.as_deref(), Some("debug"));
    }
}

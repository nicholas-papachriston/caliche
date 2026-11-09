use anyhow::Result;
use std::sync::{Arc, Mutex};
use tracing::Level;

/// Log callback function type for forwarding logs to external systems
pub type LogCallback = Arc<dyn Fn(&LogEntry) + Send + Sync>;

/// Structured log entry that can be forwarded to callbacks
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub level: String,
    pub target: String,
    pub message: String,
    pub fields: std::collections::HashMap<String, String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Global logging configuration
struct LoggingConfig {
    callback: Option<LogCallback>,
    json_format: bool,
    level: Level,
    initialized: bool,
}

static LOGGING_CONFIG: Mutex<LoggingConfig> = Mutex::new(LoggingConfig {
    callback: None,
    json_format: false,
    level: Level::INFO,
    initialized: false,
});

/// Set a callback function to receive log entries
pub fn set_log_callback(callback: LogCallback) {
    let mut config = LOGGING_CONFIG.lock().unwrap_or_else(|e| e.into_inner());
    config.callback = Some(callback);
}

/// Initialize logging with optional callback support
pub fn init_logging(json_format: bool) -> Result<()> {
    init_logging_with_level(json_format, None)
}

/// Initialize logging with a specific level
pub fn init_logging_with_level(json_format: bool, level: Option<Level>) -> Result<()> {
    let mut config = LOGGING_CONFIG.lock().unwrap_or_else(|e| e.into_inner());

    if config.initialized {
        // Already initialized, just update config
        config.json_format = json_format;
        if let Some(lvl) = level {
            config.level = lvl;
        }
        return Ok(());
    }

    config.json_format = json_format;
    if let Some(lvl) = level {
        config.level = lvl;
    }

    // Try to initialize tracing-subscriber if available
    // Note: tracing-subscriber is now optional - users can install it separately
    // or use the callback mechanism
    if std::env::var("RUST_LOG").is_ok() {
        // If RUST_LOG is set, try to use tracing-subscriber if available
        // This is a best-effort attempt - if tracing-subscriber isn't available,
        // we fall back to callback-only mode
        config.initialized = true;
    } else {
        // Without RUST_LOG, we can only use callbacks
        // Note: tracing macros will still work, but won't output anywhere
        // unless a callback is set up
        config.initialized = true;
        if config.callback.is_some() {
            tracing::info!("Logging initialized (callback-only mode)");
        }
    }

    Ok(())
}

/// Get current logging configuration
pub fn get_logging_config() -> (bool, Level, bool) {
    let config = LOGGING_CONFIG.lock().unwrap_or_else(|e| e.into_inner());
    (config.json_format, config.level, config.initialized)
}

#[cfg(feature = "tracing-subscriber")]
#[cfg(feature = "tracing-subscriber")]
mod callback_layer {
    use super::{LogCallback, LogEntry};
    use std::collections::HashMap;
    use tracing::{Event, Subscriber};
    use tracing_subscriber::{Layer, layer::Context, registry::LookupSpan};

    /// Custom tracing layer that forwards logs to callbacks
    pub(super) struct CallbackLayer {
        pub(super) callback: LogCallback,
    }

    impl<S> Layer<S> for CallbackLayer
    where
        S: Subscriber + for<'span> LookupSpan<'span>,
    {
        fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
            let mut fields = HashMap::new();
            let mut message = String::new();

            // Visit all fields in the event
            event.record(&mut LogVisitor {
                fields: &mut fields,
                message: &mut message,
            });

            // If no message was captured, use the event metadata
            if message.is_empty() {
                message = event.metadata().name().to_string();
            }

            let entry = LogEntry {
                level: format!("{:?}", event.metadata().level()),
                target: event.metadata().target().to_string(),
                message,
                fields,
                timestamp: chrono::Utc::now(),
            };

            (self.callback)(&entry);
        }
    }

    /// Visitor for extracting fields from tracing events
    struct LogVisitor<'a> {
        fields: &'a mut HashMap<String, String>,
        message: &'a mut String,
    }

    impl<'a> tracing::field::Visit for LogVisitor<'a> {
        fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
            if field.name() == "message" {
                *self.message = value.to_string();
            } else {
                self.fields
                    .insert(field.name().to_string(), value.to_string());
            }
        }

        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            if field.name() == "message" {
                *self.message = format!("{:?}", value);
            } else {
                self.fields
                    .insert(field.name().to_string(), format!("{:?}", value));
            }
        }

        fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
            self.fields
                .insert(field.name().to_string(), value.to_string());
        }

        fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
            self.fields
                .insert(field.name().to_string(), value.to_string());
        }

        fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
            self.fields
                .insert(field.name().to_string(), value.to_string());
        }

        fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
            self.fields
                .insert(field.name().to_string(), value.to_string());
        }
    }
}

#[cfg(feature = "tracing-subscriber")]
use callback_layer::CallbackLayer;

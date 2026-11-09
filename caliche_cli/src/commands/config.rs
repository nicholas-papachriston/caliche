use clap::Subcommand;

/// Configuration and credential management actions
#[derive(Clone, Subcommand, Debug)]
pub enum ConfigAction {
    /// Interactive credential setup wizard
    Setup,
    /// Set an API key for a provider
    Set {
        /// Provider name (openai, anthropic, openrouter, etc.)
        provider: String,
        /// API key value (or use --env to read from environment)
        #[arg(long)]
        key: Option<String>,
        /// Read key from environment variable
        #[arg(long)]
        env: Option<String>,
    },
    /// Get the configured API key for a provider (masked)
    Get {
        /// Provider name
        provider: String,
        /// Show full key (not masked)
        #[arg(long)]
        unmask: bool,
        /// Required with --unmask because it prints a secret to stdout
        #[arg(long = "yes-i-understand-this-prints-secrets")]
        yes_i_understand_this_prints_secrets: bool,
    },
    /// Remove an API key for a provider
    Remove {
        /// Provider name
        provider: String,
    },
    /// List all providers and their configuration status
    List {
        /// Show only configured providers
        #[arg(long)]
        configured: bool,
    },
    /// Set the default provider
    Default {
        /// Provider name to set as default (optional - will show interactive selection if omitted)
        provider: Option<String>,
    },
    /// Set the default model for a provider (interactive if model not specified)
    Model {
        /// Provider name
        provider: String,
        /// Model name/ID (optional - will show interactive selection if omitted)
        model: Option<String>,
    },
    /// Configure AWS credentials for Bedrock
    Aws {
        /// AWS Access Key ID
        #[arg(long)]
        access_key: Option<String>,
        /// AWS Secret Access Key
        #[arg(long)]
        secret_key: Option<String>,
        /// AWS Region
        #[arg(long)]
        region: Option<String>,
        /// Read credentials from AWS CLI profile
        #[arg(long)]
        profile: Option<String>,
    },
    /// Show merged configuration from all sources
    Show {
        /// Output format
        #[arg(long, default_value = "toml", value_parser = ["toml", "json"])]
        format: String,
        /// List which sources contributed to the merged config
        #[arg(long)]
        verbose: bool,
    },
    /// Emit the JSON Schema for the caliche.toml config file format
    Schema {
        /// Output path (writes to stdout if omitted)
        #[arg(long)]
        output: Option<std::path::PathBuf>,
    },
    /// Show config file path
    Path,
    /// Export credentials to an environment file
    Export {
        /// Shell type (bash, zsh, fish, powershell)
        #[arg(long, default_value = "bash")]
        shell: String,
        /// Output file path. Defaults to a file in the system temp directory.
        #[arg(long)]
        output: Option<std::path::PathBuf>,
        /// Print raw export statements to stdout instead of writing a file
        #[arg(long)]
        stdout: bool,
        /// Required with --stdout because it prints secrets to stdout
        #[arg(long = "yes-i-understand-this-prints-secrets")]
        yes_i_understand_this_prints_secrets: bool,
    },
    /// Manage the credential vault (directory-scoped credential storage)
    #[command(after_help = "EXAMPLES:\n    \
        caliche config vault status                        # Show credentials available in CWD\n    \
        caliche config vault list                          # List all credential scopes\n    \
        caliche config vault import                        # Import .env from CWD into keychain\n    \
        caliche config vault link ~/other-project          # Use another project's credentials here\n    \
        caliche config vault promote openai                # Promote CWD credential to global")]
    Vault {
        #[command(subcommand)]
        action: VaultAction,
    },
    /// Manage gateway integration credentials (Telegram/Discord/Twilio/Slack/Teams)
    #[command(after_help = "EXAMPLES:\n    \
        caliche config gateway list\n    \
        caliche config gateway set telegram_bot_token --key <token>\n    \
        caliche config gateway set slack_signing_secret --env SLACK_SIGNING_SECRET\n    \
        caliche config gateway set teams_webhook_url --key <url>\n    \
        caliche config gateway set twilio_auth_token --env TWILIO_AUTH_TOKEN\n    \
        caliche config gateway get twilio_account_sid --unmask --yes-i-understand-this-prints-secrets")]
    Gateway {
        #[command(subcommand)]
        action: GatewayConfigAction,
    },
}

/// Gateway credential management actions
#[derive(Clone, Subcommand, Debug)]
pub enum GatewayConfigAction {
    /// Set a gateway credential
    Set {
        /// Credential name (telegram_bot_token, discord_bot_token, twilio_account_sid, twilio_auth_token, twilio_from, slack_bot_token, slack_signing_secret, teams_webhook_url)
        name: String,
        /// Credential value (or use --env to read from environment)
        #[arg(long)]
        key: Option<String>,
        /// Read value from environment variable
        #[arg(long)]
        env: Option<String>,
    },
    /// Get a configured gateway credential (masked by default)
    Get {
        /// Credential name
        name: String,
        /// Show full value (not masked)
        #[arg(long)]
        unmask: bool,
        /// Required with --unmask because it prints a secret to stdout
        #[arg(long = "yes-i-understand-this-prints-secrets")]
        yes_i_understand_this_prints_secrets: bool,
    },
    /// Remove a gateway credential
    Remove {
        /// Credential name
        name: String,
    },
    /// List supported gateway credentials and status
    List {
        /// Show only configured credentials
        #[arg(long)]
        configured: bool,
    },
}

/// Credential vault subcommands
#[derive(Clone, Subcommand, Debug)]
pub enum VaultAction {
    /// Show all credentials accessible from the current directory (all scopes resolved)
    Status,
    /// List all known credential scopes
    List,
    /// Show credentials stored in a scope (masked)
    Show {
        /// Directory path of the scope to show (defaults to CWD)
        #[arg(long)]
        scope: Option<String>,
    },
    /// Link another directory's credentials to the current directory
    Link {
        /// Source directory whose credentials to use
        source_dir: String,
        /// Only link a specific provider (omit to link all)
        #[arg(long)]
        provider: Option<String>,
    },
    /// Remove a scope link from the current directory
    Unlink {
        /// Source directory to unlink
        source_dir: String,
        /// Only unlink a specific provider (omit to unlink all)
        #[arg(long)]
        provider: Option<String>,
    },
    /// Scan a directory for .env files and import credentials into the keychain
    Import {
        /// Directory to scan (defaults to CWD)
        #[arg(long)]
        dir: Option<String>,
        /// Also scan parent directories
        #[arg(long)]
        parents: bool,
    },
    /// Promote a scoped credential to the global scope
    Promote {
        /// Provider to promote (e.g. openai, anthropic)
        provider: String,
    },
}

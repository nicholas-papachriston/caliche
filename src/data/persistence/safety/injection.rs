#![expect(
    clippy::expect_used,
    reason = "LazyLock regex literals are reviewed constants; compile failure is a programmer bug"
)]

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

/// SQL safety checker for detecting injection attempts and dangerous operations.
///
/// Uses `&'static [Regex]` slices referencing module-level `LazyLock` statics so
/// that no regex compilation occurs after program start and `Clone`/`new()` are
/// allocation-free.
#[derive(Debug, Clone, Copy)]
pub struct SqlSafetyChecker;

/// Result of a safety check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyCheckResult {
    pub is_safe: bool,
    pub risks: Vec<SecurityRisk>,
    pub severity: RiskSeverity,
}

/// Security risk classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityRisk {
    pub severity: RiskSeverity,
    pub risk_type: RiskType,
    pub description: String,
    pub location: Option<String>,
}

/// Risk severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RiskSeverity {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

/// Types of security risks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskType {
    SqlInjection,
    UnauthorizedAccess,
    DataExfiltration,
    DestructiveOperation,
    PerformanceImpact,
    PrivacyViolation,
}

// Compile every regex exactly once at first use via LazyLock.
// The patterns are constant string literals; a compilation failure here is a bug,
// not a runtime condition, so `.expect()` with a descriptive message is appropriate.

static DANGEROUS_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        // SQL injection patterns
        Regex::new(r"(?i)\bOR\b\s+\d+\s*=\s*\d+").expect("DANGEROUS_PATTERNS[0]"),
        Regex::new(r#"(?i)\bOR\b\s+['"]?1['"]?\s*=\s*['"]?1['"]?"#).expect("DANGEROUS_PATTERNS[1]"),
        Regex::new(r"(?i);\s*DROP\s+").expect("DANGEROUS_PATTERNS[2]"),
        Regex::new(r"(?i);\s*DELETE\s+FROM\s+").expect("DANGEROUS_PATTERNS[3]"),
        Regex::new(r"(?i)UNION\s+SELECT").expect("DANGEROUS_PATTERNS[4]"),
        Regex::new(r"(?i)exec\s*\(|execute\s*\(").expect("DANGEROUS_PATTERNS[5]"),
        Regex::new(r"(?i)xp_cmdshell").expect("DANGEROUS_PATTERNS[6]"),
        // Comment-based injection (quote-terminated payloads, not benign comments)
        Regex::new(r#"(?i)['"]\s*--"#).expect("DANGEROUS_PATTERNS[7]"),
        Regex::new(r#"(?i)['"]\s*#"#).expect("DANGEROUS_PATTERNS[8]"),
        Regex::new(r#"(?i)['"]\s*/\*"#).expect("DANGEROUS_PATTERNS[9]"),
        // Stacked queries with dangerous follow-on statements
        Regex::new(r"(?i);\s*(?:DROP|DELETE|INSERT|UPDATE|CREATE|ALTER|TRUNCATE|EXEC|EXECUTE)\b")
            .expect("DANGEROUS_PATTERNS[10]"),
    ]
});

static DESTRUCTIVE_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(r"(?i)^\s*DROP\s+").expect("DESTRUCTIVE_PATTERNS[0]"),
        Regex::new(r"(?i)^\s*TRUNCATE\s+").expect("DESTRUCTIVE_PATTERNS[1]"),
        // DELETE without WHERE
        Regex::new(r"(?i)^\s*DELETE\s+FROM\s+\w+\s*;?\s*$").expect("DESTRUCTIVE_PATTERNS[2]"),
        // UPDATE without WHERE
        Regex::new(r"(?i)^\s*UPDATE\s+\w+\s+SET\s+.*\s*;?\s*$").expect("DESTRUCTIVE_PATTERNS[3]"),
    ]
});

static SYSTEM_TABLE_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(r"(?i)\bpg_catalog\b").expect("SYSTEM_TABLE_PATTERNS[0]"),
        Regex::new(r"(?i)\binformation_schema\b").expect("SYSTEM_TABLE_PATTERNS[1]"),
        Regex::new(r"(?i)\bpg_shadow\b").expect("SYSTEM_TABLE_PATTERNS[2]"),
        Regex::new(r"(?i)\bpg_authid\b").expect("SYSTEM_TABLE_PATTERNS[3]"),
        Regex::new(r"(?i)\bmysql\.user\b").expect("SYSTEM_TABLE_PATTERNS[4]"),
        Regex::new(r"(?i)\bsys\.").expect("SYSTEM_TABLE_PATTERNS[5]"),
    ]
});

// Sanitize-specific patterns: also compiled once.
static SANITIZE_COMMENT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"--[^\n]*").expect("SANITIZE_COMMENT_RE"));
static SANITIZE_BLOCK_COMMENT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"/\*.*?\*/").expect("SANITIZE_BLOCK_COMMENT_RE"));
static SANITIZE_WHITESPACE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s+").expect("SANITIZE_WHITESPACE_RE"));

impl SqlSafetyChecker {
    /// Create a new SQL safety checker.
    ///
    /// Construction is free: the checker holds no data — all regex state lives
    /// in module-level `LazyLock` statics and is compiled at most once per process.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Check if a SQL query is safe to execute.
    pub fn check_safety(&self, sql: &str) -> SafetyCheckResult {
        let mut risks = Vec::new();

        // Check for SQL injection patterns
        for pattern in DANGEROUS_PATTERNS.iter() {
            if let Some(matched) = pattern.find(sql) {
                risks.push(SecurityRisk {
                    severity: RiskSeverity::Critical,
                    risk_type: RiskType::SqlInjection,
                    description: format!(
                        "Potential SQL injection pattern detected: {}",
                        matched.as_str()
                    ),
                    location: Some(matched.as_str().to_owned()),
                });
            }
        }

        // Check for destructive operations
        for pattern in DESTRUCTIVE_PATTERNS.iter() {
            if pattern.is_match(sql) {
                risks.push(SecurityRisk {
                    severity: RiskSeverity::High,
                    risk_type: RiskType::DestructiveOperation,
                    description: "Potentially destructive operation detected (DROP, TRUNCATE, or DELETE/UPDATE without WHERE)".to_string(),
                    location: None,
                });
            }
        }

        // Check for system table access
        for pattern in SYSTEM_TABLE_PATTERNS.iter() {
            if pattern.is_match(sql) {
                risks.push(SecurityRisk {
                    severity: RiskSeverity::Medium,
                    risk_type: RiskType::UnauthorizedAccess,
                    description: "Access to system tables detected".to_string(),
                    location: None,
                });
            }
        }

        // Determine overall severity
        let severity = risks
            .iter()
            .map(|r| r.severity)
            .max()
            .unwrap_or(RiskSeverity::Info);

        let is_safe = !matches!(severity, RiskSeverity::High | RiskSeverity::Critical);

        SafetyCheckResult {
            is_safe,
            risks,
            severity,
        }
    }

    /// Sanitize a SQL query by removing comments and normalizing whitespace.
    ///
    /// Reuses the module-level `LazyLock` regex statics — no allocations beyond
    /// the returned `String`.
    pub fn sanitize(&self, sql: &str) -> String {
        let without_comments = SANITIZE_COMMENT_RE.replace_all(sql, "");
        let without_block_comments = SANITIZE_BLOCK_COMMENT_RE.replace_all(&without_comments, "");
        SANITIZE_WHITESPACE_RE
            .replace_all(without_block_comments.trim(), " ")
            .into_owned()
    }
}

impl Default for SqlSafetyChecker {
    fn default() -> Self {
        Self::new()
    }
}

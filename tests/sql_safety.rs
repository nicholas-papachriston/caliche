//! SQL injection safety checker integration tests.

use caliche::persistence::safety::{RiskType, SqlSafetyChecker};

#[test]
fn test_sql_injection_detection() {
    let checker = SqlSafetyChecker::new();
    let malicious_sql = "SELECT * FROM users WHERE id = 1 OR 1=1";
    let result = checker.check_safety(malicious_sql);
    assert!(!result.is_safe);
    assert!(
        result
            .risks
            .iter()
            .any(|r| r.risk_type == RiskType::SqlInjection)
    );
}

#[test]
fn test_destructive_operation_detection() {
    let checker = SqlSafetyChecker::new();
    let destructive_sql = "DROP TABLE users";
    let result = checker.check_safety(destructive_sql);
    assert!(!result.is_safe);
    assert!(
        result
            .risks
            .iter()
            .any(|r| r.risk_type == RiskType::DestructiveOperation)
    );
}

#[test]
fn test_safe_query() {
    let checker = SqlSafetyChecker::new();
    let safe_sql = "SELECT id, name FROM users WHERE status = 'active' LIMIT 10";
    let result = checker.check_safety(safe_sql);
    assert!(result.is_safe);
}

#[test]
fn benign_sql_comments_are_allowed() {
    let checker = SqlSafetyChecker::new();
    let safe_sql = "SELECT id FROM users -- active accounts only";
    let result = checker.check_safety(safe_sql);
    assert!(result.is_safe);
}

#[test]
fn quote_terminated_comment_injection_is_blocked() {
    let checker = SqlSafetyChecker::new();
    let malicious_sql = "SELECT * FROM users WHERE name = 'admin'--";
    let result = checker.check_safety(malicious_sql);
    assert!(!result.is_safe);
}

#[test]
fn test_sanitize_removes_comments() {
    let checker = SqlSafetyChecker::new();
    let sql_with_comment = "SELECT * FROM users -- malicious comment";
    let sanitized = checker.sanitize(sql_with_comment);
    assert!(!sanitized.contains("--"));
}

/// Benign SQL that includes comments must not trigger false-positive injection flags.
#[test]
fn benign_sql_comment_corpus() {
    let checker = SqlSafetyChecker::new();
    let corpus = [
        "SELECT id FROM users -- active accounts only",
        "SELECT COUNT(*) FROM orders /* revenue report */ WHERE status = 'paid'",
        "SELECT id, name FROM products\n-- filter discontinued\nWHERE active = true",
        "SELECT u.id FROM users u /* join hint */ INNER JOIN roles r ON r.user_id = u.id",
        "SELECT 1 /* sanity check */",
        "WITH recent AS (SELECT id FROM events -- last 24h\nWHERE created_at > NOW() - INTERVAL '1 day')\nSELECT * FROM recent",
    ];

    for sql in corpus {
        let result = checker.check_safety(sql);
        assert!(
            result.is_safe,
            "benign SQL with comments should be safe; query={sql:?}, risks={:?}",
            result.risks
        );
    }
}

#[test]
fn all_pattern_categories_are_exercised() {
    let checker = SqlSafetyChecker::new();

    let _ = checker.check_safety("SELECT 1 OR 1=1");
    let _ = checker.check_safety("DROP TABLE users");
    let _ = checker.check_safety("SELECT * FROM information_schema.tables");
    let _ = checker.sanitize("SELECT 1 -- comment\n/* block */");

    let _ = checker.check_safety("SELECT * FROM users WHERE id = 1");
}

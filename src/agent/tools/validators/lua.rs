//! Lightweight Lua validation: balance of block keywords and common typos.

/// Validate Lua source. Returns Ok(()) or list of issue descriptions.
pub fn validate(content: &str) -> super::ValidationResult {
    let mut issues = Vec::new();

    // Common typos
    if content.contains("fucntion") {
        issues.push("Possible typo: 'fucntion' -> 'function'".to_string());
    }
    if content.contains("retrurn") {
        issues.push("Possible typo: 'retrurn' -> 'return'".to_string());
    }
    if content.contains("functon") {
        issues.push("Possible typo: 'functon' -> 'function'".to_string());
    }
    if content.contains("retrun") {
        issues.push("Possible typo: 'retrun' -> 'return'".to_string());
    }
    // "else if" in Lua should be "elseif" (one word)
    if content.contains("else if") && !content.contains("elseif") {
        issues.push("Lua uses 'elseif' (one word), not 'else if'".to_string());
    }

    // Simple balance check: function/repeat/do/then/elseif vs end
    let block_starts = count_matches(content, &["function", "repeat", " do ", " then ", "elseif"]);
    let ends_approx = content.matches("end").count();
    if block_starts > 0 && ends_approx > 0 && ends_approx < block_starts {
        issues.push(format!(
            "Possible missing 'end': ~{block_starts} block openers (function/repeat/do/then/elseif) vs ~{ends_approx} 'end'"
        ));
    }

    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

fn count_matches(s: &str, patterns: &[&str]) -> usize {
    let mut n = 0;
    for p in patterns {
        n += s.matches(p).count();
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lua_typo_fucntion() {
        let r = validate("local x = fucntion() end");
        assert!(r.is_err());
        let e = r.unwrap_err();
        assert!(e.iter().any(|m| m.contains("fucntion")));
    }

    #[test]
    fn test_lua_typo_retrurn() {
        let r = validate("retrurn 1");
        assert!(r.is_err());
    }

    #[test]
    fn test_lua_valid() {
        assert!(validate("function f() return 1 end").is_ok());
    }
}

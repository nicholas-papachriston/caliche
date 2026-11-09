//! Protocol version date parsing.
//!
//! The server speaks one revision. Admission lives in [`crate::revision`].
//! This module only parses `YYYY-MM-DD` strings.

/// Parse an MCP protocol version string (`YYYY-MM-DD`) into a comparable
/// integer date. Returns `None` when the string is malformed.
pub fn parse_protocol_date(version: &str) -> Option<u32> {
    let mut parts = version.split('-');
    let year: u32 = parts.next()?.parse().ok()?;
    let month: u32 = parts.next()?.parse().ok()?;
    let day: u32 = parts.next()?.parse().ok()?;
    if parts.next().is_some() || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    Some(year * 10_000 + month * 100 + day)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_protocol_date() {
        assert_eq!(parse_protocol_date("2026-07-28"), Some(20_260_728));
        assert_eq!(parse_protocol_date("not-a-date"), None);
        assert_eq!(parse_protocol_date("2026-13-01"), None);
    }
}

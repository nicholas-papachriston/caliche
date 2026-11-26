//! Each case of the walk matrix and the documented rule it comes from.
//! The CLI reader's own cases in `caliche_cli` are not listed here.
//!
//! The rules paraphrase gitignore(5) and glob(7). Where the documents say
//! nothing or disagree, as for a byte order mark, a NUL or a backslash in
//! brackets, the rule says so, and the git run in the matrix decides.

use std::collections::BTreeSet;

use super::parity_tests::{ENCODINGS, LAYERED, SYNTAX};

const BLANK: &str = "gitignore(5) PATTERN FORMAT: a blank line matches no file.";
const BOM: &str = "gitignore(5) does not cover a byte order mark; the git oracle skips a UTF-8 one at the start of the file.";
const BRACE: &str = "gitignore(5) PATTERN FORMAT: names *, ?, ranges, ** and backslash as special and nothing else, so braces and commas are literal.";
const BYTES: &str = "gitignore(5) does not cover encodings; the git oracle matches bytes, so a line need not be UTF-8.";
const COMMENT: &str = "gitignore(5) PATTERN FORMAT: a line starting with # is a comment; a backslash before a leading # makes it a pattern.";
const CRLF: &str = "gitignore(5) does not cover line ends; the git oracle drops one carriage return before each newline.";
const DIR_FILE: &str = "gitignore(5) PATTERN FORMAT: a pattern without a trailing slash matches a file or a directory.";
const DSTAR_LEAD: &str = "gitignore(5) PATTERN FORMAT: a leading **/ matches in all directories.";
const DSTAR_MID: &str = "gitignore(5) PATTERN FORMAT: /**/ matches zero or more directories.";
const DSTAR_OTHER: &str =
    "gitignore(5) PATTERN FORMAT: other consecutive asterisks are regular asterisks.";
const DSTAR_RUN: &str = "gitignore(5) PATTERN FORMAT says other consecutive asterisks are regular asterisks; for a run of three or more that is a whole part of the path, the git run decides, and it reads the run as **.";
const DSTAR_TRAIL: &str = "gitignore(5) PATTERN FORMAT: a trailing /** matches everything inside.";
const ESC: &str = "gitignore(5) PATTERN FORMAT: a backslash escapes the character after it.";
const ESC_END: &str = "gitignore(5) does not cover a backslash at the end of a pattern, with nothing to escape; the git run decides, and the pattern then matches nothing.";
const NEG: &str = "gitignore(5) PATTERN FORMAT: a leading ! negates the pattern and re-includes a file an earlier pattern excluded; a backslash before a leading ! makes it literal.";
const NEG_PARENT: &str = "gitignore(5) PATTERN FORMAT: a file cannot be re-included when a parent directory of it is excluded.";
const NUL: &str = "gitignore(5) does not cover NUL; the git oracle ends the pattern at it.";
const POSIX: &str = "glob(7) Character classes: [:name:] inside brackets names a class, as git's fnmatch(3) reading of gitignore(5) brackets uses it.";
const RANGE: &str = "gitignore(5) PATTERN FORMAT: a range such as [a-zA-Z] matches one character in it; brackets follow fnmatch(3) with FNM_PATHNAME as glob(7) describes: a leading ! negates, a leading ] and an end - are members, a range runs in collating order, and no bracket matches a slash.";
const CARET: &str = "glob(7) leaves a leading ^ in brackets undefined; the git run decides, and it negates the bracket.";
const CLASS_ESC: &str = "gitignore(5) PATTERN FORMAT: a backslash escapes the character after it, while glob(7) says a character in brackets stands for itself; the git run decides, and it reads the backslash as an escape.";
const UNCLOSED: &str = "gitignore(5) and glob(7) do not say what a [ with no closing ] does in a pattern; the git run decides, and the pattern then matches nothing.";
const WS_OTHER: &str = "gitignore(5) PATTERN FORMAT speaks only of trailing spaces; for other whitespace, and for an escape before it, the git run decides, and it keeps them in the pattern.";
const POSIX_EDGE: &str = "glob(7) does not cover an unknown class name, an empty [::] or a class as the end of a range; the git run decides.";
const DSTAR_ONLY: &str = "gitignore(5) PATTERN FORMAT does not cover a pattern of only **; the git run decides, and it matches every path.";
const POSIX_NAME: &str = "glob(7) does not say where a class name ends; the git run decides: the name runs to the first ], and with no : right before that ] the [ is a plain member.";
const POSIX_SPACE: &str = "glob(7) defers [:space:] to isspace(3); git's own reading, as the git run shows, is space, tab, newline and carriage return only.";
const DSTAR_ESC: &str = "gitignore(5) PATTERN FORMAT does not cover ** before an escaped slash; the git run decides: it crosses directories and needs at least one.";
const SPACE_MID: &str = "gitignore(5) PATTERN FORMAT does not cover a space inside a pattern; the git run decides, and it matches a space.";
const SLASH: &str = "gitignore(5) PATTERN FORMAT: a slash at the start or in the middle makes the pattern relative to the directory of its ignore file; otherwise it matches at any level.";
const SOURCES: &str = "gitignore(5) DESCRIPTION: sources by precedence, highest first: .gitignore files, where a deeper file overrides a higher one, then $GIT_DIR/info/exclude, then core.excludesFile; within one source the last matching pattern decides.";
const STAR: &str = "gitignore(5) PATTERN FORMAT: * matches anything but a slash, and ? one character that is not a slash.";
const TRAIL: &str =
    "gitignore(5) PATTERN FORMAT: trailing spaces are ignored unless quoted with a backslash.";
const TRAIL_SLASH: &str = "gitignore(5) PATTERN FORMAT: a trailing slash matches only a directory.";

/// Every case as (table, case, rule).
const RULES: [(&str, &str, &str); 159] = [
    ("encodings", "bom", BOM),
    ("encodings", "bom-brace", BOM),
    ("encodings", "bad-after", BYTES),
    ("encodings", "bad-before", BYTES),
    ("encodings", "crlf", CRLF),
    ("encodings", "bom-crlf", BOM),
    ("encodings", "space-crlf", CRLF),
    ("encodings", "bad-crlf", CRLF),
    ("encodings", "nul", NUL),
    ("encodings", "tab", WS_OTHER),
    ("encodings", "bad-negation", BYTES),
    ("encodings", "tab-only", WS_OTHER),
    ("encodings", "esc-space-space", TRAIL),
    ("encodings", "nbsp", WS_OTHER),
    ("encodings", "negated-tab", WS_OTHER),
    ("encodings", "esc-tab", WS_OTHER),
    ("encodings", "tab-space", WS_OTHER),
    ("encodings", "esc-space-tab", WS_OTHER),
    ("encodings", "dbl-bs-space", TRAIL),
    ("encodings", "ideo-space", WS_OTHER),
    ("encodings", "formfeed", WS_OTHER),
    ("encodings", "cr-cr", CRLF),
    ("encodings", "space-only-esc", TRAIL),
    ("encodings", "dbl-bs-tab", WS_OTHER),
    ("encodings", "trail-spaces", TRAIL),
    ("encodings", "space-esc-space", TRAIL),
    ("encodings", "braces", BRACE),
    ("encodings", "braces-comma", BRACE),
    ("encodings", "brace-tab", BRACE),
    ("encodings", "esc-in-class", CLASS_ESC),
    ("encodings", "posix", POSIX),
    ("encodings", "posix-neg", POSIX),
    ("encodings", "posix-space", POSIX),
    ("encodings", "posix-bad", BYTES),
    ("encodings", "mixed", BYTES),
    ("syntax", "star-ext", STAR),
    ("syntax", "star-mid", STAR),
    ("syntax", "star-under-dir", STAR),
    ("syntax", "star-basename", STAR),
    ("syntax", "star-dir-star", STAR),
    ("syntax", "qmark", STAR),
    ("syntax", "qmark-slash", STAR),
    ("syntax", "dstar-lead", DSTAR_LEAD),
    ("syntax", "dstar-lead-dir", DSTAR_LEAD),
    ("syntax", "dstar-trail", DSTAR_TRAIL),
    ("syntax", "dstar-mid", DSTAR_MID),
    ("syntax", "dstar-inword", DSTAR_OTHER),
    ("syntax", "dstar-after-slash", DSTAR_OTHER),
    ("syntax", "dstar-ext", DSTAR_LEAD),
    ("syntax", "lead-slash", SLASH),
    ("syntax", "lead-slash-star", SLASH),
    ("syntax", "mid-slash", SLASH),
    ("syntax", "trail-slash", TRAIL_SLASH),
    ("syntax", "trail-slash-star", TRAIL_SLASH),
    ("syntax", "range", RANGE),
    ("syntax", "range-bang", RANGE),
    ("syntax", "range-caret", CARET),
    ("syntax", "range-reverse", RANGE),
    ("syntax", "class-close-first", RANGE),
    ("syntax", "class-neg-close", RANGE),
    ("syntax", "class-dash-end", RANGE),
    ("syntax", "class-dash-start", RANGE),
    ("syntax", "class-esc-close", CLASS_ESC),
    ("syntax", "class-esc-dash", CLASS_ESC),
    ("syntax", "class-esc-range", CLASS_ESC),
    ("syntax", "class-slash", RANGE),
    ("syntax", "class-unclosed", UNCLOSED),
    ("syntax", "posix-alnum", POSIX),
    ("syntax", "posix-alpha", POSIX),
    ("syntax", "posix-blank", POSIX),
    ("syntax", "posix-cntrl", POSIX),
    ("syntax", "posix-digit", POSIX),
    ("syntax", "posix-graph", POSIX),
    ("syntax", "posix-lower", POSIX),
    ("syntax", "posix-print", POSIX),
    ("syntax", "posix-punct", POSIX),
    ("syntax", "posix-space-xy", POSIX),
    ("syntax", "posix-upper", POSIX),
    ("syntax", "posix-xdigit", POSIX),
    ("syntax", "posix-two", POSIX),
    ("syntax", "posix-neg-alpha", POSIX),
    ("syntax", "posix-unknown", POSIX_EDGE),
    ("syntax", "esc-at-end", ESC_END),
    ("syntax", "comment-line", COMMENT),
    ("syntax", "posix-first-close", POSIX_NAME),
    ("syntax", "posix-name-unclosed", POSIX_NAME),
    ("syntax", "posix-empty-name", POSIX_NAME),
    ("syntax", "posix-space-git", POSIX_SPACE),
    ("syntax", "qmark-no-slash", STAR),
    ("syntax", "glued-dstar", DSTAR_OTHER),
    ("syntax", "dstar-esc-slash", DSTAR_ESC),
    ("syntax", "esc-star", ESC),
    ("syntax", "esc-qmark", ESC),
    ("syntax", "esc-bracket", ESC),
    ("syntax", "esc-hash", COMMENT),
    ("syntax", "esc-bang", NEG),
    ("syntax", "esc-backslash", ESC),
    ("syntax", "bang-mid", NEG),
    ("syntax", "braces-empty", BRACE),
    ("syntax", "space-mid", SPACE_MID),
    ("syntax", "negation", NEG),
    ("syntax", "negation-in-dir", NEG),
    ("syntax", "negation-under-dir", NEG_PARENT),
    ("syntax", "negation-star", NEG),
    ("syntax", "negation-esc-class", NEG),
    ("syntax", "negation-posix-space", NEG),
    ("syntax", "negation-posix-dash", NEG),
    ("syntax", "negation-brace", NEG),
    ("syntax", "negation-escape", ESC),
    ("syntax", "negation-unclosed", UNCLOSED),
    ("syntax", "dstar-zero", DSTAR_MID),
    ("syntax", "dstar-lead-path", DSTAR_LEAD),
    ("syntax", "dstar-only", DSTAR_ONLY),
    ("syntax", "dstar-triple", DSTAR_RUN),
    ("syntax", "dstar-reinclude", DSTAR_TRAIL),
    ("syntax", "star-slash", STAR),
    ("syntax", "negation-dirs", NEG),
    ("syntax", "dir-and-file", DIR_FILE),
    ("syntax", "blank-lines", BLANK),
    ("syntax", "hash-after-space", COMMENT),
    ("syntax", "esc-space-lead", ESC),
    ("syntax", "posix-alpha-upper", POSIX),
    ("syntax", "posix-alnum-upper", POSIX),
    ("syntax", "posix-xdigit-upper", POSIX),
    ("syntax", "posix-upper-all", POSIX),
    ("syntax", "posix-lower-all", POSIX),
    ("syntax", "posix-empty", POSIX_EDGE),
    ("syntax", "range-to-posix", POSIX_EDGE),
    ("syntax", "range-nonascii", BYTES),
    ("syntax", "posix-dash-end", POSIX),
    ("syntax", "class-esc-dash-mid", CLASS_ESC),
    ("syntax", "negation-posix-space-2", NEG),
    ("syntax", "negation-esc-dash", NEG),
    ("syntax", "negation-posix-unknown", NEG),
    ("syntax", "negation-posix-dash-end", NEG),
    ("syntax", "negation-brace-alt", NEG),
    ("syntax", "negation-brace-comma", NEG),
    ("syntax", "negation-esc-close", NEG),
    ("syntax", "negation-trail-space", NEG),
    ("syntax", "negation-after-tab", NEG),
    ("syntax", "negation-crlf", NEG),
    ("syntax", "negation-esc-space-crlf", TRAIL),
    ("syntax", "negation-nul", NEG),
    ("syntax", "negation-brace-tab", NEG),
    ("syntax", "negation-posix-alpha", NEG),
    ("layered", "nested-anchored", SLASH),
    ("layered", "nested-middle-slash", SLASH),
    ("layered", "nested-basename", SLASH),
    ("layered", "nested-dstar-lead", DSTAR_LEAD),
    ("layered", "nested-dstar-trail", DSTAR_TRAIL),
    ("layered", "nested-dir-only", TRAIL_SLASH),
    ("layered", "nested-negation", SOURCES),
    ("layered", "nested-hides", SOURCES),
    ("layered", "nested-under-excluded", NEG_PARENT),
    ("layered", "gitignore-over-global", SOURCES),
    ("layered", "gitignore-over-exclude", SOURCES),
    ("layered", "exclude-over-global", SOURCES),
    ("layered", "global-negation-loses", SOURCES),
    ("layered", "exclude-negation-loses", SOURCES),
];

/// Every case in the matrix, as (table, case).
fn cases() -> BTreeSet<(&'static str, &'static str)> {
    let mut all = BTreeSet::new();
    all.extend(ENCODINGS.iter().map(|case| ("encodings", case.0)));
    all.extend(SYNTAX.iter().map(|case| ("syntax", case.0)));
    all.extend(LAYERED.iter().map(|case| ("layered", case.0)));
    all
}

/// Every case in the walk matrix names exactly one documented rule, and
/// every rule entry names a case that exists.
#[test]
fn every_matrix_case_names_its_documented_rule() {
    let named: Vec<(&str, &str)> = RULES
        .iter()
        .map(|&(table, case, _)| (table, case))
        .collect();
    let unique: BTreeSet<(&str, &str)> = named.iter().copied().collect();
    assert_eq!(unique.len(), named.len(), "a case is named twice");
    assert_eq!(unique, cases());
}

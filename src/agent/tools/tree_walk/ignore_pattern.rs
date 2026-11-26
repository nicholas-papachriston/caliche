//! One ignore pattern, read and matched as gitignore(5) describes it.
//!
//! A pattern is matched against the bytes of a `/`-separated path, never
//! as text, so a name or a pattern that is not UTF-8 matches byte for
//! byte. Where gitignore(5) and glob(7) say nothing, the behavior follows
//! what the git binary does, as the parity tests check.

/// A POSIX character class inside brackets, ASCII only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Posix {
    Alnum,
    Alpha,
    Blank,
    Cntrl,
    Digit,
    Graph,
    Lower,
    Print,
    Punct,
    Space,
    Upper,
    Xdigit,
}

impl Posix {
    fn named(name: &[u8]) -> Option<Self> {
        Some(match name {
            b"alnum" => Self::Alnum,
            b"alpha" => Self::Alpha,
            b"blank" => Self::Blank,
            b"cntrl" => Self::Cntrl,
            b"digit" => Self::Digit,
            b"graph" => Self::Graph,
            b"lower" => Self::Lower,
            b"print" => Self::Print,
            b"punct" => Self::Punct,
            b"space" => Self::Space,
            b"upper" => Self::Upper,
            b"xdigit" => Self::Xdigit,
            _ => return None,
        })
    }

    fn matches(self, byte: u8) -> bool {
        match self {
            Self::Alnum => byte.is_ascii_alphanumeric(),
            Self::Alpha => byte.is_ascii_alphabetic(),
            Self::Blank => byte == b' ' || byte == b'\t',
            Self::Cntrl => byte.is_ascii_control(),
            Self::Digit => byte.is_ascii_digit(),
            Self::Graph => byte.is_ascii_graphic(),
            Self::Lower => byte.is_ascii_lowercase(),
            Self::Print => byte.is_ascii_graphic() || byte == b' ',
            Self::Punct => byte.is_ascii_punctuation(),
            Self::Space => matches!(byte, b' ' | b'\t' | b'\n' | b'\r'),
            Self::Upper => byte.is_ascii_uppercase(),
            Self::Xdigit => byte.is_ascii_hexdigit(),
        }
    }
}

/// One member of a bracket expression.
#[derive(Debug, Clone, Copy)]
enum Member {
    Byte(u8),
    Range(u8, u8),
    Posix(Posix),
}

impl Member {
    fn matches(self, byte: u8) -> bool {
        match self {
            Self::Byte(member) => member == byte,
            Self::Range(low, high) => (low..=high).contains(&byte),
            Self::Posix(class) => class.matches(byte),
        }
    }
}

/// One element of a compiled pattern.
#[derive(Debug, Clone)]
enum Token {
    /// One byte.
    Byte(u8),
    /// `?`: one byte other than `/`.
    One,
    /// `*`: any run of bytes without a `/`.
    Star,
    /// `**/` at the start or after a `/`: zero or more whole directories.
    Dirs,
    /// `**` at the end, or before an escaped `/`, at the start or after a
    /// `/`: any run of bytes, `/` among them.
    Any,
    /// A bracket expression: one byte other than `/`.
    Class { negated: bool, members: Vec<Member> },
}

/// A compiled pattern, without its `!`, trailing `/` or leading `/`.
#[derive(Debug, Clone)]
pub(super) struct Glob {
    tokens: Vec<Token>,
}

impl Glob {
    /// Compile `pattern`, in time linear in its length. `None` when it can
    /// match nothing: a bracket with no closing `]`, an unknown POSIX
    /// class name, or a backslash at the end.
    pub(super) fn compile(pattern: &[u8]) -> Option<Self> {
        let closes = next_closes(pattern);
        let mut tokens = Vec::new();
        let mut at = 0;
        while let Some(&byte) = pattern.get(at) {
            match byte {
                b'\\' => {
                    tokens.push(Token::Byte(*pattern.get(at + 1)?));
                    at += 2;
                }
                b'?' => {
                    tokens.push(Token::One);
                    at += 1;
                }
                b'*' => at = star(pattern, at, &mut tokens),
                b'[' => {
                    let (class, next) = class(pattern, &closes, at + 1)?;
                    tokens.push(class);
                    at = next;
                }
                _ => {
                    tokens.push(Token::Byte(byte));
                    at += 1;
                }
            }
        }
        Some(Self { tokens })
    }

    /// Whether the glob matches all of `text`. It walks the tokens once,
    /// keeping the set of text positions reached so far, so time is the
    /// product of the two lengths and the stack stays flat.
    pub(super) fn matches(&self, text: &[u8]) -> bool {
        let mut reached = vec![false; text.len() + 1];
        reached[0] = true;
        let mut next = vec![false; text.len() + 1];
        for token in &self.tokens {
            step(token, text, &reached, &mut next);
            std::mem::swap(&mut reached, &mut next);
            if !reached.contains(&true) {
                return false;
            }
        }
        reached[text.len()]
    }
}

/// For each index of `pattern`, the index of the first `]` at or after it.
fn next_closes(pattern: &[u8]) -> Vec<Option<usize>> {
    let mut closes = vec![None; pattern.len() + 1];
    for at in (0..pattern.len()).rev() {
        closes[at] = if pattern[at] == b']' {
            Some(at)
        } else {
            closes[at + 1]
        };
    }
    closes
}

/// Compile the run of `*` at `at`; returns the index after it. A run of
/// two or more that starts a part of the path crosses directories, as git
/// reads it: before a `/` as whole directories, at the end or before an
/// escaped `/` as any bytes. Any other run is one `*`.
fn star(pattern: &[u8], at: usize, tokens: &mut Vec<Token>) -> usize {
    let run = pattern[at..]
        .iter()
        .take_while(|&&byte| byte == b'*')
        .count();
    let end = at + run;
    let starts_part = at == 0 || pattern[at - 1] == b'/';
    let after = (pattern.get(end), pattern.get(end + 1));
    match (run >= 2 && starts_part, after) {
        (true, (Some(b'/'), _)) => {
            tokens.push(Token::Dirs);
            end + 1
        }
        (true, (None, _) | (Some(b'\\'), Some(b'/'))) => {
            tokens.push(Token::Any);
            end
        }
        _ => {
            tokens.push(Token::Star);
            end
        }
    }
}

/// Compile the bracket expression whose body starts at `at`, after its
/// `[`; returns it and the index after its `]`. `None` when it can match
/// nothing.
fn class(pattern: &[u8], closes: &[Option<usize>], mut at: usize) -> Option<(Token, usize)> {
    let negated = matches!(pattern.get(at), Some(b'!' | b'^'));
    if negated {
        at += 1;
    }
    let mut members = Vec::new();
    let mut first = true;
    loop {
        let byte = *pattern.get(at)?;
        if byte == b']' && !first {
            return Some((Token::Class { negated, members }, at + 1));
        }
        first = false;
        if byte == b'['
            && pattern.get(at + 1) == Some(&b':')
            && let Some((member, next)) = posix(pattern, closes, at + 2)?
        {
            members.push(member);
            at = next;
            continue;
        }
        let (low, next) = member_byte(pattern, at)?;
        at = next;
        if pattern.get(at) == Some(&b'-') && !matches!(pattern.get(at + 1), None | Some(b']')) {
            let (high, next) = member_byte(pattern, at + 1)?;
            members.push(Member::Range(low, high));
            at = next;
        } else {
            members.push(Member::Byte(low));
        }
    }
}

/// The POSIX class whose name starts at `at`, after `[:`, and the index
/// after its closing `]`. As git reads it, the name runs to the first `]`:
/// `Ok(None)` when no `:` comes right before that `]`, so the `[` is a
/// plain member; `None` when there is no `]` at all or the name is not a
/// class, so the pattern can match nothing.
fn posix(pattern: &[u8], closes: &[Option<usize>], at: usize) -> Option<Option<(Member, usize)>> {
    let close = (*closes.get(at)?)?;
    if close == at || pattern[close - 1] != b':' {
        return Some(None);
    }
    let class = Posix::named(&pattern[at..close - 1])?;
    Some(Some((Member::Posix(class), close + 1)))
}

/// The bracket member byte at `at`, unescaped, and the index after it.
fn member_byte(pattern: &[u8], at: usize) -> Option<(u8, usize)> {
    match *pattern.get(at)? {
        b'\\' => Some((*pattern.get(at + 1)?, at + 2)),
        byte => Some((byte, at + 1)),
    }
}

/// Advance the positions `reached` in `text` over `token` into `next`.
fn step(token: &Token, text: &[u8], reached: &[bool], next: &mut [bool]) {
    next.fill(false);
    match token {
        Token::Star | Token::Any => {
            let crosses = matches!(token, Token::Any);
            let mut open = false;
            for at in 0..=text.len() {
                open = open || reached[at];
                next[at] = open;
                if !crosses && text.get(at) == Some(&b'/') {
                    open = false;
                }
            }
        }
        Token::Dirs => {
            let mut seen = false;
            for at in 0..=text.len() {
                let after_slash = at > 0 && text[at - 1] == b'/';
                seen = seen || (at > 0 && reached[at - 1]);
                next[at] = reached[at] || (seen && after_slash);
            }
        }
        Token::One | Token::Byte { .. } | Token::Class { .. } => {
            for (at, &byte) in text.iter().enumerate() {
                next[at + 1] = reached[at] && one(token, byte);
            }
        }
    }
}

/// Whether the single-byte token `token` matches `byte`.
fn one(token: &Token, byte: u8) -> bool {
    match token {
        Token::One => byte != b'/',
        Token::Byte(wanted) => *wanted == byte,
        Token::Class { negated, members } => {
            byte != b'/' && members.iter().any(|member| member.matches(byte)) != *negated
        }
        Token::Star | Token::Dirs | Token::Any => false,
    }
}

#[cfg(test)]
mod tests {
    use super::Glob;

    fn hits(pattern: &str, text: &str) -> bool {
        Glob::compile(pattern.as_bytes()).is_some_and(|glob| glob.matches(text.as_bytes()))
    }

    #[test]
    fn double_stars_match_whole_directories_only() {
        assert!(hits("a/**/b", "a/b"));
        assert!(hits("a/**/b", "a/x/y/b"));
        assert!(!hits("a/**/b", "a/xb"));
        assert!(hits("**/b", "x/y/b"));
        assert!(hits("a/**", "a/x/y"));
        assert!(!hits("a/**", "a"));
        assert!(hits("a/***/b", "a/x/y/b"));
        assert!(hits("a/***/b", "a/b"));
        assert!(!hits("a**b", "a/b"));
    }

    #[test]
    fn many_stars_stay_fast() {
        let pattern = "*a".repeat(40) + "b";
        let text = "a".repeat(400);
        assert!(!hits(&pattern, &text));
    }

    /// A POSIX name runs to the first `]`; with no `:` right before it,
    /// the `[` is a plain member, as git reads it.
    #[test]
    fn a_posix_name_ends_at_the_first_bracket() {
        assert!(hits("[[:a]b:]", "ab:]"));
        assert!(hits("[[:a]b:]", "[b:]"));
        assert!(!hits("[[:a]b:]", "bb:]"));
        assert!(hits("a[[:b]x", "a[x"));
        assert!(hits("a[[:b]x", "a:x"));
        assert!(hits("a[[:b]x", "abx"));
        assert!(!hits("a[[:b]x", "acx"));
        assert!(hits("x[[:]y", "x[y"));
        assert!(hits("x[[:]y", "x:y"));
        assert!(!hits("x[[:]y", "x]y"));
        assert!(!hits("x[[::]]y", "x:y"));
        assert!(!hits("x[[:a", "x["));
    }

    /// `?` never matches a `/`, and `**` inside a name is one `*`.
    #[test]
    fn single_wildcards_stay_in_one_name() {
        assert!(!hits("a?b", "a/b"));
        assert!(hits("a?b", "axb"));
        assert!(!hits("a**/b", "ab"));
        assert!(hits("a**/b", "ax/b"));
        assert!(!hits("a**/b", "a/x/b"));
    }

    /// `**` before an escaped `/` crosses directories but needs at least
    /// one, as git reads it.
    #[test]
    fn double_stars_before_an_escaped_slash_cross_directories() {
        assert!(hits("a/**\\/b", "a/x/b"));
        assert!(hits("a/**\\/b", "a/x/y/b"));
        assert!(!hits("a/**\\/b", "a/b"));
        assert!(hits("**\\/b", "x/y/b"));
        assert!(!hits("**\\/b", "b"));
    }

    /// git's `[:space:]` is space, tab, newline and carriage return only.
    #[test]
    fn the_space_class_is_gits() {
        assert!(hits("x[[:space:]]y", "x\ty"));
        assert!(hits("x[[:space:]]y", "x\ry"));
        assert!(!hits("x[[:space:]]y", "x\x0by"));
        assert!(!hits("x[[:space:]]y", "x\x0cy"));
    }

    /// A pattern of a million `**/` matches without deep recursion.
    #[test]
    fn a_long_chain_of_double_stars_matches_on_a_flat_stack() {
        let mut pattern = "**/".repeat(1 << 20);
        pattern.push_str("zz");
        let glob = Glob::compile(pattern.as_bytes()).expect("compiles");
        let run = std::thread::Builder::new()
            .stack_size(64 * 1024)
            .spawn(move || (glob.matches(b"a/zz"), glob.matches(b"a.txt")))
            .expect("thread")
            .join()
            .expect("no overflow");
        assert_eq!(run, (true, false));
    }

    /// A line of a million `[:` after a `[` compiles in linear time: it
    /// has no `]`, so it matches nothing.
    #[test]
    fn many_posix_openers_compile_in_linear_time() {
        let mut pattern = b"[".to_vec();
        pattern.extend(b"[:".repeat(1 << 20));
        let start = std::time::Instant::now();
        assert!(Glob::compile(&pattern).is_none());
        let mut closed = b"[".to_vec();
        closed.extend(b"[:x".repeat(1 << 18));
        closed.extend(b"]y");
        assert!(Glob::compile(&closed).is_some_and(|glob| glob.matches(b"xy")));
        assert!(start.elapsed() < std::time::Duration::from_secs(5));
    }
}

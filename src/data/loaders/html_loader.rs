//! HTML document loader with cleaning and text extraction.

use super::document::{Document, DocumentLoader, LoaderError};
use async_trait::async_trait;
use std::path::PathBuf;
use tokio::fs;
use tracing::debug;

#[cfg(feature = "document-loaders")]
use html5ever::tendril::StrTendril;
#[cfg(feature = "document-loaders")]
use html5ever::tokenizer::states::RawKind;
#[cfg(feature = "document-loaders")]
use html5ever::tokenizer::{
    BufferQueue, CharacterTokens, EndTag, StartTag, Tag, TagToken, Token, TokenSink,
    TokenSinkResult, Tokenizer, TokenizerOpts,
};
#[cfg(feature = "document-loaders")]
use std::cell::RefCell;
#[cfg(feature = "document-loaders")]
use std::collections::HashMap;

/// Default wrap width used by `html2text` when rendering HTML to plain text.
const DEFAULT_HTML_TEXT_WIDTH: usize = 80;

/// Load documents from HTML files.
///
/// Extracts text content and optionally preserves structure.
#[derive(Debug, Clone)]
pub struct HtmlLoader {
    path: PathBuf,
    remove_scripts: bool,
    remove_styles: bool,
    text_width: usize,
}

impl HtmlLoader {
    /// Create a new HTML loader.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            remove_scripts: true,
            remove_styles: true,
            text_width: DEFAULT_HTML_TEXT_WIDTH,
        }
    }

    /// Set whether to remove script tags.
    #[must_use]
    pub const fn with_remove_scripts(mut self, remove: bool) -> Self {
        self.remove_scripts = remove;
        self
    }

    /// Set whether to remove style tags.
    #[must_use]
    pub const fn with_remove_styles(mut self, remove: bool) -> Self {
        self.remove_styles = remove;
        self
    }

    /// Override the target line width used when converting HTML to plain text
    /// via `html2text`. Defaults to [`DEFAULT_HTML_TEXT_WIDTH`]. Values below
    /// `1` are clamped to `1` since `0` causes the renderer to panic.
    #[must_use]
    pub fn with_text_width(mut self, width: usize) -> Self {
        self.text_width = width.max(1);
        self
    }

    /// Remove `<tag>...</tag>` blocks in-place without cloning the entire
    /// input to lowercase. The tag name is matched case-insensitively. Any
    /// unterminated tag (missing `>` or missing closing tag) stops the scan at
    /// that point, preserving the remainder of the input verbatim.
    #[cfg(feature = "document-loaders")]
    fn strip_tag_blocks(input: &str, tag: &str) -> String {
        debug_assert!(
            tag.bytes().all(|b| b.is_ascii_lowercase()),
            "HtmlLoader::strip_tag_blocks expects an ASCII-lowercase tag name"
        );
        let bytes = input.as_bytes();
        let tag_bytes = tag.as_bytes();
        let mut output = String::with_capacity(input.len());
        let mut cursor = 0usize;

        while let Some(open_pos) = find_tag_open(bytes, cursor, tag_bytes) {
            output.push_str(&input[cursor..open_pos]);
            // Skip past `<tag` then find the closing `>` that ends the opening
            // tag. We scan bytes directly; `>` is ASCII so it cannot appear
            // inside a multibyte UTF-8 sequence.
            let after_tag_name = open_pos + 1 + tag_bytes.len();
            let Some(tag_close_rel) = bytes[after_tag_name..].iter().position(|b| *b == b'>')
            else {
                break;
            };
            let after_open = after_tag_name + tag_close_rel + 1;

            if let Some(close_pos) = find_tag_close(bytes, after_open, tag_bytes) {
                cursor = close_pos + tag_bytes.len() + 3;
            } else {
                cursor = after_open;
                break;
            }
        }

        output.push_str(&input[cursor..]);
        output
    }
}

/// Locate the next `<tag` opening sequence starting at `from`, matching the
/// tag name case-insensitively. The byte immediately after `tag` must be one
/// of: whitespace, `>`, or `/` (self-closing) so that e.g. `<scripts>` does
/// not match `<script>`.
#[cfg(feature = "document-loaders")]
fn find_tag_open(bytes: &[u8], from: usize, tag: &[u8]) -> Option<usize> {
    let mut i = from;
    while i + 1 + tag.len() <= bytes.len() {
        if bytes[i] == b'<'
            && bytes[i + 1..i + 1 + tag.len()].eq_ignore_ascii_case(tag)
            && matches!(
                bytes.get(i + 1 + tag.len()),
                Some(b' ' | b'\t' | b'\n' | b'\r' | b'>' | b'/')
            )
        {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Locate the next `</tag>` close sequence starting at `from`.
#[cfg(feature = "document-loaders")]
fn find_tag_close(bytes: &[u8], from: usize, tag: &[u8]) -> Option<usize> {
    let needed = 3 + tag.len(); // </ + tag + >
    let mut i = from;
    while i + needed <= bytes.len() {
        if bytes[i] == b'<'
            && bytes[i + 1] == b'/'
            && bytes[i + 2..i + 2 + tag.len()].eq_ignore_ascii_case(tag)
            && bytes[i + 2 + tag.len()] == b'>'
        {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Reads the first `<title>` and every `<meta>` with `content` and `name` (or
/// `property`) using the HTML tokenizer, so entities, case, quoting, comments and
/// raw text elements follow the HTML parsing rules. A later meta tag with the same
/// name, `title` included, replaces the earlier value.
#[cfg(feature = "document-loaders")]
fn read_head_metadata(html: &str) -> HashMap<String, String> {
    let tokenizer = Tokenizer::new(HeadMetadataSink::default(), TokenizerOpts::default());
    let input = BufferQueue::default();
    input.push_back(StrTendril::from_slice(html));
    // The sink never pauses for scripts, so one call reads the whole input.
    let _ = tokenizer.feed(&input);
    tokenizer.end();
    tokenizer.sink.state.into_inner().into_metadata()
}

/// How the tokenizer reads the text after an HTML start tag named `name`.
#[cfg(feature = "document-loaders")]
fn html_text_mode(name: &str) -> TokenSinkResult<()> {
    match name {
        "script" => TokenSinkResult::RawData(RawKind::ScriptData),
        "style" | "xmp" | "iframe" | "noembed" | "noframes" | "noscript" => {
            TokenSinkResult::RawData(RawKind::Rawtext)
        }
        "textarea" => TokenSinkResult::RawData(RawKind::Rcdata),
        "plaintext" => TokenSinkResult::Plaintext,
        _ => TokenSinkResult::Continue,
    }
}

#[cfg(feature = "document-loaders")]
#[derive(Default)]
struct HeadMetadataSink {
    state: RefCell<HeadMetadata>,
}

#[cfg(feature = "document-loaders")]
#[derive(Default)]
struct HeadMetadata {
    /// The first title, once it is closed.
    title: Option<String>,
    /// The text of the first title while it is open.
    open_title: Option<String>,
    /// Open `<svg>` and `<math>` elements. Their `<title>` is not raw text.
    foreign_depth: usize,
    /// Inside `<select>`, where `<title>` and `<meta>` are ignored.
    in_select: bool,
    meta: Vec<(String, String)>,
}

#[cfg(feature = "document-loaders")]
impl HeadMetadata {
    fn into_metadata(self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        // A title that is never closed runs to the end of the input.
        if let Some(title) = self.title.or(self.open_title) {
            metadata.insert("title".to_owned(), title);
        }
        metadata.extend(self.meta);
        metadata
    }

    fn start_tag(&mut self, tag: &Tag) -> TokenSinkResult<()> {
        let name = &*tag.name;
        if self.in_select {
            return self.start_tag_in_select(tag);
        }
        let in_foreign_title = self.foreign_depth > 0 && self.open_title.is_some();
        if self.foreign_depth > 0 && !in_foreign_title && breaks_out_of_foreign_content(tag) {
            self.foreign_depth = 0;
        }
        let html_rules = self.foreign_depth == 0 || in_foreign_title;
        match name {
            "svg" | "math" if !tag.self_closing => self.foreign_depth += 1,
            "title" => return self.start_title(tag),
            "meta" => self.read_meta(tag),
            "select" if html_rules => self.in_select = true,
            _ => {}
        }
        if !html_rules {
            return TokenSinkResult::Continue;
        }
        html_text_mode(name)
    }

    /// Inside `<select>`, only a few tags change the state.
    fn start_tag_in_select(&mut self, tag: &Tag) -> TokenSinkResult<()> {
        match &*tag.name {
            "select" => self.in_select = false,
            "input" | "keygen" | "textarea" => {
                self.in_select = false;
                return self.start_tag(tag);
            }
            "script" => return TokenSinkResult::RawData(RawKind::ScriptData),
            _ => {}
        }
        TokenSinkResult::Continue
    }

    fn start_title(&mut self, tag: &Tag) -> TokenSinkResult<()> {
        if self.title.is_none() && self.open_title.is_none() {
            self.open_title = Some(String::new());
        }
        if self.foreign_depth == 0 {
            // An HTML title holds raw text up to `</title>`, even when written `<title/>`.
            return TokenSinkResult::RawData(RawKind::Rcdata);
        }
        if tag.self_closing && self.title.is_none() {
            self.title = self.open_title.take();
        }
        TokenSinkResult::Continue
    }

    fn read_meta(&mut self, tag: &Tag) {
        let attr = |key: &str| {
            tag.attrs
                .iter()
                .find(|attr| &*attr.name.local == key)
                .map(|attr| attr.value.to_string())
        };
        if let (Some(name), Some(content)) =
            (attr("name").or_else(|| attr("property")), attr("content"))
        {
            self.meta.push((name, content));
        }
    }

    fn end_tag(&mut self, tag: &Tag) {
        match &*tag.name {
            "title" if self.title.is_none() && self.open_title.is_some() => {
                self.title = self.open_title.take();
            }
            "select" => self.in_select = false,
            "svg" | "math" => self.foreign_depth = self.foreign_depth.saturating_sub(1),
            "p" | "br" if self.open_title.is_none() => self.foreign_depth = 0,
            _ => {}
        }
    }
}

/// HTML start tags that close any open `<svg>` or `<math>` content.
#[cfg(feature = "document-loaders")]
fn breaks_out_of_foreign_content(tag: &Tag) -> bool {
    match &*tag.name {
        "b" | "big" | "blockquote" | "body" | "br" | "center" | "code" | "dd" | "div" | "dl"
        | "dt" | "em" | "embed" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "head" | "hr" | "i"
        | "img" | "li" | "listing" | "menu" | "meta" | "nobr" | "ol" | "p" | "pre" | "ruby"
        | "s" | "small" | "span" | "strong" | "strike" | "sub" | "sup" | "table" | "tt" | "u"
        | "ul" | "var" => true,
        "font" => tag
            .attrs
            .iter()
            .any(|attr| matches!(&*attr.name.local, "color" | "face" | "size")),
        _ => false,
    }
}

#[cfg(feature = "document-loaders")]
impl TokenSink for HeadMetadataSink {
    type Handle = ();

    fn process_token(&self, token: Token, _line_number: u64) -> TokenSinkResult<()> {
        let mut state = self.state.borrow_mut();
        match token {
            TagToken(tag) if tag.kind == StartTag => return state.start_tag(&tag),
            TagToken(tag) if tag.kind == EndTag => state.end_tag(&tag),
            CharacterTokens(text) => {
                if !state.in_select
                    && let Some(title) = state.open_title.as_mut()
                {
                    title.push_str(&text);
                }
            }
            _ => {}
        }
        TokenSinkResult::Continue
    }

    /// Lets `<![CDATA[...]]>` be text inside `<svg>` and `<math>`, as the tree builder does.
    fn adjusted_current_node_present_but_not_in_html_namespace(&self) -> bool {
        self.state.borrow().foreign_depth > 0
    }
}

#[async_trait]
impl DocumentLoader for HtmlLoader {
    #[cfg(feature = "document-loaders")]
    async fn load(&self) -> Result<Vec<Document>, LoaderError> {
        let html_content = fs::read_to_string(&self.path).await?;
        let html_for_text = if self.remove_scripts {
            Self::strip_tag_blocks(&html_content, "script")
        } else {
            html_content.clone()
        };
        let html_for_text = if self.remove_styles {
            Self::strip_tag_blocks(&html_for_text, "style")
        } else {
            html_for_text
        };

        // Metadata (title, meta tags) comes from the original HTML.
        // `html_for_text` has script/style blocks stripped and is what we
        // convert to plain text for the document body.
        let mut metadata = read_head_metadata(&html_content);

        // Convert to plain text; fall back to the script/style-stripped HTML
        // source (moved, not cloned) when html2text cannot parse.
        let text = {
            let bytes = html_for_text.as_bytes();
            html2text::from_read(bytes, self.text_width)
        }
        .unwrap_or_else(|e| {
            debug!(
                path = %self.path.display(),
                error = %e,
                "html2text conversion failed; falling back to raw HTML"
            );
            html_for_text
        });
        metadata.insert("source".to_owned(), self.path.display().to_string());

        Ok(vec![Document::new(text).with_metadata_map(metadata)])
    }

    #[cfg(not(feature = "document-loaders"))]
    async fn load(&self) -> Result<Vec<Document>, LoaderError> {
        Err(LoaderError::LoadFailed(
            "HTML loader requires 'document-loaders' feature".to_string(),
        ))
    }

    fn name(&self) -> &'static str {
        "html"
    }
}

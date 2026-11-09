//! `HtmlLoader` reads the title and the `<meta name/property>` tags with HTML parsing
//! rules: entities, case, quoting, raw text elements, comments, and foreign content.
#![cfg(feature = "document-loaders")]

use caliche::loaders::{DocumentLoader, HtmlLoader};
use std::collections::BTreeMap;

/// Each case is a name, the HTML input, and the expected metadata without `source`.
const CASES: &[(&str, &str, &[(&str, &str)])] = &[
    (
        "entities",
        r#"<html><head><title>Fish &amp; Chips &#39;n&#39; &quot;More&quot; &nbsp;&mdash;&eacute;</title><meta name="description" content="A &amp; B &#39;c&#39; &quot;d&quot;"></head><body>x</body></html>"#,
        &[
            ("description", "A & B 'c' \"d\""),
            ("title", "Fish & Chips 'n' \"More\" \u{a0}\u{2014}\u{e9}"),
        ],
    ),
    (
        "upper case tags and attributes",
        r#"<HTML><HEAD><TITLE>Upper Case</TITLE><META NAME="Author" CONTENT="Jane"><META PROPERTY="og:title" CONTENT="OG"></HEAD></HTML>"#,
        &[
            ("Author", "Jane"),
            ("og:title", "OG"),
            ("title", "Upper Case"),
        ],
    ),
    (
        "quoting",
        r#"<head><meta name='single' content='one'><meta name="double" content="two"><meta name=unquoted content=three></head>"#,
        &[("double", "two"), ("single", "one"), ("unquoted", "three")],
    ),
    (
        "title in a comment",
        "<head><!-- <title>Hidden</title> --><title>Visible</title></head>",
        &[("title", "Visible")],
    ),
    (
        "title in script",
        r#"<head><script>var s = "<title>Script</title>";</script><title>Real</title></head>"#,
        &[("title", "Real")],
    ),
    (
        "title in style",
        "<head><style>/* <title>Style</title> */</style><title>Real</title></head>",
        &[("title", "Real")],
    ),
    (
        "several titles",
        "<head><title>First</title><title>Second</title></head>",
        &[("title", "First")],
    ),
    (
        "whitespace in the title",
        "<head><title>\n  Spaced \t Title\n</title></head>",
        &[("title", "\n  Spaced \t Title\n")],
    ),
    (
        "meta with name and property",
        r#"<head><meta name="n" property="p" content="both"><meta property="p2" name="n2" content="c2"></head>"#,
        &[("n", "both"), ("n2", "c2")],
    ),
    (
        "non-ASCII text",
        r#"<head><title>Café 東京 🚀</title><meta name="ключ" content="значение"></head>"#,
        &[("title", "Café 東京 🚀"), ("ключ", "значение")],
    ),
    (
        "title inside svg",
        "<body><svg><title>Svg <tspan>Title</tspan></title></svg></body>",
        &[("title", "Svg Title")],
    ),
    (
        "svg title before the html title",
        "<svg><title>Svg</title></svg><title>Html</title>",
        &[("title", "Svg")],
    ),
    (
        "title after body",
        "<html><head></head><body><p>x</p></body></html><title>After</title>",
        &[("title", "After")],
    ),
    (
        "nested noscript",
        r#"<head><noscript><noscript><title>NS</title><meta name="ns" content="x"></noscript></noscript><title>Real</title></head>"#,
        &[("title", "Real")],
    ),
    (
        "noscript in body",
        r#"<body><noscript><meta name="ns" content="x"><title>NS</title></noscript></body>"#,
        &[],
    ),
    (
        "meta without content or name",
        r#"<head><meta name="x"><meta content="orphan"><meta charset="utf-8"></head>"#,
        &[],
    ),
    (
        "duplicate meta name",
        r#"<head><meta name="k" content="first"><meta name="k" content="second"></head>"#,
        &[("k", "second")],
    ),
    (
        "meta named title",
        r#"<head><title>Doc</title><meta name="title" content="Meta"></head>"#,
        &[("title", "Meta")],
    ),
    (
        "title in textarea",
        "<body><textarea><title>T</title></textarea><title>Real</title></body>",
        &[("title", "Real")],
    ),
    (
        "unterminated title",
        "<head><title>Open",
        &[("title", "Open")],
    ),
    (
        "title in template",
        "<head><template><title>Tpl</title></template><title>Real</title></head>",
        &[("title", "Tpl")],
    ),
    (
        "cdata in svg title",
        "<svg><title><![CDATA[a<b]]></title></svg>",
        &[("title", "a<b")],
    ),
    (
        "duplicate attribute",
        r#"<head><meta name="a" name="b" content="c"></head>"#,
        &[("a", "c")],
    ),
    (
        "empty title",
        "<head><title></title></head>",
        &[("title", "")],
    ),
    (
        "self-closing title",
        "<head><title/>Text</title></head>",
        &[("title", "Text")],
    ),
    (
        "title in plaintext",
        "<body><plaintext><title>P</title>",
        &[],
    ),
    (
        "title in select",
        "<body><select><title>Sel</title></select></body>",
        &[],
    ),
    (
        "title in table",
        "<body><table><title>Tbl</title></table></body>",
        &[("title", "Tbl")],
    ),
];

#[expect(
    clippy::expect_used,
    reason = "test helper: a failure fails the calling test"
)]
async fn metadata(dir: &std::path::Path, html: &str) -> BTreeMap<String, String> {
    let path = dir.join("page.html");
    std::fs::write(&path, html).expect("write fixture");
    let docs = HtmlLoader::new(&path).load().await.expect("load html");
    let mut map: BTreeMap<String, String> = docs[0].metadata.clone().into_iter().collect();
    assert_eq!(map.remove("source"), Some(path.display().to_string()));
    map
}

#[tokio::test]
async fn html_metadata_follows_html_parsing_rules() {
    let dir = tempfile::tempdir().expect("temp dir");
    let mut mismatches = Vec::new();
    for (name, html, expected) in CASES {
        let expected: BTreeMap<String, String> = expected
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        let actual = metadata(dir.path(), html).await;
        if actual != expected {
            mismatches.push(format!("{name}: {actual:?}"));
        }
    }
    assert_eq!(mismatches, Vec::<String>::new());
}

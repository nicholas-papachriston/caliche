use caliche_web::app::DOCUMENT_TITLE;

/// dx 0.7.6 `replace_or_insert_before("{app_title}", "</title", title)`.
fn dx_076_title_html(html: &str, title: &str) -> String {
    let mut content = html.to_string();
    if content.contains("{app_title}") {
        content = content.replace("{app_title}", title);
    } else if let Some(pos) = content.find("</title") {
        content.insert_str(pos, title);
    }
    content
}

fn web_app_title(toml: &str) -> String {
    let Some((_, after)) = toml.split_once("[web.app]") else {
        panic!("Dioxus.toml missing [web.app]");
    };
    let body = after.split("\n[").next().unwrap_or(after);
    for line in body.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("title") else {
            continue;
        };
        let rest = rest.trim();
        let Some(rest) = rest.strip_prefix('=') else {
            continue;
        };
        return rest.trim().trim_matches('"').to_string();
    }
    panic!("[web.app] missing title");
}

fn title_texts(html: &str) -> Vec<&str> {
    let mut texts = Vec::new();
    let mut rest = html;
    while let Some(start) = rest.find("<title>") {
        rest = &rest[start + "<title>".len()..];
        let end = rest.find("</title>").unwrap_or(rest.len());
        texts.push(&rest[..end]);
        rest = &rest[end..];
    }
    texts
}

#[test]
fn served_index_has_one_caliche_title() {
    let source = include_str!("../index.html");
    let configured = web_app_title(include_str!("../Dioxus.toml"));
    assert_eq!(configured, DOCUMENT_TITLE);
    assert_eq!(source.matches("<title>").count(), 1);
    assert!(
        source.contains("<title>{app_title}</title>"),
        "index.html must leave title text to the dx placeholder"
    );

    let served = dx_076_title_html(source, &configured);
    assert_eq!(title_texts(&served), vec![DOCUMENT_TITLE]);
    assert!(!served.contains(&format!("{DOCUMENT_TITLE}{DOCUMENT_TITLE}")));
    assert!(!served.contains("{app_title}"));

    // The previous source literal is what dx concatenated into the served file.
    let literal = source.replace(
        "<title>{app_title}</title>",
        &format!("<title>{DOCUMENT_TITLE}</title>"),
    );
    let doubled = dx_076_title_html(&literal, DOCUMENT_TITLE);
    assert!(doubled.contains(&format!("<title>{DOCUMENT_TITLE}{DOCUMENT_TITLE}</title>")));
}

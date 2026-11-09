use caliche_web::{CONTEXT_REFERENCES, ContextReference, parse_context_references};

#[test]
fn parse_context_reference() {
    assert_eq!(
        ContextReference::parse("diff"),
        Some(ContextReference::GitDiff)
    );
    assert_eq!(
        ContextReference::parse("clipboard"),
        Some(ContextReference::Clipboard)
    );
    assert_eq!(
        ContextReference::parse("src/main.rs"),
        Some(ContextReference::File("src/main.rs".to_string()))
    );
    assert_eq!(
        ContextReference::parse("src/"),
        Some(ContextReference::Directory("src/".to_string()))
    );
}

#[test]
fn parses_context_references_from_message() {
    let result = parse_context_references("Look at @src/main.rs and fix it");
    assert_eq!(result.message, "Look at and fix it");
    assert_eq!(result.references.len(), 1);

    let result = parse_context_references("@diff shows the changes");
    assert_eq!(result.message, "shows the changes");
    assert_eq!(result.references.len(), 1);
}

#[test]
fn parse_context_references_preserves_order_and_collapses_gaps() {
    let result = parse_context_references("Compare @diff with @status before editing");

    assert_eq!(result.message, "Compare with before editing");
    assert_eq!(
        result.references,
        vec![ContextReference::GitDiff, ContextReference::GitStatus]
    );
}

#[test]
fn parse_context_references_ignores_email_style_at_signs() {
    let result = parse_context_references("Email dev@example.com before using @clipboard");

    assert_eq!(result.message, "Email dev@example.com before using");
    assert_eq!(result.references, vec![ContextReference::Clipboard]);
}

#[test]
fn context_references_list() {
    assert!(CONTEXT_REFERENCES.iter().any(|(r, _)| *r == "@diff"));
    assert!(CONTEXT_REFERENCES.iter().any(|(r, _)| *r == "@clipboard"));
}

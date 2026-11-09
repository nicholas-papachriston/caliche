use caliche::llm::provider_utils::{ContentPartJsonFormat, content_parts_to_json_array};
use caliche::messages::ContentPart;

#[test]
fn google_image_url_uses_text_fallback() {
    let blocks = content_parts_to_json_array(
        &[ContentPart::ImageUrl {
            url: "https://example.com/image.png".into(),
        }],
        ContentPartJsonFormat::Google,
    );

    assert_eq!(
        blocks[0]["text"],
        "[Image URL attachment: https://example.com/image.png]"
    );
}

#[test]
fn google_image_url_fallback_matches_bedrock_format() {
    let url = "https://cdn.example.com/photo.jpg";
    let google = content_parts_to_json_array(
        &[ContentPart::ImageUrl { url: url.into() }],
        ContentPartJsonFormat::Google,
    );
    let bedrock = content_parts_to_json_array(
        &[ContentPart::ImageUrl { url: url.into() }],
        ContentPartJsonFormat::Bedrock,
    );

    assert_eq!(google[0], bedrock[0]);
}

#[test]
fn openai_image_url_preserves_native_shape() {
    let blocks = content_parts_to_json_array(
        &[ContentPart::ImageUrl {
            url: "https://example.com/image.png".into(),
        }],
        ContentPartJsonFormat::OpenAI,
    );

    assert_eq!(blocks[0]["type"], "image_url");
    assert_eq!(
        blocks[0]["image_url"]["url"],
        "https://example.com/image.png"
    );
}

//! `process_document` reads image metadata for every format the `image` crate decodes.
#![cfg(feature = "tools-document")]

use caliche::Tool;
use caliche::agent::tools::ProcessDocumentTool;
use image::{DynamicImage, ImageFormat, Rgb32FImage, RgbImage, RgbaImage};
use serde_json::json;

const WIDTH: u32 = 3;
const HEIGHT: u32 = 2;

/// Each format the tool reads, with its file extension and an image its encoder
/// accepts. The tool detects the format from the file's magic bytes, so TGA, which
/// has none, is not read. DDS is left out because the `image` crate has no DDS encoder.
fn samples() -> Vec<(ImageFormat, &'static str, DynamicImage)> {
    let rgb = DynamicImage::ImageRgb8(RgbImage::new(WIDTH, HEIGHT));
    let rgba = DynamicImage::ImageRgba8(RgbaImage::new(WIDTH, HEIGHT));
    let float = DynamicImage::ImageRgb32F(Rgb32FImage::new(WIDTH, HEIGHT));
    vec![
        (ImageFormat::Png, "png", rgb.clone()),
        (ImageFormat::Jpeg, "jpg", rgb.clone()),
        (ImageFormat::Gif, "gif", rgba.clone()),
        (ImageFormat::WebP, "webp", rgba.clone()),
        (ImageFormat::Bmp, "bmp", rgb.clone()),
        (ImageFormat::Tiff, "tiff", rgb.clone()),
        (ImageFormat::Ico, "ico", rgba.clone()),
        (ImageFormat::Qoi, "qoi", rgba.clone()),
        (ImageFormat::Pnm, "ppm", rgb),
        (
            ImageFormat::Farbfeld,
            "ff",
            DynamicImage::ImageRgba16(rgba.to_rgba16()),
        ),
        (ImageFormat::Hdr, "hdr", float.clone()),
        (ImageFormat::OpenExr, "exr", float),
    ]
}

#[tokio::test]
async fn metadata_reads_dimensions_for_each_decodable_format() {
    let dir = tempfile::tempdir().unwrap();
    let mut failures = Vec::new();
    for (format, ext, image) in samples() {
        let path = dir.path().join(format!("sample.{ext}"));
        // One `image` feature turns on both the encoder and the decoder of a format.
        if let Err(e) = image.save_with_format(&path, format) {
            failures.push(format!("{ext}: {e}"));
            continue;
        }
        let result = ProcessDocumentTool
            .execute(json!({ "path": path.to_str().unwrap(), "operation": "metadata" }))
            .await;
        if !(result.success && result.content.contains("- Dimensions: 3x2 pixels")) {
            failures.push(format!(
                "{ext}: {:?}",
                result.error.unwrap_or(result.content)
            ));
        }
    }
    assert_eq!(failures, Vec::<String>::new());
}

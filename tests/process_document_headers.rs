//! `process_document` metadata reads width, height and color type from the image
//! header for each format and header variant, and reports bad input as before.
#![cfg(feature = "tools-document")]

use caliche::Tool;
use caliche::agent::tools::ProcessDocumentTool;
use image::{DynamicImage, ImageFormat, RgbImage, RgbaImage};
use serde_json::json;
use std::io::Cursor;
use std::path::Path;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/image-headers");

/// What the tool reports: the dimension and color lines, or the error text.
#[derive(Debug, PartialEq, Eq)]
enum Report {
    Metadata(String, String),
    Error(String),
}

fn metadata(dimensions: &str, color: &str) -> Report {
    Report::Metadata(
        format!("- Dimensions: {dimensions} pixels"),
        format!("- Color Type: {color}"),
    )
}

#[expect(
    clippy::expect_used,
    reason = "test helper: a failure fails the calling test"
)]
fn encode(image: &DynamicImage, format: ImageFormat) -> Vec<u8> {
    let mut bytes = Cursor::new(Vec::new());
    image.write_to(&mut bytes, format).expect("encode image");
    bytes.into_inner()
}

#[expect(
    clippy::expect_used,
    reason = "test helper: a failure fails the calling test"
)]
fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(Path::new(FIXTURES).join(name)).expect("read fixture")
}

/// A 3x2 BMP stored top to bottom: the height field is negative.
fn bmp_top_down() -> Vec<u8> {
    let mut bytes = encode(
        &DynamicImage::ImageRgb8(RgbImage::new(3, 2)),
        ImageFormat::Bmp,
    );
    bytes[22..26].copy_from_slice(&(-2i32).to_le_bytes());
    bytes
}

/// A 3x2 uncompressed 8-bit grayscale TIFF in either byte order.
#[expect(
    clippy::expect_used,
    reason = "test helper: a failure fails the calling test"
)]
fn tiff(big_endian: bool) -> Vec<u8> {
    let u16b = |v: u16| {
        if big_endian {
            v.to_be_bytes()
        } else {
            v.to_le_bytes()
        }
    };
    let u32b = |v: u32| {
        if big_endian {
            v.to_be_bytes()
        } else {
            v.to_le_bytes()
        }
    };
    // (tag, type, value): type 3 is SHORT and 4 is LONG.
    let entries: [(u16, u16, u32); 9] = [
        (256, 3, 3),
        (257, 3, 2),
        (258, 3, 8),
        (259, 3, 1),
        (262, 3, 1),
        (273, 4, 122),
        (277, 3, 1),
        (278, 3, 2),
        (279, 4, 6),
    ];
    let mut bytes = Vec::new();
    bytes.extend_from_slice(if big_endian { b"MM" } else { b"II" });
    bytes.extend_from_slice(&u16b(42));
    bytes.extend_from_slice(&u32b(8));
    bytes.extend_from_slice(&u16b(9));
    for (tag, kind, value) in entries {
        bytes.extend_from_slice(&u16b(tag));
        bytes.extend_from_slice(&u16b(kind));
        bytes.extend_from_slice(&u32b(1));
        if kind == 3 {
            bytes.extend_from_slice(&u16b(u16::try_from(value).expect("SHORT value fits u16")));
            bytes.extend_from_slice(&[0, 0]);
        } else {
            bytes.extend_from_slice(&u32b(value));
        }
    }
    bytes.extend_from_slice(&u32b(0));
    assert_eq!(bytes.len(), 122);
    bytes.extend_from_slice(&[0, 40, 80, 120, 160, 200]);
    bytes
}

/// A PNG whose header is valid but whose image data fails its checksum.
#[expect(
    clippy::expect_used,
    reason = "test helper: a failure fails the calling test"
)]
fn png_with_corrupt_pixels() -> Vec<u8> {
    let mut bytes = encode(
        &DynamicImage::ImageRgb8(RgbImage::new(4, 3)),
        ImageFormat::Png,
    );
    let idat = bytes
        .windows(4)
        .position(|w| w == b"IDAT")
        .expect("IDAT chunk");
    bytes[idat + 6] ^= 0xFF;
    bytes
}

fn cases() -> Vec<(&'static str, Vec<u8>, Report)> {
    let rgb = DynamicImage::ImageRgb8(RgbImage::new(3, 2));
    let rgba = DynamicImage::ImageRgba8(RgbaImage::new(3, 2));
    let png = encode(&rgb, ImageFormat::Png);
    vec![
        ("png", png.clone(), metadata("3x2", "Rgb8")),
        (
            "png rgba",
            encode(&rgba, ImageFormat::Png),
            metadata("3x2", "Rgba8"),
        ),
        (
            "gif",
            encode(&rgba, ImageFormat::Gif),
            metadata("3x2", "Rgba8"),
        ),
        (
            "bmp",
            encode(&rgb, ImageFormat::Bmp),
            metadata("3x2", "Rgb8"),
        ),
        ("bmp top-down", bmp_top_down(), metadata("3x2", "Rgb8")),
        ("tiff little-endian", tiff(false), metadata("3x2", "L8")),
        ("tiff big-endian", tiff(true), metadata("3x2", "L8")),
        ("jpeg sof0", fixture("sof0.jpg"), metadata("5x3", "Rgb8")),
        ("jpeg sof2", fixture("sof2.jpg"), metadata("5x3", "Rgb8")),
        ("webp vp8", fixture("vp8.webp"), metadata("6x5", "Rgb8")),
        ("webp vp8l", fixture("vp8l.webp"), metadata("6x5", "Rgb8")),
        ("webp vp8x", fixture("vp8x.webp"), metadata("7x4", "Rgba8")),
        (
            "truncated header",
            png[..20].to_vec(),
            Report::Error("Failed to parse image: unexpected end of file".to_string()),
        ),
        (
            "not an image",
            b"plain text, not an image".to_vec(),
            Report::Error(
                "Failed to parse image: The image format could not be determined".to_string(),
            ),
        ),
    ]
}

#[expect(
    clippy::expect_used,
    reason = "test helper: a failure fails the calling test"
)]
async fn report(dir: &Path, name: &str, bytes: &[u8]) -> Report {
    let path = dir.join(name.replace(' ', "_"));
    std::fs::write(&path, bytes).expect("write image");
    let result = ProcessDocumentTool
        .execute(
            json!({ "path": path.to_str().expect("UTF-8 temp path"), "operation": "metadata" }),
        )
        .await;
    if !result.success {
        return Report::Error(result.error.unwrap_or_default());
    }
    let line = |prefix: &str| {
        result
            .content
            .lines()
            .find(|line| line.starts_with(prefix))
            .unwrap_or_default()
            .to_string()
    };
    Report::Metadata(line("- Dimensions:"), line("- Color Type:"))
}

#[tokio::test]
async fn metadata_reads_each_header_variant() {
    let dir = tempfile::tempdir().expect("temp dir");
    let mut mismatches = Vec::new();
    for (name, bytes, expected) in cases() {
        let actual = report(dir.path(), name, &bytes).await;
        if actual != expected {
            mismatches.push(format!("{name}: {actual:?}"));
        }
    }
    assert_eq!(mismatches, Vec::<String>::new());
}

/// The only change from a full decode: a valid header with corrupt pixel data
/// now reports its metadata instead of a decode error.
#[tokio::test]
async fn metadata_ignores_corrupt_pixel_data() {
    let dir = tempfile::tempdir().expect("temp dir");
    let actual = report(dir.path(), "corrupt.png", &png_with_corrupt_pixels()).await;
    assert_eq!(actual, metadata("4x3", "Rgb8"));
}

use crate::agent::tools::common::{params, path};
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use image::ImageDecoder;
use serde_json::json;
use std::path::Path;
use tokio::process::Command;

const NAME: &str = "process_document";
const DESCRIPTION: &str = "Processes images and documents to extract text, metadata, and information. \
For images: Uses LLM vision capabilities (preferred) or OCR fallback. \
For PDFs: Extracts text content. \
For metadata: Returns image properties (dimensions, format, size). \
Vision mode provides the image for the LLM to analyze directly - most accurate for complex images.";

#[derive(Debug)]
pub struct ProcessDocumentTool;

impl ToolMeta for ProcessDocumentTool {
    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(DESCRIPTION)
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(NAME)
    }

    fn name(&self) -> &str {
        NAME
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the image or document file to process. Supports local paths (relative or absolute) and S3 URIs (s3://bucket-name/path/to/file.ext)"
                },
                "operation": {
                    "type": "string",
                    "enum": ["vision", "ocr", "metadata", "extract_text", "auto"],
                    "description": "Operation to perform: 'vision' uses LLM vision to analyze images (most accurate, no external deps), 'ocr' extracts text using Tesseract (fallback), 'metadata' returns image properties, 'extract_text' extracts text from PDFs, 'auto' automatically chooses best operation - prefers vision for images (default)"
                },
                "language": {
                    "type": "string",
                    "description": "Language code for OCR (e.g., 'eng' for English, 'spa' for Spanish). Default: 'eng'",
                    "default": "eng"
                }
            },
            "required": ["path"]
        })
    }
}

#[async_trait]
impl Tool for ProcessDocumentTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let path_str = match params::extract_string(&arguments, "path", None) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(e),
        };

        let operation = params::extract_string_or(&arguments, "operation", "auto");
        let language = params::extract_string_or(&arguments, "language", "eng");

        // Validate and resolve path
        let normalized_path = match path::validate_and_resolve_path(&path_str, Some(true)).await {
            Ok(p) => p,
            Err(e) => return ToolResult::error(e),
        };

        // Check file size (max 50MB for document processing)
        if let Err(e) = path::validate_file_size(&normalized_path, 50_000_000) {
            return ToolResult::error(e);
        }

        // Determine operation based on file type if auto
        let file_ext = normalized_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let actual_operation = if operation == "auto" {
            Self::auto_detect_operation(&file_ext)
        } else {
            operation.clone()
        };

        match actual_operation.as_str() {
            "vision" => Self::perform_vision(&normalized_path).await,
            "ocr" => Self::perform_ocr(&normalized_path, &language).await,
            "metadata" => Self::extract_image_metadata(&normalized_path).await,
            "extract_text" => Self::extract_pdf_text(&normalized_path).await,
            _ => ToolResult::error(format!("Unknown operation: {actual_operation}")),
        }
    }
}

impl ProcessDocumentTool {
    fn auto_detect_operation(file_ext: &str) -> String {
        match file_ext {
            "pdf" => "extract_text",
            // Prefer vision for images (LLM will handle it), fallback to OCR if vision fails
            "png" | "jpg" | "jpeg" | "webp" | "gif" | "bmp" | "tiff" | "tif" => "vision",
            _ => "metadata",
        }
        .to_string()
    }

    async fn perform_vision(path: &Path) -> ToolResult {
        // Read the image file
        let image_data = match path::read_resolved_bytes_async(path).await {
            Ok(data) => data,
            Err(e) => return ToolResult::error(format!("Failed to read image file: {e}")),
        };

        // Encode to base64 with data URL format
        let base64_data = BASE64.encode(&image_data);

        // Detect media type from file extension
        let media_type = path
            .extension()
            .and_then(|e| e.to_str())
            .map_or("image/jpeg", |ext| match ext.to_lowercase().as_str() {
                "png" => "image/png",
                "jpg" | "jpeg" => "image/jpeg",
                "webp" => "image/webp",
                "gif" => "image/gif",
                "bmp" => "image/bmp",
                _ => "image/jpeg", // default
            });

        // Get file metadata
        let file_size = image_data.len();
        let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("image");

        // Return base64-encoded image that the LLM can process
        // Modern vision-capable LLMs (Claude 3.5, GPT-4V) can analyze images from base64 data
        let data_url = format!("data:{media_type};base64,{base64_data}");

        ToolResult::success(format!(
            "Image loaded: {} ({} bytes, {:.2} MB)\n\
            Format: {}\n\n\
            [Image content provided as data URL for vision analysis]\n\
            {}\n\n\
            The image is now available for analysis. You can describe what you see, extract text, \
            answer questions about the image, or provide any other analysis.\n\n\
            Note: If you cannot process images, you can use operation='ocr' for text-only extraction.",
            filename,
            file_size,
            file_size as f64 / 1_048_576.0,
            media_type,
            data_url
        ))
    }

    async fn perform_ocr(path: &Path, language: &str) -> ToolResult {
        // Check if tesseract is available
        let tesseract_check = Command::new("tesseract").arg("--version").output().await;

        if tesseract_check.is_err() {
            return ToolResult::error(
                "Tesseract OCR is not installed. Please install tesseract-ocr to use OCR functionality. \
                On macOS: 'brew install tesseract', On Ubuntu: 'apt-get install tesseract-ocr'".to_string()
            );
        }

        // Tesseract writes the text to its standard output, so no file is
        // created where another process could plant a link.
        let output = Command::new("tesseract")
            .arg(path.to_str().unwrap_or(""))
            .arg("stdout")
            .arg("-l")
            .arg(language)
            .output()
            .await;

        match output {
            Ok(result) => {
                if !result.status.success() {
                    let error_msg = String::from_utf8_lossy(&result.stderr);
                    return ToolResult::error(format!("OCR failed: {error_msg}"));
                }
                let text = String::from_utf8_lossy(&result.stdout);
                if text.trim().is_empty() {
                    ToolResult::success("No text detected in image")
                } else {
                    ToolResult::success(format!("Extracted text from image:\n\n{}", text.trim()))
                }
            }
            Err(e) => ToolResult::error(format!("Failed to run OCR: {e}")),
        }
    }

    async fn extract_image_metadata(path: &Path) -> ToolResult {
        // Read the image file
        let image_data = match path::read_resolved_bytes_async(path).await {
            Ok(data) => data,
            Err(e) => return ToolResult::error(format!("Failed to read image file: {e}")),
        };

        match read_image_header(&image_data) {
            Ok((width, height, color_type)) => {
                let file_size = image_data.len();

                let file_ext = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("unknown");

                ToolResult::success(format!(
                    "Image Metadata:\n\
                    - Format: {}\n\
                    - Dimensions: {}x{} pixels\n\
                    - Color Type: {:?}\n\
                    - File Size: {} bytes ({:.2} MB)\n\
                    - Aspect Ratio: {:.2}",
                    file_ext.to_uppercase(),
                    width,
                    height,
                    color_type,
                    file_size,
                    file_size as f64 / 1_048_576.0,
                    f64::from(width) / f64::from(height)
                ))
            }
            Err(e) => ToolResult::error(format!("Failed to parse image: {e}")),
        }
    }

    async fn extract_pdf_text(path: &Path) -> ToolResult {
        // Try using pdftotext if available (part of poppler-utils)
        let pdftotext_check = Command::new("pdftotext").arg("-v").output().await;

        if pdftotext_check.is_ok() {
            return Self::extract_pdf_with_pdftotext(path).await;
        }

        // Fallback to lopdf crate
        match path::read_resolved_bytes_async(path).await {
            Ok(bytes) => Self::extract_pdf_with_lopdf(&bytes),
            Err(e) => ToolResult::error(format!("Failed to read PDF file: {e}")),
        }
    }

    async fn extract_pdf_with_pdftotext(path: &Path) -> ToolResult {
        // `-` sends the text to standard output, so no file is created where
        // another process could plant a link.
        let output = Command::new("pdftotext")
            .arg(path.to_str().unwrap_or(""))
            .arg("-")
            .output()
            .await;

        match output {
            Ok(result) => {
                if !result.status.success() {
                    let error_msg = String::from_utf8_lossy(&result.stderr);
                    return ToolResult::error(format!("PDF text extraction failed: {error_msg}"));
                }
                let text = String::from_utf8_lossy(&result.stdout);
                if text.trim().is_empty() {
                    ToolResult::success("No text found in PDF")
                } else {
                    ToolResult::success(format!(
                        "Extracted text from PDF ({} characters):\n\n{}",
                        text.len(),
                        text.trim()
                    ))
                }
            }
            Err(e) => ToolResult::error(format!("Failed to run pdftotext: {e}")),
        }
    }

    fn extract_pdf_with_lopdf(bytes: &[u8]) -> ToolResult {
        match lopdf::Document::load_mem(bytes) {
            Ok(document) => {
                let mut extracted_text = Vec::new();
                let pages = document.get_pages();

                for (page_num, _) in &pages {
                    if let Ok(text) = document.extract_text(&[*page_num]) {
                        extracted_text.push(format!("--- Page {page_num} ---\n{text}"));
                    }
                }

                if extracted_text.is_empty() {
                    ToolResult::success(
                        "No text found in PDF (may be image-based or encrypted)".to_string(),
                    )
                } else {
                    let full_text = extracted_text.join("\n\n");
                    ToolResult::success(format!(
                        "Extracted text from PDF ({} pages, {} characters):\n\n{}",
                        pages.len(),
                        full_text.len(),
                        full_text
                    ))
                }
            }
            Err(e) => ToolResult::error(format!(
                "Failed to load PDF: {e}. Note: pdftotext (poppler-utils) is not installed. \
                    Install it for better PDF support: On macOS: 'brew install poppler', On Ubuntu: 'apt-get install poppler-utils'"
            )),
        }
    }
}

/// Reads the width, height and color type from the image header without decoding the
/// pixels. The decoded size is still checked against the default limits, as a full
/// decode would, so an image too large to decode reports the same error.
fn read_image_header(data: &[u8]) -> image::ImageResult<(u32, u32, image::ColorType)> {
    let decoder = image::ImageReader::new(std::io::Cursor::new(data))
        .with_guessed_format()?
        .into_decoder()?;
    image::Limits::default().reserve(decoder.total_bytes())?;
    let (width, height) = decoder.dimensions();
    Ok((width, height, decoder.color_type()))
}

#[cfg(test)]
#[cfg(target_os = "macos")]
mod temp_output_tests {
    use super::ProcessDocumentTool;
    use crate::agent::tools::Tool;
    use serde_json::json;
    use std::path::{Path, PathBuf};

    /// Links planted at every temp name a run in the next minute could pick,
    /// removed when dropped.
    struct Planted(Vec<PathBuf>);

    impl Planted {
        fn at(prefix: &str, suffix: &str, victim: &Path) -> Self {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock after 1970")
                .as_secs();
            let mut links = Vec::new();
            for secs in now.saturating_sub(1)..now + 60 {
                let link = std::env::temp_dir().join(format!("{prefix}{secs}{suffix}"));
                if std::os::unix::fs::symlink(victim, &link).is_ok() {
                    links.push(link);
                }
            }
            Self(links)
        }
    }

    impl Drop for Planted {
        fn drop(&mut self) {
            for link in &self.0 {
                let _ = std::fs::remove_file(link);
            }
        }
    }

    /// A one-page PDF whose text is `Hello`.
    fn hello_pdf() -> Vec<u8> {
        let stream = "BT /F1 24 Tf 20 100 Td (Hello) Tj ET";
        let objects = [
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R \
             /Resources << /Font << /F1 5 0 R >> >> >>"
                .to_string(),
            format!(
                "<< /Length {} >>\nstream\n{stream}\nendstream",
                stream.len()
            ),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
        ];
        let mut pdf = String::from("%PDF-1.4\n");
        let mut offsets = Vec::new();
        for (index, body) in objects.iter().enumerate() {
            offsets.push(pdf.len());
            pdf.push_str(&format!("{} 0 obj\n{body}\nendobj\n", index + 1));
        }
        let xref = pdf.len();
        pdf.push_str(&format!(
            "xref\n0 {}\n0000000000 65535 f \n",
            objects.len() + 1
        ));
        for offset in offsets {
            pdf.push_str(&format!("{offset:010} 00000 n \n"));
        }
        pdf.push_str(&format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
            objects.len() + 1
        ));
        pdf.into_bytes()
    }

    #[tokio::test]
    async fn pdf_text_is_not_written_through_a_planted_temp_link() {
        let dir = tempfile::tempdir().expect("dir");
        let victim = dir.path().join("victim.txt");
        std::fs::write(&victim, "keep").expect("victim");
        let pdf = dir.path().join("hello.pdf");
        std::fs::write(&pdf, hello_pdf()).expect("pdf");
        let planted = Planted::at("pdf_text_", "", &victim);

        let result = ProcessDocumentTool
            .execute(json!({ "path": pdf.to_string_lossy(), "operation": "extract_text" }))
            .await;
        drop(planted);

        assert_eq!(std::fs::read_to_string(&victim).expect("victim"), "keep");
        assert!(result.success, "{:?}", result.error);
        assert!(result.content.ends_with("\n\nHello"), "{}", result.content);
    }

    #[tokio::test]
    async fn ocr_text_is_not_written_through_a_planted_temp_link() {
        let dir = tempfile::tempdir().expect("dir");
        let victim = dir.path().join("victim.txt");
        std::fs::write(&victim, "keep").expect("victim");
        let png = dir.path().join("blank.png");
        image::RgbImage::from_pixel(32, 32, image::Rgb([255, 255, 255]))
            .save(&png)
            .expect("png");
        let planted = Planted::at("ocr_output_", ".txt", &victim);

        let result = ProcessDocumentTool
            .execute(json!({ "path": png.to_string_lossy(), "operation": "ocr" }))
            .await;
        drop(planted);

        assert_eq!(std::fs::read_to_string(&victim).expect("victim"), "keep");
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.content, "No text detected in image");
    }
}

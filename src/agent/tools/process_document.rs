use crate::agent::tools::common::{params, path};
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use image::ImageDecoder;
use serde_json::json;
use std::path::Path;
use tokio::fs;
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
        let image_data = match fs::read(path).await {
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

        // Create a temporary file for OCR output
        let output_base = std::env::temp_dir().join(format!(
            "ocr_output_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)
        ));

        let output_txt = output_base.with_extension("txt");

        // Run tesseract
        let output = Command::new("tesseract")
            .arg(path.to_str().unwrap_or(""))
            .arg(output_base.to_str().unwrap_or(""))
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

                // Read the OCR output
                match fs::read_to_string(&output_txt).await {
                    Ok(text) => {
                        // Clean up temporary file
                        let _ = fs::remove_file(&output_txt).await;

                        if text.trim().is_empty() {
                            ToolResult::success("No text detected in image")
                        } else {
                            ToolResult::success(format!(
                                "Extracted text from image:\n\n{}",
                                text.trim()
                            ))
                        }
                    }
                    Err(e) => {
                        let _ = fs::remove_file(&output_txt).await;
                        ToolResult::error(format!("Failed to read OCR output: {e}"))
                    }
                }
            }
            Err(e) => ToolResult::error(format!("Failed to run OCR: {e}")),
        }
    }

    async fn extract_image_metadata(path: &Path) -> ToolResult {
        // Read the image file
        let image_data = match fs::read(path).await {
            Ok(data) => data,
            Err(e) => return ToolResult::error(format!("Failed to read image file: {e}")),
        };

        match read_image_header(&image_data) {
            Ok((width, height, color_type)) => {
                let file_size = match fs::metadata(path).await {
                    Ok(m) => m.len(),
                    Err(_) => 0,
                };

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
        Self::extract_pdf_with_lopdf(path)
    }

    async fn extract_pdf_with_pdftotext(path: &Path) -> ToolResult {
        // Create a temporary file for text output
        let output_path = std::env::temp_dir().join(format!(
            "pdf_text_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)
        ));

        let output = Command::new("pdftotext")
            .arg(path.to_str().unwrap_or(""))
            .arg(output_path.to_str().unwrap_or(""))
            .output()
            .await;

        match output {
            Ok(result) => {
                if !result.status.success() {
                    let error_msg = String::from_utf8_lossy(&result.stderr);
                    return ToolResult::error(format!("PDF text extraction failed: {error_msg}"));
                }

                match fs::read_to_string(&output_path).await {
                    Ok(text) => {
                        let _ = fs::remove_file(&output_path).await;

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
                    Err(e) => {
                        let _ = fs::remove_file(&output_path).await;
                        ToolResult::error(format!("Failed to read PDF output: {e}"))
                    }
                }
            }
            Err(e) => ToolResult::error(format!("Failed to run pdftotext: {e}")),
        }
    }

    fn extract_pdf_with_lopdf(path: &Path) -> ToolResult {
        match lopdf::Document::load(path) {
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

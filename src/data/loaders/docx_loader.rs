//! DOCX document loader.

use super::document::{Document, DocumentLoader, LoaderError};
use async_trait::async_trait;
use std::io::Read;
use std::path::PathBuf;
use zip::ZipArchive;

#[derive(Debug, Clone)]
pub struct DocxLoader {
    path: PathBuf,
}

impl DocxLoader {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

#[async_trait]
impl DocumentLoader for DocxLoader {
    async fn load(&self) -> Result<Vec<Document>, LoaderError> {
        let path = self.path.clone();
        let text = tokio::task::spawn_blocking(move || {
            let bytes = std::fs::read(&path)?;
            let reader = std::io::Cursor::new(bytes);
            let mut archive = ZipArchive::new(reader)
                .map_err(|e| LoaderError::ParseError(format!("Invalid DOCX archive: {}", e)))?;
            let mut file = archive
                .by_name("word/document.xml")
                .map_err(|e| LoaderError::ParseError(format!("Missing document.xml: {}", e)))?;
            let mut xml = String::new();
            file.read_to_string(&mut xml).map_err(|e| {
                LoaderError::ParseError(format!("Failed to read document.xml: {}", e))
            })?;

            let mut text = String::new();
            let mut reader = quick_xml::Reader::from_str(&xml);
            loop {
                match reader.read_event() {
                    Ok(quick_xml::events::Event::Text(t)) => {
                        let s = t.decode().map_err(|e| {
                            LoaderError::ParseError(format!("Failed to decode DOCX text node: {e}"))
                        })?;
                        if !s.is_empty() {
                            text.push_str(&s);
                            text.push(' ');
                        }
                    }
                    Ok(quick_xml::events::Event::Eof) => break,
                    Ok(_) => {}
                    Err(e) => {
                        return Err(LoaderError::ParseError(format!(
                            "Failed to parse DOCX XML: {e}"
                        )));
                    }
                }
            }
            Ok(text)
        })
        .await
        .map_err(|e| {
            LoaderError::load_failed_with("Failed to join DOCX parsing task", anyhow::Error::new(e))
        })??;

        Ok(vec![Document::new(text.trim().to_string())])
    }

    fn name(&self) -> &str {
        "docx"
    }
}

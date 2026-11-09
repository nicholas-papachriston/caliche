//! XLSX document loader.

use super::document::{Document, DocumentLoader, LoaderError};
use async_trait::async_trait;
use calamine::{Data, Reader};
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct XlsxLoader {
    path: PathBuf,
}

impl XlsxLoader {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

#[async_trait]
impl DocumentLoader for XlsxLoader {
    async fn load(&self) -> Result<Vec<Document>, LoaderError> {
        let path = self.path.clone();
        let out = tokio::task::spawn_blocking(move || {
            let mut workbook = calamine::open_workbook_auto(&path).map_err(|e| {
                LoaderError::ParseError(format!("Failed to open spreadsheet: {}", e))
            })?;
            let mut out = String::new();
            for sheet_name in workbook.sheet_names() {
                if let Ok(range) = workbook.worksheet_range(&sheet_name) {
                    out.push_str(&format!("\n# Sheet: {}\n", sheet_name));
                    for row in range.rows() {
                        let mut cols = Vec::with_capacity(row.len());
                        for cell in row {
                            let value = match cell {
                                Data::String(s) => s.clone(),
                                Data::Float(n) => n.to_string(),
                                Data::Int(n) => n.to_string(),
                                Data::Bool(b) => b.to_string(),
                                Data::DateTime(dt) => dt.to_string(),
                                Data::DateTimeIso(s) => s.clone(),
                                Data::DurationIso(s) => s.clone(),
                                Data::Error(e) => format!("{:?}", e),
                                Data::Empty => String::new(),
                            };
                            cols.push(value);
                        }
                        out.push_str(&cols.join("\t"));
                        out.push('\n');
                    }
                }
            }
            Ok::<String, LoaderError>(out)
        })
        .await
        .map_err(|e| {
            LoaderError::load_failed_with("Failed to join XLSX parsing task", anyhow::Error::new(e))
        })??;

        Ok(vec![Document::new(out)])
    }

    fn name(&self) -> &str {
        "xlsx"
    }
}

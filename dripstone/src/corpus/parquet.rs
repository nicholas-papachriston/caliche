//! Parquet embedding shards, including Cohere English `en/NNNN.parquet`.
//!
//! A staged directory (`DRIPSTONE_CORPUS_PARQUET_DIR` or `DRIPSTONE_INT_PARQUET_DIR`)
//! is used as-is. Otherwise Cohere shards are downloaded with the resumable
//! loader until the row cap is met.

use std::fs::File;
use std::path::{Path, PathBuf};

use crate::loader::curl::run_curl;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{
    ArrowDictionaryKeyType, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
    UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{ArrayRef, FixedSizeListArray, ListArray};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::errors::DripstoneError;
use crate::loader::download::{DownloadOptions, ensure_downloaded, sanitize_relative_path};

use super::catalog::{COHERE_LANG_PREFIX, COHERE_REPO, COHERE_REVISION};
use super::layout::{MaterializedRows, query_id, train_id};

const ID_COLUMN: &str = "_id";
const TITLE_COLUMN: &str = "title";
const URL_COLUMN: &str = "url";

pub(super) fn cohere_shard_relative(index: usize) -> String {
    format!("{COHERE_LANG_PREFIX}/{index:04}.parquet")
}

pub(super) fn huggingface_dataset_url(repo: &str, revision: &str, filename: &str) -> String {
    format!("https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}")
}

pub(super) fn read_parquet_dir(
    root: &Path,
    recursive: bool,
    emb_column: &str,
    train_max: usize,
    query_max: usize,
) -> Result<MaterializedRows, DripstoneError> {
    let paths = collect_parquet_paths(root, recursive)?;
    read_parquet_paths(
        &paths,
        emb_column,
        train_max,
        query_max,
        Some(root.display().to_string()),
    )
}

pub(super) fn download_cohere_shards(
    source_root: &Path,
    train_max: usize,
    query_max: usize,
    shard_max: usize,
) -> Result<Vec<PathBuf>, DripstoneError> {
    let need = train_max.saturating_add(query_max);
    let token = huggingface_token();
    let options = download_options(token.as_deref(), None);
    let mut paths = Vec::new();
    let mut have = 0_usize;
    for index in 0..shard_max {
        let relative = cohere_shard_relative(index);
        let url = huggingface_dataset_url(COHERE_REPO, COHERE_REVISION, &relative);
        if index > 0 && !remote_ok(&url, token.as_deref())? {
            break;
        }
        let dest = sanitize_relative_path(source_root, &relative)?;
        if index == 0 {
            ensure_downloaded(&url, &dest, &options).map_err(|error| {
                DripstoneError::storage(format!(
                    "{error}; stage the English shards and set DRIPSTONE_CORPUS_PARQUET_DIR to that en/ directory"
                ))
            })?;
        } else {
            ensure_downloaded(&url, &dest, &options)?;
        }
        have = have.saturating_add(parquet_row_count(&dest)?);
        paths.push(dest);
        if have >= need {
            break;
        }
    }
    if paths.is_empty() {
        return Err(DripstoneError::storage(
            "no Cohere en/ shards were downloaded; set DRIPSTONE_CORPUS_PARQUET_DIR to a staged en/ directory",
        ));
    }
    Ok(paths)
}

pub(super) fn read_parquet_paths(
    paths: &[PathBuf],
    emb_column: &str,
    train_max: usize,
    query_max: usize,
    source_path: Option<String>,
) -> Result<MaterializedRows, DripstoneError> {
    let need = train_max.saturating_add(query_max);
    let mut flat = Vec::with_capacity(need);
    let mut metas = Vec::with_capacity(train_max);
    let mut source_ids = Vec::with_capacity(train_max);
    let mut dimension = None;
    let mut exhausted = true;
    'files: for path in paths {
        let file = File::open(path).map_err(|error| {
            DripstoneError::storage_with_source(format!("open {}", path.display()), error)
        })?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|error| {
            DripstoneError::storage(format!("parquet reader {}: {error}", path.display()))
        })?;
        let projection = projection_for(&builder, emb_column)?;
        let reader = builder
            .with_projection(projection)
            .build()
            .map_err(|error| {
                DripstoneError::storage(format!("parquet build {}: {error}", path.display()))
            })?;
        for batch in reader {
            let batch = batch.map_err(|error| {
                DripstoneError::storage(format!("parquet batch {}: {error}", path.display()))
            })?;
            let emb = batch.column_by_name(emb_column).ok_or_else(|| {
                DripstoneError::storage(format!(
                    "column `{emb_column}` missing in {}",
                    path.display()
                ))
            })?;
            let ids = batch.column_by_name(ID_COLUMN);
            let titles = batch.column_by_name(TITLE_COLUMN);
            let urls = batch.column_by_name(URL_COLUMN);
            for row in 0..batch.num_rows() {
                let vector = embedding_list_row(emb, row)?;
                match dimension {
                    Some(expected) if vector.len() != expected => {
                        return Err(DripstoneError::storage(format!(
                            "embedding dim mismatch: expected {expected}, got {}",
                            vector.len()
                        )));
                    }
                    Some(_) => {}
                    None if vector.is_empty() => {
                        return Err(DripstoneError::storage(
                            "empty embedding vector".to_string(),
                        ));
                    }
                    None => dimension = Some(vector.len()),
                }
                if flat.len() < train_max {
                    source_ids.push(ids.and_then(|column| string_cell(column, row)));
                    metas.push(meta_value(
                        source_ids.last().and_then(Option::as_deref),
                        titles.and_then(|column| string_cell(column, row)),
                        urls.and_then(|column| string_cell(column, row)),
                    ));
                }
                flat.push(vector);
                if flat.len() >= need {
                    exhausted = false;
                    break 'files;
                }
            }
        }
    }
    if flat.len() < need {
        return Err(DripstoneError::storage(format!(
            "only read {} embedding rows; need {need} (add Parquet shards, enable DRIPSTONE_CORPUS_PARQUET_RECURSIVE=1, or lower TRAIN_MAX + QUERY_MAX)",
            flat.len()
        )));
    }
    let mut rows = flat.into_iter();
    let train: Vec<Vec<f32>> = rows.by_ref().take(train_max).collect();
    let queries: Vec<Vec<f32>> = rows.take(query_max).collect();
    let dimension = dimension.unwrap_or(0);
    Ok(MaterializedRows {
        train_ids: (0..train.len()).map(train_id).collect(),
        train,
        train_meta: metas,
        query_ids: (0..queries.len()).map(query_id).collect(),
        queries,
        neighbors: None,
        neighbors_width: 0,
        dimension,
        source_exhausted: exhausted,
        source_url: None,
        source_path,
        has_qrels: false,
        judgments: None,
    })
}

fn projection_for(
    builder: &ParquetRecordBatchReaderBuilder<File>,
    emb_column: &str,
) -> Result<ProjectionMask, DripstoneError> {
    let schema = builder.schema();
    let mut indices = Vec::new();
    for (index, field) in schema.fields().iter().enumerate() {
        let name = field.name().as_str();
        if name == emb_column || name == ID_COLUMN || name == TITLE_COLUMN || name == URL_COLUMN {
            indices.push(index);
        }
    }
    if indices.is_empty() {
        return Err(DripstoneError::storage(format!(
            "parquet schema has no `{emb_column}` column"
        )));
    }
    Ok(ProjectionMask::roots(builder.parquet_schema(), indices))
}

fn collect_parquet_paths(root: &Path, recursive: bool) -> Result<Vec<PathBuf>, DripstoneError> {
    let mut paths = Vec::new();
    if root.is_file() {
        if is_parquet(root) {
            paths.push(root.to_path_buf());
            return Ok(paths);
        }
        return Err(DripstoneError::storage(format!(
            "parquet path is not a .parquet file: {}",
            root.display()
        )));
    }
    if !root.is_dir() {
        return Err(DripstoneError::storage(format!(
            "parquet path not found: {}",
            root.display()
        )));
    }
    if recursive {
        collect_recursive(root, &mut paths)?;
    } else {
        let entries = std::fs::read_dir(root).map_err(|error| {
            DripstoneError::storage_with_source(format!("read {}", root.display()), error)
        })?;
        for entry in entries {
            let entry = entry
                .map_err(|error| DripstoneError::storage_with_source("parquet dir entry", error))?;
            let path = entry.path();
            if path.is_file() && is_parquet(&path) {
                paths.push(path);
            }
        }
    }
    paths.sort();
    if paths.is_empty() {
        return Err(DripstoneError::storage(format!(
            "no .parquet files under {}",
            root.display()
        )));
    }
    Ok(paths)
}

fn collect_recursive(root: &Path, paths: &mut Vec<PathBuf>) -> Result<(), DripstoneError> {
    let entries = std::fs::read_dir(root).map_err(|error| {
        DripstoneError::storage_with_source(format!("read {}", root.display()), error)
    })?;
    for entry in entries {
        let entry = entry
            .map_err(|error| DripstoneError::storage_with_source("parquet dir entry", error))?;
        let path = entry.path();
        if path.is_dir() {
            collect_recursive(&path, paths)?;
        } else if path.is_file() && is_parquet(&path) {
            paths.push(path);
        }
    }
    Ok(())
}

fn is_parquet(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("parquet"))
}

fn parquet_row_count(path: &Path) -> Result<usize, DripstoneError> {
    let file = File::open(path).map_err(|error| {
        DripstoneError::storage_with_source(format!("open {}", path.display()), error)
    })?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|error| {
        DripstoneError::storage(format!("parquet metadata {}: {error}", path.display()))
    })?;
    let rows = builder.metadata().file_metadata().num_rows();
    usize::try_from(rows).map_err(|_| {
        DripstoneError::storage(format!("parquet row count {rows} does not fit usize"))
    })
}

fn primitive_values_to_f32(values: &ArrayRef) -> Result<Vec<f32>, DripstoneError> {
    if let Some(array) = values.as_primitive_opt::<Float32Type>() {
        return Ok(array.values().to_vec());
    }
    if let Some(array) = values.as_primitive_opt::<Float64Type>() {
        return Ok(array.values().iter().map(|value| *value as f32).collect());
    }
    Err(DripstoneError::storage(format!(
        "parquet emb inner type {:?} is not f32 or f64",
        values.data_type()
    )))
}

fn embedding_list_row(column: &ArrayRef, row: usize) -> Result<Vec<f32>, DripstoneError> {
    if column.is_null(row) {
        return Err(DripstoneError::storage(format!(
            "parquet: null embedding at row {row}"
        )));
    }
    if let Some(list) = column.as_any().downcast_ref::<FixedSizeListArray>() {
        return primitive_values_to_f32(&list.value(row));
    }
    if let Some(list) = column.as_any().downcast_ref::<ListArray>() {
        return primitive_values_to_f32(&list.value(row));
    }
    Err(DripstoneError::storage(format!(
        "parquet emb column type {:?} is not a list of floats",
        column.data_type()
    )))
}

pub(super) fn string_cell(column: &ArrayRef, row: usize) -> Option<String> {
    if column.is_null(row) {
        return None;
    }
    if let Some(value) = plain_string(column, row) {
        return Some(value);
    }
    dictionary_string::<Int8Type>(column, row)
        .or_else(|| dictionary_string::<Int16Type>(column, row))
        .or_else(|| dictionary_string::<Int32Type>(column, row))
        .or_else(|| dictionary_string::<Int64Type>(column, row))
        .or_else(|| dictionary_string::<UInt8Type>(column, row))
        .or_else(|| dictionary_string::<UInt16Type>(column, row))
        .or_else(|| dictionary_string::<UInt32Type>(column, row))
        .or_else(|| dictionary_string::<UInt64Type>(column, row))
}

fn plain_string(column: &ArrayRef, row: usize) -> Option<String> {
    if let Some(array) = column.as_string_opt::<i32>() {
        return Some(array.value(row).to_string());
    }
    if let Some(array) = column.as_string_opt::<i64>() {
        return Some(array.value(row).to_string());
    }
    if let Some(array) = column.as_string_view_opt() {
        return Some(array.value(row).to_string());
    }
    None
}

fn dictionary_string<K: ArrowDictionaryKeyType>(column: &ArrayRef, row: usize) -> Option<String> {
    let dict = column.as_dictionary_opt::<K>()?;
    let index = dict.key(row)?;
    plain_string(dict.values(), index)
}

fn meta_value(
    source_id: Option<&str>,
    title: Option<String>,
    url: Option<String>,
) -> Option<serde_json::Value> {
    if source_id.is_none() && title.is_none() && url.is_none() {
        return None;
    }
    let mut object = serde_json::Map::new();
    if let Some(id) = source_id {
        object.insert(
            "source_id".to_string(),
            serde_json::Value::String(id.to_string()),
        );
    }
    if let Some(title) = title {
        object.insert(TITLE_COLUMN.to_string(), serde_json::Value::String(title));
    }
    if let Some(url) = url {
        object.insert(URL_COLUMN.to_string(), serde_json::Value::String(url));
    }
    Some(serde_json::Value::Object(object))
}

fn download_options(token: Option<&str>, expected_bytes: Option<u64>) -> DownloadOptions {
    DownloadOptions::default()
        .with_expected_bytes(expected_bytes)
        .with_huggingface_token(token)
}

pub(super) fn huggingface_token() -> Option<String> {
    for name in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Ok(value) = std::env::var(name) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

fn remote_ok(url: &str, token: Option<&str>) -> Result<bool, DripstoneError> {
    let mut headers = Vec::new();
    if let Some(token) = token {
        headers.push(("Authorization".to_string(), format!("Bearer {token}")));
    }
    let output = run_curl(
        &[
            "-sI",
            "-L",
            "--max-redirs",
            "5",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            url,
        ],
        &headers,
    )
    .map_err(|error| DripstoneError::storage_with_source("curl status probe failed", error))?;
    let code = String::from_utf8_lossy(&output.stdout);
    let status = code.trim().parse::<u16>().unwrap_or(0);
    match status {
        200..=299 => Ok(true),
        404 => Ok(false),
        _ => Err(DripstoneError::storage(format!(
            "HEAD {url} returned HTTP {status}"
        ))),
    }
}

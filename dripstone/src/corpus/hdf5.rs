//! Capped reads of ann-benchmarks HDF5 (`train`, `test`, `neighbors`).
//!
//! Hyperslab reads stop at the row cap so GIST's ~3.6 GB file is not decoded whole.

use std::path::Path;

use hdf5_reader::{Hdf5File, SliceInfo, SliceInfoElem};
use ndarray::ArrayD;

use crate::errors::DripstoneError;

use super::layout::{MaterializedRows, query_id, train_id};

pub(super) fn read_hdf5(
    path: &Path,
    train_max: usize,
    query_max: usize,
    neighbors_k: usize,
    source_url: Option<String>,
) -> Result<MaterializedRows, DripstoneError> {
    let file = Hdf5File::open(path).map_err(hdf5_err)?;
    let train_ds = file.dataset("train").map_err(hdf5_err)?;
    let train = read_float_prefix(&train_ds, train_max)?;
    if train.rows.is_empty() {
        return Err(DripstoneError::storage(format!(
            "hdf5 {} has an empty train set",
            path.display()
        )));
    }
    let test_ds = file.dataset("test").map_err(hdf5_err)?;
    let queries = read_float_prefix(&test_ds, query_max)?;
    if queries.dimension != train.dimension {
        return Err(DripstoneError::storage(format!(
            "hdf5 query dim {} != train dim {}",
            queries.dimension, train.dimension
        )));
    }

    let neighbors = match file.dataset("neighbors") {
        Ok(dataset) => read_neighbor_prefix(&dataset, queries.rows.len(), neighbors_k).ok(),
        Err(_) => None,
    };
    let neighbors_width = neighbors.as_ref().map(|read| read.width).unwrap_or(0);
    let neighbor_rows = neighbors.map(|read| read.rows);

    let query_count = queries.rows.len();
    Ok(MaterializedRows {
        train_ids: (0..train.rows.len()).map(train_id).collect(),
        train: train.rows,
        train_meta: Vec::new(),
        query_ids: (0..query_count).map(query_id).collect(),
        queries: queries.rows,
        neighbors: neighbor_rows,
        neighbors_width,
        dimension: train.dimension,
        source_exhausted: train.source_rows <= train_max && queries.source_rows <= query_max,
        source_url,
        source_path: None,
        has_qrels: false,
        judgments: None,
    })
}

struct FloatPrefix {
    rows: Vec<Vec<f32>>,
    dimension: usize,
    source_rows: usize,
}

fn read_float_prefix(
    dataset: &hdf5_reader::Dataset<'_>,
    max_rows: usize,
) -> Result<FloatPrefix, DripstoneError> {
    let shape = dataset.shape();
    if shape.len() != 2 {
        return Err(DripstoneError::storage(format!(
            "expected a 2D float dataset, got ndim {}",
            shape.len()
        )));
    }
    let source_rows = usize::try_from(shape[0])
        .map_err(|_| DripstoneError::storage("hdf5 row count does not fit usize".to_string()))?;
    let dimension = usize::try_from(shape[1])
        .map_err(|_| DripstoneError::storage("hdf5 dimension does not fit usize".to_string()))?;
    let take = max_rows.min(source_rows);
    if take == 0 || dimension == 0 {
        return Ok(FloatPrefix {
            rows: Vec::new(),
            dimension,
            source_rows,
        });
    }
    let selection = row_prefix(take);
    let rows = match dataset.read_slice::<f32>(&selection) {
        Ok(array) => rows_from_f32(array)?,
        Err(error_f32) => match dataset.read_slice::<f64>(&selection) {
            Ok(array) => rows_from_f64(array)?,
            Err(error_f64) => {
                return Err(DripstoneError::storage(format!(
                    "hdf5 float slice: f32: {error_f32}; f64: {error_f64}"
                )));
            }
        },
    };
    Ok(FloatPrefix {
        rows,
        dimension,
        source_rows,
    })
}

struct NeighborPrefix {
    rows: Vec<Vec<i32>>,
    width: usize,
}

fn read_neighbor_prefix(
    dataset: &hdf5_reader::Dataset<'_>,
    max_queries: usize,
    k_cap: usize,
) -> Result<NeighborPrefix, DripstoneError> {
    let shape = dataset.shape();
    if shape.len() != 2 {
        return Err(DripstoneError::storage(format!(
            "neighbors: expected 2D, got ndim {}",
            shape.len()
        )));
    }
    let source_rows = usize::try_from(shape[0]).unwrap_or(0);
    let width = usize::try_from(shape[1]).unwrap_or(0);
    let queries = max_queries.min(source_rows);
    let k = k_cap.min(width);
    if queries == 0 || k == 0 {
        return Ok(NeighborPrefix {
            rows: Vec::new(),
            width,
        });
    }
    let selection = SliceInfo {
        selections: vec![
            SliceInfoElem::Slice {
                start: 0,
                end: u64::try_from(queries).unwrap_or(u64::MAX),
                step: 1,
            },
            SliceInfoElem::Slice {
                start: 0,
                end: u64::try_from(k).unwrap_or(u64::MAX),
                step: 1,
            },
        ],
    };
    let rows = match dataset.read_slice::<i64>(&selection) {
        Ok(array) => rows_from_i64(array)?,
        Err(_) => {
            let array = dataset.read_slice::<i32>(&selection).map_err(hdf5_err)?;
            rows_from_i32(array)?
        }
    };
    Ok(NeighborPrefix { rows, width })
}

fn row_prefix(rows: usize) -> SliceInfo {
    SliceInfo {
        selections: vec![
            SliceInfoElem::Slice {
                start: 0,
                end: u64::try_from(rows).unwrap_or(u64::MAX),
                step: 1,
            },
            SliceInfoElem::Slice {
                start: 0,
                end: u64::MAX,
                step: 1,
            },
        ],
    }
}

fn rows_from_f32(array: ArrayD<f32>) -> Result<Vec<Vec<f32>>, DripstoneError> {
    if array.ndim() != 2 {
        return Err(DripstoneError::storage(format!(
            "expected 2D float dataset, got ndim {}",
            array.ndim()
        )));
    }
    let shape = array.shape();
    let rows = shape[0];
    let dimension = shape[1];
    let mut out = Vec::with_capacity(rows);
    for row_index in 0..rows {
        let row = array.slice(ndarray::s![row_index, ..]);
        let vector: Vec<f32> = row.iter().copied().collect();
        if vector.len() != dimension {
            return Err(DripstoneError::storage(
                "hdf5 row length mismatch".to_string(),
            ));
        }
        out.push(vector);
    }
    Ok(out)
}

fn rows_from_f64(array: ArrayD<f64>) -> Result<Vec<Vec<f32>>, DripstoneError> {
    if array.ndim() != 2 {
        return Err(DripstoneError::storage(format!(
            "expected 2D float dataset, got ndim {}",
            array.ndim()
        )));
    }
    let shape = array.shape();
    let rows = shape[0];
    let dimension = shape[1];
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(rows);
    for row_index in 0..rows {
        let row = array.slice(ndarray::s![row_index, ..]);
        out.push(row.iter().map(|value| *value as f32).collect());
        if out.last().is_some_and(|vector| vector.len() != dimension) {
            return Err(DripstoneError::storage(
                "hdf5 row length mismatch".to_string(),
            ));
        }
    }
    Ok(out)
}

fn rows_from_i64(array: ArrayD<i64>) -> Result<Vec<Vec<i32>>, DripstoneError> {
    if array.ndim() != 2 {
        return Err(DripstoneError::storage(
            "neighbors slice was not 2D".to_string(),
        ));
    }
    let shape = array.shape();
    let mut out = Vec::with_capacity(shape[0]);
    for row_index in 0..shape[0] {
        let row = array.slice(ndarray::s![row_index, ..]);
        let mut neighbors = Vec::with_capacity(shape[1]);
        for value in row {
            let index = i32::try_from(*value).map_err(|_| {
                DripstoneError::storage(format!("neighbor index {value} does not fit i32"))
            })?;
            neighbors.push(index);
        }
        out.push(neighbors);
    }
    Ok(out)
}

fn rows_from_i32(array: ArrayD<i32>) -> Result<Vec<Vec<i32>>, DripstoneError> {
    if array.ndim() != 2 {
        return Err(DripstoneError::storage(
            "neighbors slice was not 2D".to_string(),
        ));
    }
    let shape = array.shape();
    let mut out = Vec::with_capacity(shape[0]);
    for row_index in 0..shape[0] {
        let row = array.slice(ndarray::s![row_index, ..]);
        out.push(row.iter().copied().collect());
    }
    Ok(out)
}

fn hdf5_err(error: hdf5_reader::error::Error) -> DripstoneError {
    DripstoneError::storage(format!("hdf5: {error}"))
}

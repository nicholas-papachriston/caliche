//! Exact neighbor lists for integration recall (same vectors as ingested `base` / `queries`).

use dripstone::{DripstoneError, Metric};

/// Brute-force top-`k` neighbor indices per query over `base`, using `metric`.
///
/// Indices are `0..base.len()` and match `doc{i}` chunk IDs built from `base` row order.
pub fn brute_force_neighbors(
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
    metric: Metric,
) -> Result<Vec<Vec<i32>>, DripstoneError> {
    let k = k.min(base.len()).max(1);
    let mut out = Vec::with_capacity(queries.len());
    for q in queries {
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(base.len());
        for (i, b) in base.iter().enumerate() {
            let s = metric.compute(q, b)?;
            scored.push((i, s));
        }
        if metric.higher_is_better() {
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        let neighbors: Vec<i32> = scored
            .into_iter()
            .take(k)
            .map(|(i, _)| {
                i32::try_from(i).map_err(|_| {
                    DripstoneError::storage("neighbor index does not fit i32".to_string())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        out.push(neighbors);
    }
    Ok(out)
}

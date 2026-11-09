use super::DenseMetric;
use crate::errors::DripstoneError;

pub(crate) struct ScoreParts {
    pub dot: f32,
    pub euclidean: f32,
    pub manhattan: f32,
    pub chebyshev: f32,
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn accumulate_pair(
    metric: DenseMetric,
    query: f32,
    candidate: f32,
    parts: &mut ScoreParts,
    candidate_squared_sum: &mut f32,
    track_squared: bool,
) {
    match metric {
        DenseMetric::Cosine => {
            parts.dot += query * candidate;
            if track_squared {
                *candidate_squared_sum += candidate * candidate;
            }
        }
        DenseMetric::Dot => parts.dot += query * candidate,
        DenseMetric::Euclidean => {
            let diff = query - candidate;
            parts.euclidean += diff * diff;
        }
        DenseMetric::Manhattan => parts.manhattan += (query - candidate).abs(),
        DenseMetric::Chebyshev => {
            parts.chebyshev = parts.chebyshev.max((query - candidate).abs());
        }
    }
}

pub(crate) fn finish_scored(
    metric: DenseMetric,
    parts: ScoreParts,
    query_norm: Option<f32>,
    candidate_norm: Option<f32>,
    missing_candidate: &'static str,
) -> Result<f32, DripstoneError> {
    match metric {
        DenseMetric::Cosine => {
            let query_norm = query_norm.ok_or_else(|| {
                DripstoneError::storage("missing prepared query norm for cosine scoring")
            })?;
            let candidate_norm =
                candidate_norm.ok_or_else(|| DripstoneError::storage(missing_candidate))?;
            Ok(super::cosine_or_zero(parts.dot, query_norm, candidate_norm))
        }
        DenseMetric::Dot => Ok(parts.dot),
        DenseMetric::Euclidean => Ok(parts.euclidean.sqrt()),
        DenseMetric::Manhattan => Ok(parts.manhattan),
        DenseMetric::Chebyshev => Ok(parts.chebyshev),
    }
}

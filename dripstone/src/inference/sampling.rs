//! Sampling utilities for text generation.
//!
//! Provides common sampling strategies: greedy, temperature, top-k, top-p (nucleus).

use crate::errors::DripstoneError;

/// Sample the next token from logits using the specified parameters.
///
/// # Arguments
/// * `logits` - Raw logits from the model (shape: [vocab_size])
/// * `temperature` - Sampling temperature (1.0 = no change, <1.0 = more deterministic, >1.0 = more random)
/// * `top_k` - If Some, only sample from the top K tokens
/// * `top_p` - If Some, only sample from tokens with cumulative probability <= top_p (nucleus sampling)
/// * `repetition_penalty` - If Some, apply penalty to previously generated tokens (1.0 = no penalty, >1.0 = reduce repetition)
/// * `previous_tokens` - If Some, list of previously generated token IDs to apply penalty to
pub fn sample_token(
    logits: &[f32],
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    repetition_penalty: Option<f32>,
    previous_tokens: Option<&[usize]>,
) -> Result<usize, DripstoneError> {
    if logits.is_empty() {
        return Err(DripstoneError::storage("empty logits"));
    }

    // Greedy / near-greedy: avoid full vocab allocation + sort.
    let greedy_temp = temperature <= 0.0;
    let needs_penalty = matches!(
        (repetition_penalty, previous_tokens),
        (Some(penalty), Some(prev)) if penalty > 0.0 && penalty.total_cmp(&1.0).is_ne() && !prev.is_empty()
    );
    if greedy_temp && !needs_penalty {
        return sample_greedy(logits);
    }

    let mut logits_vec: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(idx, &val)| (idx, val))
        .collect();

    apply_repetition_penalty(
        &mut logits_vec,
        needs_penalty,
        repetition_penalty,
        previous_tokens,
    );

    if greedy_temp {
        return logits_vec
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .ok_or_else(|| DripstoneError::storage("empty logits"));
    }

    scale_and_rank_logits(&mut logits_vec, temperature, top_k, top_p);
    let mut probs = softmax_pairs(&logits_vec);
    apply_nucleus(&mut probs, top_p);
    sample_from_distribution(&probs)
}

fn apply_repetition_penalty(
    logits_vec: &mut [(usize, f32)],
    needs_penalty: bool,
    repetition_penalty: Option<f32>,
    previous_tokens: Option<&[usize]>,
) {
    if !needs_penalty {
        return;
    }
    let Some((penalty, prev_tokens)) = repetition_penalty.zip(previous_tokens) else {
        return;
    };
    let prev_set: std::collections::HashSet<usize> = prev_tokens.iter().copied().collect();
    for (token_id, logit) in logits_vec {
        if !prev_set.contains(token_id) {
            continue;
        }
        if *logit > 0.0 {
            *logit /= penalty;
        } else {
            *logit *= penalty;
        }
    }
}

fn scale_and_rank_logits(
    logits_vec: &mut Vec<(usize, f32)>,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
) {
    if temperature.total_cmp(&1.0).is_ne() && temperature > 0.0 {
        let inv_temp = 1.0 / temperature;
        for (_, logit) in logits_vec.iter_mut() {
            *logit *= inv_temp;
        }
    }

    let needs_sort = top_k.is_some() || top_p.is_some();
    if let Some(k) = top_k {
        logits_vec
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        logits_vec.truncate(k.max(1));
    } else if needs_sort {
        logits_vec
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    }
}

fn softmax_pairs(logits_vec: &[(usize, f32)]) -> Vec<(usize, f32)> {
    let max_logit = logits_vec
        .iter()
        .map(|(_, l)| *l)
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits_vec.iter().map(|(_, l)| (l - max_logit).exp()).sum();
    logits_vec
        .iter()
        .map(|(idx, l)| (*idx, (l - max_logit).exp() / exp_sum))
        .collect()
}

fn apply_nucleus(probs: &mut Vec<(usize, f32)>, top_p: Option<f32>) {
    let Some(p) = top_p else {
        return;
    };
    let mut cumsum = 0.0f32;
    let mut cutoff_idx = probs.len();
    for (i, (_, prob)) in probs.iter().enumerate() {
        cumsum += prob;
        if cumsum >= p {
            cutoff_idx = i + 1;
            break;
        }
    }
    probs.truncate(cutoff_idx);
    let sum: f32 = probs.iter().map(|(_, p)| *p).sum();
    if sum > 0.0 {
        for (_, prob) in probs.iter_mut() {
            *prob /= sum;
        }
    }
}

fn sample_from_distribution(probs: &[(usize, f32)]) -> Result<usize, DripstoneError> {
    if probs.is_empty() {
        return Err(DripstoneError::storage(
            "no valid token after sampling (empty distribution)",
        ));
    }
    let r: f32 = fastrand::f32();
    let mut cumsum = 0.0f32;
    let mut selected_token = probs[0].0;

    for (idx, prob) in probs {
        cumsum += prob;
        if r <= cumsum {
            selected_token = *idx;
            break;
        }
    }

    Ok(selected_token)
}

/// Sample the next token greedily (highest logit).
pub fn sample_greedy(logits: &[f32]) -> Result<usize, DripstoneError> {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .ok_or_else(|| DripstoneError::storage("empty logits"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_sampling() {
        let logits = vec![1.0, 3.0, 2.0, 0.5];
        let token = sample_greedy(&logits).unwrap();
        assert_eq!(token, 1); // Index of max value (3.0)
    }

    #[test]
    fn sample_token_temperature_zero_is_greedy() {
        let logits = vec![1.0, 3.0, 2.0, 0.5];
        let token = sample_token(&logits, 0.0, None, None, None, None).unwrap();
        assert_eq!(token, 1);
    }

    #[test]
    fn test_temperature_sampling() {
        let logits = vec![1.0, 1.0, 1.0];
        // With high temperature, should still sample, but probabilities more uniform
        let _token = sample_token(&logits, 2.0, None, None, None, None).unwrap();
        // Can't assert exact value due to randomness, but shouldn't panic
    }

    #[test]
    fn test_top_k_sampling() {
        let logits = vec![10.0, 9.0, 8.0, 1.0, 0.5];
        let _token = sample_token(&logits, 1.0, Some(3), None, None, None).unwrap();
        // Should only sample from top 3 (indices 0, 1, 2)
    }

    #[test]
    fn test_repetition_penalty() {
        let logits = vec![10.0, 9.0, 8.0, 1.0, 0.5];
        let previous_tokens = vec![0, 1]; // Tokens we've already generated
        // With repetition penalty, previously generated tokens should have lower logits
        let _token =
            sample_token(&logits, 1.0, None, None, Some(1.5), Some(&previous_tokens)).unwrap();
        // Should penalize tokens 0 and 1, making them less likely
    }
}

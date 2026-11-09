//! `TiktokenCounter::for_model` maps every model to `o200k_base` or `cl100k_base`.

#![cfg(feature = "token-tiktoken")]

use caliche::memory::TiktokenCounter;
use caliche::memory::TokenCounter;

const SAMPLE: &str =
    "Grüße aus München — 東京タワーへようこそ!\n\t\t        let x = 42; // ünïcödé\n";

fn count(model: &str) -> Option<usize> {
    TiktokenCounter::for_model(model)
        .ok()
        .map(|counter| counter.estimate_text_tokens(SAMPLE))
}

#[test]
fn o200k_models_use_o200k() {
    assert_eq!(count("gpt-4o"), Some(O200K));
}

#[test]
fn cl100k_models_use_cl100k() {
    assert_eq!(count("gpt-4"), Some(CL100K));
}

#[test]
fn p50k_and_r50k_models_use_cl100k() {
    assert_eq!(count("text-davinci-003"), Some(CL100K));
    assert_eq!(count("davinci"), Some(CL100K));
}

#[test]
fn unknown_models_use_cl100k() {
    assert_eq!(count("not-a-model"), Some(CL100K));
}

const O200K: usize = 28;
const CL100K: usize = 36;

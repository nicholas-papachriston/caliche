//! Local Model Inference Example
//!
//! This example demonstrates how to use local model weight inference
//! with Caliche. Supports loading models from local files or
//! HuggingFace Hub repositories using multiple backends.
//!
//! # Supported Backends
//!
//! | Backend | Model Format | Best For |
//! |---------|--------------|----------|
//! | Candle  | SafeTensors  | GPU/CPU inference, HuggingFace models |
//! | ORT     | ONNX         | Cross-platform, optimized inference |
//! | MLX     | SafeTensors  | Apple Silicon GPU (feature `inference-mlx`) |
//! | GGUF    | llama.cpp    | Quantized models (`local-inference-llamacpp`) |
//!
//! # Prerequisites
//!
//! Enable the `local-inference` feature (Candle + ORT). Add
//! `local-inference-llamacpp` only when you need GGUF:
//! ```bash
//! cargo build --features local-inference
//! cargo build --features local-inference,local-inference-llamacpp
//! ```
//!
//! # Usage
//!
//! ```bash
//! cargo run --example local_inference --features local-inference
//! ```
//!
//! Set `CALICHE_LOCAL_INFERENCE_DEMO=quality` for a higher-quality Candle model (slower).
//! Omit or set `CALICHE_LOCAL_INFERENCE_DEMO=fast` for a tiny model and quick run (default).
//! Call [`LocalInferenceProvider::fit_catalog_models`] (or CLI `/local-model fit`)
//! before loading large weights.

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]
#![expect(
    clippy::expect_used,
    reason = "the demo token limits are small constants that fit u32"
)]

use caliche::llm::provider::Provider;
use caliche::llm::{
    CandleBootstrapConfig, FitOptions, FitStatus, GenerationParams, InferenceBackendType,
    LocalInferenceBootstrapConfig, LocalInferenceConfig, LocalInferenceProvider,
    OrtBootstrapConfig,
};
use caliche::messages::{CompletionRequest, Message};
use dripstone::inference::DeviceConfig;
use dripstone::loader::ModelSource;
use futures_util::StreamExt;
use std::sync::Arc;

/// Demo mode: `fast` (tiny model, quick run) or `quality` (larger model, better output, slower).
fn demo_mode() -> DemoMode {
    match std::env::var("CALICHE_LOCAL_INFERENCE_DEMO").as_deref() {
        Ok("quality") => DemoMode::Quality,
        _ => DemoMode::Fast,
    }
}

#[derive(Clone, Copy)]
enum DemoMode {
    Fast,
    Quality,
}

/// Candle model and generation settings for the selected demo mode.
struct CandleDemoConfig {
    local_dir: &'static str,
    repo: &'static str,
    temperature: f32,
    max_tokens: usize,
    completion_max_tokens: usize,
    stream_max_tokens: usize,
    system_msg: &'static str,
    user_msg: &'static str,
    stream_prompt: &'static str,
    label: &'static str,
}

impl CandleDemoConfig {
    fn for_mode(mode: DemoMode) -> Self {
        match mode {
            DemoMode::Fast => CandleDemoConfig {
                local_dir: "./models/tiny-random-llama-candle",
                repo: "hf-internal-testing/tiny-random-LlamaForCausalLM",
                temperature: 0.3,
                max_tokens: 48,
                completion_max_tokens: 32,
                stream_max_tokens: 16,
                system_msg: "You are a concise assistant. Keep responses under 2 sentences.",
                user_msg: "In one sentence, what is Rust?",
                stream_prompt: "Say hello in exactly five words, then stop.",
                label: "tiny random Llama",
            },
            DemoMode::Quality => CandleDemoConfig {
                local_dir: "./models/tinyllama-1.1b-candle",
                repo: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                temperature: 0.7,
                max_tokens: 256,
                completion_max_tokens: 100,
                stream_max_tokens: 50,
                system_msg: "You are a helpful assistant.",
                user_msg: "What is Rust programming language?",
                stream_prompt: "Say hello in 5 words.",
                label: "TinyLlama 1.1B",
            },
        }
    }

    fn bootstrap(&self) -> LocalInferenceBootstrapConfig {
        LocalInferenceBootstrapConfig {
            candle: Some(CandleBootstrapConfig {
                local_dir: self.local_dir.into(),
                repo: Some(self.repo.to_string()),
                ..CandleBootstrapConfig::default()
            }),
            ..LocalInferenceBootstrapConfig::default()
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing for logging
    tracing_subscriber::fmt::init();

    let mode = demo_mode();
    println!(
        "  Demo mode: {} (set CALICHE_LOCAL_INFERENCE_DEMO=quality for larger model)\n",
        match mode {
            DemoMode::Fast => "fast",
            DemoMode::Quality => "quality",
        }
    );

    // =========================================================================
    // Example 1: Candle Backend with HuggingFace Model
    // =========================================================================
    example_candle_huggingface(mode).await;

    // =========================================================================
    // Example 2: ONNX Runtime Backend with Local Model
    // =========================================================================
    example_ort_local().await;

    // =========================================================================
    // Example 3: MLX Backend (Apple Silicon)
    // =========================================================================
    example_mlx().await;

    // =========================================================================
    // Example 4: Using as Provider with Streaming
    // =========================================================================
    example_provider_streaming(mode).await;

    // =========================================================================
    // Example 5: Custom Generation Parameters
    // =========================================================================
    example_custom_generation_params(mode).await;

    // =========================================================================
    // Example 6: Model Source Auto-Detection
    // =========================================================================
    example_model_source_detection();

    println!("=== Example 7: Catalog Fit (host memory probe) ===\n");
    example_catalog_fit();

    Ok(())
}

/// Example 1: Using Candle backend with a bootstrapped local model
async fn example_candle_huggingface(mode: DemoMode) {
    let demo = CandleDemoConfig::for_mode(mode);
    let config = LocalInferenceConfig::new(
        ModelSource::huggingface(demo.repo, None),
        InferenceBackendType::Candle,
    )
    .with_bootstrap(demo.bootstrap())
    .with_temperature(demo.temperature)
    .with_max_tokens(demo.max_tokens);

    println!("  Model: {} (local file)", demo.local_dir);
    println!("  Backend: Candle (SafeTensors)");
    println!("  Temperature: {}", demo.temperature);
    println!("  Max Tokens: {}", demo.max_tokens);

    match caliche::llm::LocalInferenceProvider::with_config(config).await {
        Ok(provider) => {
            println!("  Provider created: {}", provider.name());

            let messages = Arc::new(vec![
                Message::system(demo.system_msg),
                Message::user(demo.user_msg),
            ]);

            let mut request = CompletionRequest::new(messages);
            request.max_tokens = Some(
                demo.completion_max_tokens
                    .try_into()
                    .expect("completion_max_tokens fits u32"),
            );
            request.temperature = Some(demo.temperature);

            match provider.complete(request).await {
                Ok(response) => {
                    println!("  Response: {}", response.content.extract_text());
                }
                Err(e) => {
                    println!(
                        "  Note: Generation failed (expected if model not downloaded): {}",
                        e
                    );
                }
            }
        }
        Err(e) => {
            println!("  Note: {}", e);
            println!("  (Common causes: bootstrap download failed or missing model files)");
        }
    }
    println!();
}

/// Example 2: Using ONNX Runtime backend with a modern local ONNX model file
async fn example_ort_local() {
    // Configure for a local ONNX model
    // Bootstrap ensures model.onnx and tokenizer.json are present.
    let model_path = "./models/qwen2.5-1.5b-instruct/model.onnx";
    let bootstrap = LocalInferenceBootstrapConfig {
        ort: Some(OrtBootstrapConfig {
            local_dir: "./models/qwen2.5-1.5b-instruct".into(),
            repo: "Xenova/distilgpt2".to_string(),
            ..OrtBootstrapConfig::default()
        }),
        ..LocalInferenceBootstrapConfig::default()
    };

    let config =
        LocalInferenceConfig::new(ModelSource::local(model_path), InferenceBackendType::Ort)
            .with_bootstrap(bootstrap)
            .with_temperature(0.8)
            .with_max_tokens(512);

    println!("  Model: {} (local file)", model_path);
    println!("  Backend: ONNX Runtime");
    println!("  Temperature: 0.8");
    println!("  Max Tokens: 512");

    match caliche::llm::LocalInferenceProvider::with_config(config).await {
        Ok(provider) => {
            println!("  Provider created: {}", provider.name());
        }
        Err(e) => {
            println!("  Note: {}", e);
            println!("  (Bootstrap should create local model files on first run)");
        }
    }
    println!();
}

/// Example 3: Using MLX backend (Apple Silicon).
///
/// Requires `inference-mlx` feature and runs on macOS with Metal. Generation is
/// stubbed until the MLX forward pass is implemented; this demonstrates config and device selection.
async fn example_mlx() {
    let model_dir = "./models/tiny-random-llama-candle";
    let repo = "hf-internal-testing/tiny-random-LlamaForCausalLM";
    let bootstrap = LocalInferenceBootstrapConfig {
        candle: Some(CandleBootstrapConfig {
            local_dir: model_dir.into(),
            repo: Some(repo.to_string()),
            ..CandleBootstrapConfig::default()
        }),
        ..LocalInferenceBootstrapConfig::default()
    };

    let config = LocalInferenceConfig::new(
        ModelSource::huggingface(repo, None),
        InferenceBackendType::Mlx,
    )
    .with_device(DeviceConfig::Mlx)
    .with_bootstrap(bootstrap)
    .with_temperature(0.3)
    .with_max_tokens(32);

    println!("  Model: {} (MLX backend)", model_dir);
    println!("  Backend: MLX (Apple Silicon)");
    println!("  Device: Mlx");

    match caliche::llm::LocalInferenceProvider::with_config(config).await {
        Ok(provider) => {
            println!("  Provider created: {}", provider.name());

            let messages = Arc::new(vec![
                Message::system("You are a concise assistant."),
                Message::user("In one sentence, what is Rust?"),
            ]);
            let mut request = CompletionRequest::new(messages);
            request.max_tokens = Some(24);
            request.temperature = Some(0.3);

            match provider.complete(request).await {
                Ok(response) => {
                    println!("  Response: {}", response.content.extract_text());
                }
                Err(e) => {
                    println!("  MLX generation failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("  Note: {}", e);
            println!("  (Enable 'inference-mlx' and run on Apple Silicon for MLX backend)");
        }
    }
    println!();
}

/// Example 4: Using LocalInferenceProvider with streaming
async fn example_provider_streaming(mode: DemoMode) {
    let demo = CandleDemoConfig::for_mode(mode);
    let config = LocalInferenceConfig::new(
        ModelSource::huggingface(demo.repo, None),
        InferenceBackendType::Candle,
    )
    .with_bootstrap(demo.bootstrap())
    .with_max_tokens(demo.stream_max_tokens);

    match caliche::llm::LocalInferenceProvider::with_config(config).await {
        Ok(provider) => {
            println!("  Streaming example with {}:", demo.label);

            let messages = Arc::new(vec![Message::user(demo.stream_prompt)]);

            let mut request = CompletionRequest::new(messages);
            request.max_tokens = Some(
                demo.stream_max_tokens
                    .try_into()
                    .expect("stream_max_tokens fits u32"),
            );

            print!("  Response: ");
            let mut stream = provider.stream(request);
            while let Some(event) = stream.next().await {
                match event {
                    Ok(caliche::messages::StreamEvent::TextChunk { content }) => {
                        print!("{}", content);
                    }
                    Ok(caliche::messages::StreamEvent::Done) => {
                        println!();
                        break;
                    }
                    Err(e) => {
                        println!("\n  Streaming error: {}", e);
                        break;
                    }
                    _ => {}
                }
            }
        }
        Err(e) => {
            println!("  Note: {}", e);
        }
    }
    println!();
}

/// Example 5: Custom generation parameters
async fn example_custom_generation_params(mode: DemoMode) {
    let demo = CandleDemoConfig::for_mode(mode);
    let params = GenerationParams::new()
        .with_temperature(0.9) // Higher = more creative
        .with_top_p(0.95) // Nucleus sampling
        .with_top_k(40) // Top-k sampling
        .with_max_tokens(2048) // Maximum output length
        .with_repetition_penalty(1.1) // Reduce repetition
        .with_stop_sequence("\n\n") // Stop on double newline
        .with_stop_sequence("END"); // Stop on END token

    let config = LocalInferenceConfig::new(
        ModelSource::huggingface(demo.repo, None),
        InferenceBackendType::Candle,
    )
    .with_bootstrap(demo.bootstrap())
    .with_generation_params(params);

    println!("  Configuration created with custom parameters.");
    println!("  Model: {} (local file)", demo.local_dir);

    match caliche::llm::LocalInferenceProvider::with_config(config).await {
        Ok(provider) => {
            println!("  Provider ready: {}", provider.name());
        }
        Err(e) => {
            println!("  Note: {}", e);
        }
    }
}

/// Example 7: Probe host RAM and print which catalog models fit safely.
fn example_catalog_fit() {
    match LocalInferenceProvider::probe_host_capabilities(FitOptions::default()) {
        Ok(host) => {
            println!(
                "  Host RAM: {} available / {} total",
                host.available_ram_bytes, host.total_ram_bytes
            );
            if let Some(vram) = host.available_vram_bytes {
                println!("  VRAM free: {vram}");
            }
            for note in &host.notes {
                println!("  Note: {note}");
            }
        }
        Err(error) => println!("  Host probe failed: {error}"),
    }

    match LocalInferenceProvider::fit_catalog_models(FitOptions::default()) {
        Ok(fits) => {
            for fit in fits {
                let status = match fit.status {
                    FitStatus::FitsComfortably => "OK",
                    FitStatus::FitsTight => "TIGHT",
                    FitStatus::ExceedsMemory => "EXCEEDS",
                    FitStatus::Unknown => "UNKNOWN",
                };
                println!(
                    "  [{status}] {}  n_ctx={} max_safe={}",
                    fit.model_id, fit.recommended_n_ctx, fit.max_safe_n_ctx
                );
            }
        }
        Err(error) => println!("  Catalog fit failed: {error}"),
    }
    println!();
}

/// Example 6: Model source auto-detection
fn example_model_source_detection() {
    println!("  ModelSource::from_str() auto-detects local vs HuggingFace:");

    let examples = [
        ("Qwen/Qwen2.5-3B-Instruct", "HuggingFace repo"),
        ("Qwen/Qwen2.5-1.5B-Instruct", "HuggingFace repo"),
        ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "HuggingFace repo"),
        (
            "./models/tiny-random-llama-candle",
            "Local Candle model directory",
        ),
        ("./models/model.onnx", "Local (relative path)"),
        ("../shared/model.gguf", "Local (parent path)"),
        ("/absolute/path/model.safetensors", "Local (absolute path)"),
        (
            "sentence-transformers/all-MiniLM-L12-v2",
            "HuggingFace repo",
        ),
    ];

    for (input, description) in examples {
        let source = ModelSource::from_str(input);
        let detected = if source.is_local() {
            "Local"
        } else {
            "HuggingFace"
        };
        println!("  \"{}\"", input);
        println!("    → Detected: {} ({})", detected, description);
    }
}

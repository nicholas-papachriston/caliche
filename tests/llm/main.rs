//! LLM provider unit/integration tests (extracted from `src/llm/` inline modules).

mod anthropic_stream;
#[cfg(feature = "azure-openai")]
mod azure_compat_config;
mod cohere_stream;
mod content_part;
#[cfg(all(feature = "tools-rag", feature = "lmstudio"))]
mod embedding_stop;
mod google_stream;
mod local_first_token;
mod local_model_load;
mod streaming_accumulator;
mod streaming_request_retry;

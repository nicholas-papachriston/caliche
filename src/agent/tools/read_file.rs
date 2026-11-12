use crate::agent::tools::common::{errors, params, path};
use crate::agent::tools::hashline;
use crate::agent::tools::read_window::resolve_read_window;
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde_json::json;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader};
use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::{Instant, SystemTime};

mod outline;
use outline::structural_outline;

/// Cached file entry: mtime, size, and formatted content for fast reuse when unchanged.
#[derive(Clone)]
struct FileCacheEntry {
    mtime: SystemTime,
    size: u64,
    content: String,
    payload: serde_json::Value,
}

#[derive(Clone, Eq)]
struct FileCacheKey {
    path: std::path::PathBuf,
    mode: String,
}

impl PartialEq for FileCacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path && self.mode == other.mode
    }
}

impl Hash for FileCacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.path.hash(state);
        self.mode.hash(state);
    }
}

/// Process-wide file content cache: path -> (mtime, size, formatted content).
/// When a file is read, its formatted output is cached. On subsequent reads of
/// the same unchanged file, the cached content is returned (file memory).
static FILE_READ_CACHE: OnceLock<Mutex<HashMap<FileCacheKey, FileCacheEntry>>> = OnceLock::new();
const FILE_READ_CACHE_MAX: usize = 100;

/// Minimum file size allowed (50KB) - prevents LLM from setting too small a limit
const MIN_FILE_SIZE: u64 = 50 * 1024;

/// Default maximum file size (500KB) - balanced for context windows
const DEFAULT_MAX_SIZE: u64 = 500 * 1024;

/// Absolute maximum file size for direct read (2MB)
const MAX_FILE_SIZE: u64 = 2 * 1024 * 1024;

/// Large file threshold - above this, use smart chunking (100KB)
const LARGE_FILE_THRESHOLD: u64 = 100 * 1024;

/// Auto-outline threshold (30KB). Files larger than this without an explicit
/// `query/start_line/end_line` get a structural outline + call-out instead of a
/// raw full-file read. The downstream chat layer caps tool output around 20KB
/// and post-hoc head/tail truncation throws away most of the file's structure;
/// returning a real outline up-front preserves much more usable signal and
/// avoids forcing a wasted re-read with chunking parameters.
const AUTO_OUTLINE_THRESHOLD: u64 = 30 * 1024;

/// Number of top chunks to return for relevance search
const TOP_CHUNKS: usize = 5;

/// Chunk size in lines for BM25 scoring
const CHUNK_SIZE_LINES: usize = 50;

/// Context lines to include around relevant chunks
const CONTEXT_LINES: usize = 5;

const NAME: &str = "fs_read_internal";
const DESCRIPTION: &str = "Reads file contents with smart handling for large files. Files >30KB without a query/start_line/end_line return a structural outline (symbols + previews) instead of raw content -- pass query=\"...\" for BM25-ranked relevant sections, or start_line/end_line for a specific range, to get full text.";

#[derive(Debug, Default, Clone, Copy)]
struct ReadStageTimings {
    resolve_ms: u64,
    metadata_ms: u64,
    cache_lookup_ms: u64,
    io_ms: u64,
    split_ms: u64,
    process_ms: u64,
    total_ms: u64,
}

#[derive(Debug)]
struct ReadProcessingInput {
    content: String,
    path: String,
    file_size: u64,
    max_size: u64,
    query: Option<String>,
    num_chunks: usize,
    timings: ReadStageTimings,
}

#[derive(Debug)]
struct ReadProcessingResult {
    output: String,
    payload: serde_json::Value,
    timings: ReadStageTimings,
}

/// Clear the file content cache (file memory). Call when starting a new session
/// to avoid serving stale content across unrelated sessions.
pub fn clear_file_memory() {
    if let Some(cache) = FILE_READ_CACHE.get()
        && let Ok(mut guard) = cache.lock()
    {
        guard.clear();
    }
}
#[derive(Debug)]
pub struct ReadFileTool;

impl ToolMeta for ReadFileTool {
    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(DESCRIPTION)
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(NAME)
    }

    fn name(&self) -> &str {
        NAME
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to read (relative or absolute)"
                },
                "query": {
                    "type": "string",
                    "description": "Optional: Search query to find relevant sections in large files. Uses BM25 ranking to return the most relevant chunks."
                },
                "max_size": {
                    "type": "number",
                    "description": "Maximum file size in bytes for direct read (default: 512KB). Files larger than this use smart chunking.",
                    "default": 524_288
                },
                "start_line": {
                    "type": "number",
                    "description": "Optional: Start reading from this line number (1-indexed)."
                },
                "end_line": {
                    "type": "number",
                    "description": "Optional: Stop reading at this line number (inclusive)."
                },
                "offset": {
                    "type": "number",
                    "description": "Optional: 1-indexed first line (alias of start_line). 0 means the start of the file."
                },
                "limit": {
                    "type": "number",
                    "description": "Optional: number of lines to return from start_line or offset."
                },
                "num_chunks": {
                    "type": "number",
                    "description": "Number of relevant chunks to return when using query (default: 5)",
                    "default": 5
                }
            },
            "required": ["path"]
        })
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let total_start = measure_start();
        let mut timings = ReadStageTimings::default();
        let path_str = match params::extract_string(&arguments, "path", None) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(e),
        };

        // Get parameters
        let requested_size = params::extract_number_or(&arguments, "max_size", DEFAULT_MAX_SIZE);
        let max_size = requested_size.clamp(MIN_FILE_SIZE, MAX_FILE_SIZE);
        let query = arguments
            .get("query")
            .and_then(|v| v.as_str())
            .map(String::from);
        let window = match resolve_read_window(&arguments) {
            Ok(window) => window,
            Err(error) => return ToolResult::error(error),
        };
        let start_line = window.map(|window| window.start_line);
        let end_line = window.and_then(|window| window.end_line);
        let num_chunks =
            params::extract_number_or(&arguments, "num_chunks", TOP_CHUNKS as u64) as usize;

        // Validate and resolve path (must be a file)
        let stage_start = measure_start();
        let normalized_path = match path::validate_and_resolve_path(&path_str, Some(true)).await {
            Ok(p) => p,
            Err(e) => return ToolResult::error(e),
        };
        timings.resolve_ms = elapsed_opt(stage_start);

        // Get file metadata
        let stage_start = measure_start();
        let metadata = match std::fs::metadata(&normalized_path) {
            Ok(m) => m,
            Err(e) => return ToolResult::error(format!("Failed to read file metadata: {e}")),
        };
        timings.metadata_ms = elapsed_opt(stage_start);
        let file_size = metadata.len();
        let mtime = metadata.modified().ok();
        let cache_key = read_cache_key(&normalized_path, query.as_deref(), num_chunks);

        // Relevance searches and structural outlines are expensive enough to
        // cache. Line ranges stream from disk and are intentionally uncached.
        let use_cache = start_line.is_none() && end_line.is_none();

        let stage_start = measure_start();
        if let Some(hit) = try_cached_read(
            use_cache,
            mtime.as_ref(),
            &cache_key,
            file_size,
            timings,
            stage_start,
            total_start,
            &path_str,
        ) {
            return hit;
        }
        timings.cache_lookup_ms = elapsed_opt(stage_start);

        if start_line.is_some() || end_line.is_some() {
            return read_line_range_result(
                normalized_path,
                path_str,
                start_line,
                end_line,
                file_size,
                timings,
                total_start,
            )
            .await;
        }

        let path_for_io = normalized_path;
        let path_label = path_str.clone();
        let processed = match tokio::task::spawn_blocking(move || {
            let io_start = measure_start();
            let content = read_whole_file(&path_for_io, &path_label)?;
            let mut timings = timings;
            timings.io_ms = elapsed_opt(io_start);
            process_read_content(ReadProcessingInput {
                content,
                path: path_label,
                file_size,
                max_size,
                query,
                num_chunks,
                timings,
            })
        })
        .await
        {
            Ok(Ok(processed)) => processed,
            Ok(Err(e)) => return ToolResult::error(e.to_string()),
            Err(e) => return ToolResult::error(format!("Read processing task failed: {e}")),
        };
        let mut payload = processed.payload;
        let output = processed.output;
        let mut timings = processed.timings;
        timings.total_ms = elapsed_opt(total_start);
        insert_stage_timings(&mut payload, &timings);

        store_read_cache(use_cache, mtime, cache_key, file_size, &output, &payload);

        trace_read_complete(&path_str, read_mode_from_payload(&payload), timings);
        ToolResult::success_with_payload(output, payload)
    }

    #[cfg(feature = "sandbox")]
    async fn execute_sandboxed(
        &self,
        arguments: serde_json::Value,
        sandbox: Option<&dyn caliche_sandbox::SandboxHandleOps>,
    ) -> ToolResult {
        let Some(sandbox) = sandbox else {
            return self.execute(arguments).await;
        };
        let path_str = match params::extract_string(&arguments, "path", None) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(e),
        };
        let has_query = arguments.get("query").and_then(|v| v.as_str()).is_some();
        match resolve_read_window(&arguments) {
            Err(error) => return ToolResult::error(error),
            Ok(Some(_)) => return self.execute(arguments).await,
            Ok(None) if has_query => return self.execute(arguments).await,
            Ok(None) => {}
        }
        let total_start = measure_start();
        let stage_start = measure_start();
        match sandbox.read_file(&path_str).await {
            Ok(bytes) => {
                let file_size = bytes.len() as u64;
                let requested_size =
                    params::extract_number_or(&arguments, "max_size", DEFAULT_MAX_SIZE);
                let max_size = requested_size.clamp(MIN_FILE_SIZE, MAX_FILE_SIZE);
                let timings = ReadStageTimings {
                    io_ms: elapsed_opt(stage_start),
                    ..ReadStageTimings::default()
                };
                let processing_input = ReadProcessingInput {
                    content: String::from_utf8_lossy(&bytes).into_owned(),
                    path: path_str.clone(),
                    file_size,
                    max_size,
                    query: None,
                    num_chunks: TOP_CHUNKS,
                    timings,
                };
                match tokio::task::spawn_blocking(move || process_read_content(processing_input))
                    .await
                {
                    Ok(Ok(mut processed)) => {
                        processed.timings.total_ms = elapsed_opt(total_start);
                        insert_stage_timings(&mut processed.payload, &processed.timings);
                        trace_read_complete(
                            &path_str,
                            read_mode_from_payload(&processed.payload),
                            processed.timings,
                        );
                        ToolResult::success_with_payload(processed.output, processed.payload)
                    }
                    Ok(Err(e)) => ToolResult::error(e.to_string()),
                    Err(e) => ToolResult::error(format!("Read processing task failed: {e}")),
                }
            }
            Err(e) => ToolResult::error(format!("Sandbox read_file failed: {e}")),
        }
    }
}

fn elapsed_ms(start: Instant) -> u64 {
    start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
}

fn measure_start() -> Option<Instant> {
    crate::observability::perf::enabled().then(Instant::now)
}

fn elapsed_opt(start: Option<Instant>) -> u64 {
    start.map_or(0, elapsed_ms)
}

fn insert_stage_timings(payload: &mut serde_json::Value, timings: &ReadStageTimings) {
    if !crate::observability::perf::enabled() {
        return;
    }
    if let Some(obj) = payload.as_object_mut() {
        obj.insert(
            "stage_timings_ms".to_string(),
            stage_timings_payload(timings),
        );
    }
}

fn stage_timings_payload(timings: &ReadStageTimings) -> serde_json::Value {
    json!({
        "resolve": timings.resolve_ms,
        "metadata": timings.metadata_ms,
        "cache_lookup": timings.cache_lookup_ms,
        "io": timings.io_ms,
        "split": timings.split_ms,
        "process": timings.process_ms,
        "total": timings.total_ms,
    })
}

fn trace_read_complete(path: &str, mode: &str, timings: ReadStageTimings) {
    if !crate::observability::perf::enabled() {
        return;
    }
    tracing::debug!(
        target: "caliche::tools::fs",
        op = "read",
        path = %path,
        mode = %mode,
        resolve_ms = timings.resolve_ms,
        metadata_ms = timings.metadata_ms,
        cache_lookup_ms = timings.cache_lookup_ms,
        io_ms = timings.io_ms,
        split_ms = timings.split_ms,
        process_ms = timings.process_ms,
        total_ms = timings.total_ms,
        "fs.read.complete"
    );
}

fn read_mode_from_payload(payload: &serde_json::Value) -> &str {
    match payload.get("mode").and_then(serde_json::Value::as_str) {
        Some(mode) => mode,
        None => "unknown",
    }
}

fn process_read_content(input: ReadProcessingInput) -> Result<ReadProcessingResult> {
    let mut timings = input.timings;
    let split_start = measure_start();
    let lines: Vec<&str> = input.content.lines().collect();
    let total_lines = lines.len();
    timings.split_ms = elapsed_opt(split_start);

    let process_start = measure_start();
    let (result, mode, outlined) = if let Some(ref q) = input.query {
        // Query provided -- use BM25 relevance ranking on the full file. This
        // works for files of any size; small files just rank fewer chunks.
        (
            relevance_search(
                &lines,
                q,
                input.num_chunks,
                &input.path,
                input.file_size,
                total_lines,
            ),
            "query",
            false,
        )
    } else if input.file_size <= AUTO_OUTLINE_THRESHOLD {
        // Comfortably small -- return entire file. Below the chat layer's
        // tool-output cap, so it survives untouched.
        let tagged = hashline::format_hashlines(&lines, 1);
        (
            Ok(format!(
                "[File: {}, Size: {} bytes, {} lines]\n{}",
                input.path, input.file_size, total_lines, tagged
            )),
            "full",
            false,
        )
    } else if input.file_size <= LARGE_FILE_THRESHOLD || input.file_size <= input.max_size {
        // Medium file (30KB - 100KB, or below the caller's max_size). The
        // chat layer would post-hoc head/tail-truncate this, throwing away
        // the middle. Return a structural outline instead so the model
        // sees the shape of the file and can request specific sections
        // with start_line/end_line or query.
        (
            structural_outline(&lines, &input.path, input.file_size, total_lines),
            "outline",
            true,
        )
    } else {
        // Large file without query -- return structural outline.
        (
            structural_outline(&lines, &input.path, input.file_size, total_lines),
            "outline",
            true,
        )
    };
    timings.process_ms = elapsed_opt(process_start);

    let output = result?;
    let payload = serde_json::json!({
        "op": "read",
        "path": input.path,
        "file_size": input.file_size,
        "total_lines": total_lines,
        "query": input.query,
        "num_chunks": input.num_chunks,
        "outlined": outlined,
        "mode": mode,
        "cache_hit": false,
    });
    Ok(ReadProcessingResult {
        output,
        payload,
        timings,
    })
}

fn try_cached_read(
    use_cache: bool,
    mtime: Option<&SystemTime>,
    cache_key: &FileCacheKey,
    file_size: u64,
    timings: ReadStageTimings,
    stage_start: Option<Instant>,
    total_start: Option<Instant>,
    path_str: &str,
) -> Option<ToolResult> {
    if !use_cache {
        return None;
    }
    let mt = mtime?;
    let cache = FILE_READ_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let guard = cache.lock().ok()?;
    let entry = guard.get(cache_key)?;
    if entry.mtime != *mt || entry.size != file_size {
        return None;
    }
    let mut payload = entry.payload.clone();
    if let Some(obj) = payload.as_object_mut() {
        obj.insert("cache_hit".to_string(), serde_json::Value::Bool(true));
    }
    let timings = ReadStageTimings {
        cache_lookup_ms: elapsed_opt(stage_start),
        total_ms: elapsed_opt(total_start),
        ..timings
    };
    insert_stage_timings(&mut payload, &timings);
    trace_read_complete(path_str, "cache", timings);
    Some(ToolResult::success_with_payload(
        entry.content.clone(),
        payload,
    ))
}

fn store_read_cache(
    use_cache: bool,
    mtime: Option<SystemTime>,
    cache_key: FileCacheKey,
    file_size: u64,
    output: &str,
    payload: &serde_json::Value,
) {
    if !use_cache {
        return;
    }
    let Some(mt) = mtime else {
        return;
    };
    let cache = FILE_READ_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let Ok(mut guard) = cache.lock() else {
        return;
    };
    if guard.len() >= FILE_READ_CACHE_MAX {
        guard.clear();
    }
    guard.insert(
        cache_key,
        FileCacheEntry {
            mtime: mt,
            size: file_size,
            content: output.to_string(),
            payload: payload.clone(),
        },
    );
}

async fn read_line_range_result(
    normalized_path: std::path::PathBuf,
    path_str: String,
    start_line: Option<usize>,
    end_line: Option<usize>,
    file_size: u64,
    mut timings: ReadStageTimings,
    total_start: Option<Instant>,
) -> ToolResult {
    let stage_start = measure_start();
    let path_for_io = normalized_path;
    let display = path_str.clone();
    let result = tokio::task::spawn_blocking(move || {
        read_line_range_streaming(&path_for_io, start_line, end_line, &display, file_size)
    })
    .await;
    timings.io_ms = elapsed_opt(stage_start);
    timings.total_ms = elapsed_opt(total_start);
    match result {
        Ok(Ok((output, mut payload))) => {
            insert_stage_timings(&mut payload, &timings);
            trace_read_complete(&path_str, "line_range", timings);
            ToolResult::success_with_payload(output, payload)
        }
        Ok(Err(e)) => ToolResult::error(e.to_string()),
        Err(e) => ToolResult::error(format!("Read worker failed: {e}")),
    }
}

fn read_cache_key(path: &std::path::Path, query: Option<&str>, num_chunks: usize) -> FileCacheKey {
    let mode = query.map_or_else(
        || "outline-or-full".to_string(),
        |query| format!("query:{query}:chunks:{num_chunks}"),
    );
    FileCacheKey {
        path: path.to_path_buf(),
        mode,
    }
}

/// Read the resolved file `path` whole, following no link.
fn read_whole_file(path: &std::path::Path, label: &str) -> Result<String> {
    path::read_resolved_text(path)
        .map_err(|e| anyhow::anyhow!(errors::format_file_error("read", label, &e.to_string())))
}

fn read_line_range_streaming(
    path: &std::path::Path,
    start_line: Option<usize>,
    end_line: Option<usize>,
    display_path: &str,
    file_size: u64,
) -> Result<(String, serde_json::Value)> {
    let start = start_line.unwrap_or(1).max(1);
    let end_requested = end_line.unwrap_or(usize::MAX).max(start);
    let file = path::open_resolved_file(path)
        .with_context(|| format!("Failed to read file '{display_path}'"))?;
    let mut line_no = 0usize;
    let mut selected = Vec::new();
    let mut reached_end = false;

    for line in BufReader::new(file).lines() {
        let line = line.with_context(|| format!("Failed to read file '{display_path}'"))?;
        line_no = line_no.saturating_add(1);
        if line_no >= start && line_no <= end_requested {
            selected.push(line);
        }
        if line_no >= end_requested {
            reached_end = true;
            break;
        }
    }

    if selected.is_empty() {
        bail!("start_line {start} is beyond end of file ({line_no} lines scanned)");
    }

    let selected_refs: Vec<&str> = selected.iter().map(String::as_str).collect();
    let tagged = hashline::format_hashlines(&selected_refs, start);
    let end = start + selected.len() - 1;
    let total_fragment = if reached_end {
        format!("at least {end}")
    } else {
        line_no.to_string()
    };
    let output = format!(
        "[File: {display_path}, Size: {file_size} bytes]\n[Lines {start}-{end} of {total_fragment} total]\n{tagged}\n\n[Follow-up: use op=read path=\"{display_path}\" start_line=N end_line=M for another slice, or query=\"...\" for ranked sections.]"
    );
    let payload = serde_json::json!({
        "op": "read",
        "path": display_path,
        "file_size": file_size,
        "start_line": start,
        "end_line": end,
        "returned_lines": selected.len(),
        "total_lines_exact": !reached_end,
        "total_lines": if reached_end { serde_json::Value::Null } else { serde_json::json!(line_no) },
        "cache_hit": false,
    });
    Ok((output, payload))
}

/// Use BM25-style relevance ranking to find the most relevant chunks
fn relevance_search(
    lines: &[&str],
    query: &str,
    num_chunks: usize,
    path: &str,
    file_size: u64,
    total_lines: usize,
) -> Result<String> {
    // Tokenize query
    let query_terms: Vec<String> = tokenize(query);
    if query_terms.is_empty() {
        bail!("Query is empty or contains no searchable terms");
    }

    // Build document frequency map for BM25
    let mut doc_freq: HashMap<String, usize> = HashMap::new();
    let chunks = create_chunks(lines, CHUNK_SIZE_LINES);
    let num_docs = chunks.len();

    // Calculate document frequencies
    for chunk in &chunks {
        let chunk_text = chunk.lines.join(" ");
        let terms: std::collections::HashSet<String> = tokenize(&chunk_text).into_iter().collect();
        for term in terms {
            *doc_freq.entry(term).or_insert(0) += 1;
        }
    }

    // Score each chunk using BM25
    let k1 = 1.2;
    let b = 0.75;
    let avg_doc_len: f64 =
        chunks.iter().map(|c| c.lines.len()).sum::<usize>() as f64 / num_docs.max(1) as f64;

    let mut scored_chunks: Vec<(usize, f64, &Chunk)> = chunks
        .iter()
        .enumerate()
        .map(|(idx, chunk)| {
            let chunk_text = chunk.lines.join(" ");
            let chunk_terms = tokenize(&chunk_text);
            let doc_len = chunk_terms.len() as f64;

            let mut score = 0.0;
            for query_term in &query_terms {
                let tf = chunk_terms.iter().filter(|t| *t == query_term).count() as f64;
                let df = doc_freq.get(query_term).copied().unwrap_or(0) as f64;

                if df > 0.0 && tf > 0.0 {
                    let idf = ((num_docs as f64 - df + 0.5) / (df + 0.5)).ln_1p();
                    let tf_component =
                        (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * doc_len / avg_doc_len));
                    score += idf * tf_component;
                }
            }
            (idx, score, chunk)
        })
        .collect();

    // Sort by score descending
    scored_chunks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take top chunks
    let top_chunks: Vec<_> = scored_chunks.into_iter().take(num_chunks).collect();

    if top_chunks.is_empty() || top_chunks[0].1 == 0.0 {
        bail!(
            "No relevant content found for query '{query}'. Try different search terms or use start_line/end_line to read specific sections."
        );
    }

    // Format output with context
    let mut output = format!(
        "[File: {}, Size: {} bytes, {} lines]\n[BM25 search for: \"{}\"]\n[Showing top {} relevant chunks]\n\n",
        path,
        file_size,
        total_lines,
        query,
        top_chunks.len()
    );

    for (rank, (_chunk_idx, score, chunk)) in top_chunks.iter().enumerate() {
        let start_with_context = chunk.start_line.saturating_sub(CONTEXT_LINES);
        let end_with_context = (chunk.end_line + CONTEXT_LINES).min(total_lines);

        output.push_str(&format!(
            "--- Chunk {} (lines {}-{}, score: {:.2}) ---\n",
            rank + 1,
            start_with_context + 1,
            end_with_context,
            score
        ));

        // Include context lines
        for (i, line) in lines[start_with_context..end_with_context]
            .iter()
            .enumerate()
        {
            let line_num = start_with_context + i + 1;
            let in_main_chunk = line_num > chunk.start_line && line_num <= chunk.end_line;
            let marker = if in_main_chunk { ">" } else { " " };
            let hash = hashline::compute_line_hash(line);
            output.push_str(&format!("{marker} {line_num:>5}:{hash}| {line}\n"));
        }
        output.push('\n');
    }

    output.push_str(&format!(
        "[Use start_line/end_line to read specific sections, e.g., start_line={}, end_line={}]",
        top_chunks[0].2.start_line + 1,
        top_chunks[0].2.end_line
    ));

    Ok(output)
}

/// Chunk of text with line range
struct Chunk<'a> {
    lines: Vec<&'a str>,
    start_line: usize,
    end_line: usize,
}

/// Create overlapping chunks from lines
fn create_chunks<'a>(lines: &[&'a str], chunk_size: usize) -> Vec<Chunk<'a>> {
    let mut chunks = Vec::new();
    let overlap = chunk_size / 2;
    let mut start = 0;

    while start < lines.len() {
        let end = (start + chunk_size).min(lines.len());
        chunks.push(Chunk {
            lines: lines[start..end].to_vec(),
            start_line: start,
            end_line: end,
        });

        if end >= lines.len() {
            break;
        }
        start += overlap;
    }

    chunks
}

/// Simple tokenization for BM25
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| s.len() >= 2)
        .map(String::from)
        .collect()
}

#[cfg(test)]
mod auto_outline_tests {
    use super::{
        AUTO_OUTLINE_THRESHOLD, LARGE_FILE_THRESHOLD, ReadProcessingInput, ReadStageTimings,
        process_read_content, structural_outline,
    };
    use anyhow::Result;

    #[test]
    fn auto_outline_threshold_is_below_large_file_threshold() {
        // Outline must trigger BEFORE the post-hoc head/tail truncator clips the
        // middle. If AUTO_OUTLINE_THRESHOLD ever drifts above the medium-file
        // ceiling we lose the structural-outline window entirely.
        const {
            assert!(AUTO_OUTLINE_THRESHOLD < LARGE_FILE_THRESHOLD);
        }
    }

    #[test]
    fn medium_typescript_file_above_threshold_renders_outline_header() {
        // Simulate a ~32 KB TypeScript file with realistic symbol density (~30
        // exported functions, each padded with body content) -- mirrors the
        // round-4 regression on `src/language/jinja/jinja.ts` where a sandboxed
        // read returned head+tail-truncated raw content instead of an outline.
        let mut buf = String::with_capacity(40 * 1024);
        buf.push_str("import { foo } from \"bar\";\n\n");
        for i in 0..30 {
            buf.push_str(&format!(
                "export function gen_{i}(input: string): string {{\n"
            ));
            for _ in 0..30 {
                buf.push_str("    // body line filler to grow the file realistically\n");
            }
            buf.push_str(&format!("    return input + \"_{i}\";\n}}\n\n"));
        }
        let lines: Vec<&str> = buf.lines().collect();
        let total_lines = lines.len();
        let file_size = buf.len() as u64;
        assert!(
            file_size > AUTO_OUTLINE_THRESHOLD,
            "test fixture ({} bytes) must exceed AUTO_OUTLINE_THRESHOLD ({})",
            file_size,
            AUTO_OUTLINE_THRESHOLD
        );

        let out = structural_outline(&lines, "fixture.ts", file_size, total_lines)
            .expect("structural_outline should succeed for a medium file");

        assert!(
            out.contains("[File too large for full read - showing structure]"),
            "outline must include the truncation banner so the agent knows to request a slice; got: {}",
            out.chars().take(200).collect::<String>()
        );
        // For realistic symbol density the outline should be meaningfully
        // smaller than the source -- otherwise the structural summary is no win
        // over the chat-layer head/tail truncation we are trying to replace.
        assert!(
            (out.len() as u64) < file_size,
            "outline ({} bytes) should be smaller than raw file ({} bytes)",
            out.len(),
            file_size
        );
    }

    #[test]
    fn processing_result_includes_mode_and_stage_timings() -> Result<()> {
        let content = "alpha target\nbeta filler\ngamma target\n".repeat(20);
        let result = process_read_content(ReadProcessingInput {
            file_size: content.len() as u64,
            content,
            path: "fixture.txt".to_string(),
            max_size: LARGE_FILE_THRESHOLD,
            query: Some("target".to_string()),
            num_chunks: 1,
            timings: ReadStageTimings {
                io_ms: 7,
                ..ReadStageTimings::default()
            },
        })?;

        assert_eq!(result.payload["mode"], "query");
        assert_eq!(result.payload["cache_hit"], false);
        assert!(result.output.contains("[BM25 search for: \"target\"]"));
        assert_eq!(result.timings.io_ms, 7);
        Ok(())
    }
}

#[cfg(test)]
#[cfg(unix)]
mod link_tests {
    use super::{read_line_range_streaming, read_whole_file};

    /// A workspace file `f.txt` that is a link to a file outside, as when it
    /// is swapped for a link after the gate checked its canonical path.
    fn swapped_file() -> (tempfile::TempDir, tempfile::TempDir, std::path::PathBuf) {
        let outside = tempfile::tempdir().expect("outside dir");
        std::fs::write(outside.path().join("secret.txt"), "secret line\n").expect("secret");
        let ws = tempfile::tempdir().expect("workspace");
        let root = ws.path().canonicalize().expect("canonical workspace");
        std::os::unix::fs::symlink(outside.path().join("secret.txt"), root.join("f.txt"))
            .expect("file link");
        let file = root.join("f.txt");
        (outside, ws, file)
    }

    #[test]
    fn a_whole_read_refuses_a_file_swapped_for_a_link() {
        let (_outside, _ws, file) = swapped_file();
        let err = read_whole_file(&file, "f.txt").expect_err("a link must not be read");
        assert!(!err.to_string().contains("secret line"), "{err}");
    }

    #[test]
    fn a_line_range_read_refuses_a_file_swapped_for_a_link() {
        let (_outside, _ws, file) = swapped_file();
        let err = read_line_range_streaming(&file, Some(1), Some(1), "f.txt", 12)
            .expect_err("a link must not be read");
        assert_eq!(err.to_string(), "Failed to read file 'f.txt'");
    }
}

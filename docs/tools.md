# Tools

An [agent](concepts.md#agent) runs tools on behalf of the model. This page describes the default tool set, how `web_fetch` and `web_search` choose a backend, image generation, and the rule that stops the agent from repeating an identical failed call.

## Default tools

`enable_all_default_tools()` and `use_all_default_tools()` register three tools:

- `fs` lists, globs, searches, and reads files.
- `bash` runs a shell command.
- `web_fetch` reads a public URL as text.

`enable_extended_default_tools()` registers the default tools and adds `http`, `workspace`, and `execute_python`.

The following tools are off until you enable them: `web_search`, semantic search, Perplexity search, the language server tools, the AST tools, and the graph tools.

In the CLI, the `coding` and `default` presets of `--tools` include `web_fetch`, and the `readonly` and `none` presets do not. `--profile lean` limits the tools to `fs` and `bash`. See [Configuration](configuration.md#lean-profile).

## web_fetch

`web_fetch` returns the text of a public `http` or `https` URL. It does not run JavaScript on the page.

It tries the following backends in order:

1. TinyFish, when `TINYFISH_API_KEY` is set.
2. Firecrawl, when `FIRECRAWL_API_KEY` is set.
3. A plain HTTP request.

When neither key is set, only the plain HTTP request runs, and the URL is not sent to Firecrawl or TinyFish. The variable names are listed in [Environment variables](reference/environment-variables.md#web-search-and-fetch).

## web_search

`web_search` is off by default. When enabled, it returns ranked URLs with snippets.

It tries the following backends in order:

1. Tavily, when `TAVILY_API_KEY` is set.
2. Exa, when `EXA_API_KEY` is set.
3. TinyFish, when `TINYFISH_API_KEY` is set.
4. Firecrawl.

Firecrawl search runs even without `FIRECRAWL_API_KEY`; without a key, Firecrawl limits requests per IP address.

`perplexity_search` is a separate tool that returns a synthesized answer with citations. Enable it when you need citations.

```rust
let builder = AgentBuilder::new()
    .enable_tools(&["web_search", "web_fetch", "perplexity_search", "http"]);
```

## Image generation

When `CALICHE_IMAGE_PROVIDER` is `openai` or `google`, the agent exposes a `generate_image` tool. Generated images are written to `.caliche/artifacts/images`, unless `CALICHE_ARTIFACT_DIR` or the `/artifact-dir` REPL command selects another directory. See [Environment variables](reference/environment-variables.md#image-generation).

## Identical failures

Within one user turn, the agent runs a tool call the first time even if it fails, and the error output (stderr) of that failure stays in the context.

If the model then repeats the same call with the same arguments, the agent does not run it. The tool result is `retry_blocked_until_code_change` followed by a short hash of the earlier stderr; the earlier stderr itself is not repeated. A call with different arguments still runs.

The record of a failure is cleared when any of the following happens:

- A successful `bash` command that writes files runs in the working directory of the failure. Examples are `sed -i`, a redirect such as `cat >`, and a Python `open` for writing.
- An `fs` write, edit, or delete, or a `write_file`, `edit_file`, or `apply_patch` call, touches a path that the failure named.
- The modification time or content hash of a named path changes.

When a call omits `working_directory`, the process working directory is used, so two calls that both omit it share a directory. A bare relative name such as `main.py` counts as a path.

Timeouts are not recorded as failures, and neither are HTTP 408, 429, and 5xx responses from `web_fetch`, `http`, and `web_search`.

To turn this rule off, set `CALICHE_IDENTICAL_FAILURE_BREAKER` to `0`, `off`, `false`, `no`, or `disabled`.

The tool executor writes one JSON object to the tracing target `caliche::tool_outcome` for each tool start, tool end, and approval pause. In the REPL, this target is written to a `repl-*.log` file with file mode `0600`. Each line contains the full `action_id`, the tool name, `ok`, the `bash` exit code, the working directory, and the stderr hash. Logged stderr passes through the default transcript redactor and is truncated to 160 characters. To silence these lines, set `RUST_LOG=caliche::tool_outcome=off`. When the REPL sends its turns to an HTTP server, it prints the last 8 hexadecimal digits of the action id.

## Learning loop

The learning loop turns a sequence of tool calls into a skill. It can describe that sequence in several ways, from narrow to general:

- `prefer_weak_skills`, on by default, makes the loop choose the most general description that still fits the sequence, instead of the most specific one.
- `require_holdout_pass`, on by default, makes the loop learn from the first part of the sequence and test the skill on the tools of the rest. The loop creates or changes the skill only when the skill covers at least the `min_holdout_extent` share of those tools, which is `1.0` by default.

The loop also suggests changes to the harness. `rank_by_weakness` ranks them: a suggestion scores higher when it covers more kinds of failure with fewer specific conditions, and lower when it risks breaking something that works.

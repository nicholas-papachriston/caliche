# Caliche Style Guide

## Purpose

This guide defines the current visual and interaction language for Caliche across two primary surfaces:

- the embedded/web UI in `caliche_web`
- the interactive terminal REPL in `caliche_cli`

Caliche is a monospace operator console for chat, orchestration, debugging, files, and runtime inspection.

## Design intent

The UI and the REPL use the same rules:

- dark by default
- monospace text
- a small palette for state
- compact layout
- chat and tool activity stay on screen without extra decoration

## Product Surfaces

### Web UI

The web UI is a Dioxus app with a single shell and tabbed work areas.

- header for provider, model, tools, and mode
- horizontal tab bar for major work areas
- chat as the primary interaction surface
- optional session inspector beside chat
- specialized tabs for orchestration, reasoning, documents, plugins, metrics, and settings

### REPL

The REPL is a terminal-first experience that mirrors the same product values in ANSI text form.

- concise startup banner
- provider/model-aware prompt
- explicit interaction mode indicator
- streamed output with minimal framing
- structured tool-call rendering
- slash-command-heavy workflow

## Brand Principles

### 1. Operational Clarity

Every surface should tell the user what system they are talking to, what mode they are in, and what is happening right now.

### 2. Monospace Discipline

Character-grid alignment is part of the product identity. Prefer `ch`-based spacing and monospace rhythm over arbitrary visual decoration.

### 3. Restraint

Accent color should feel valuable because it is scarce. Most surfaces should stay neutral and let activity, errors, and mode changes stand out.

### 4. Shared Semantics

The web UI and REPL should use the same mental model for status, mode, tools, and message roles even though one is graphical and one is terminal-native.

## Color System

Canonical web UI tokens live in `caliche_web/src/styles/mod.rs`.

### Core Tokens

- `--background`: `#0a0a0a`
- `--foreground`: `#e0e0e0`
- `--muted`: `#666666`
- `--border`: `#333333`
- `--accent`: `#00ff41`
- `--accent-dim`: `#00aa2a`
- `--warning`: `#ffaa00`
- `--error`: `#ff4444`
- `--info`: `#4488ff`

### Semantic Meaning

- green is success, active state, assistant identity, and current selection
- blue is user identity and informational emphasis
- yellow is in-progress, warning, and streaming/transitional state
- red is failure, disconnection, destructive state, or blocked execution
- muted gray is metadata, separators, inactive controls, and secondary chrome

### Usage Rules

- do not invent extra accent colors for new states unless the existing palette is insufficient
- do not tint large surfaces brightly; keep most of the interface near `--background`, `--card-bg`, and `--secondary-bg`
- use border and text color before using filled backgrounds

## Typography

### Primary Font

Use `--font-mono` everywhere by default:

- `Fira Code`
- `Fira Mono`
- `SF Mono`
- `Monaco`
- fallback monospace stack

### Tone

- labels and tab names should be short and direct
- prose can remain sentence case
- code, paths, ids, commands, and token stats should stay literal
- avoid marketing language in product chrome

## Spacing And Shape

### Base Rhythm

The web UI is character-grid inspired.

- prefer `1ch`, `2ch`, and `4ch` spacing steps
- treat borders and dashed separators as part of the layout rhythm
- avoid soft rounded-card aesthetics; the product is mostly square-edged and terminal-like

### Borders

- standard structure uses `1px solid var(--border-color)`
- chat rows use dashed separators
- active tab and strong selection states use accent borders
- tool-call blocks use a stronger left border to encode status

## Motion

Motion should stay lightweight and informative.

- short fades are acceptable for new messages
- the chat tab may pulse a streaming mark while a turn is in flight
- the header does not show a connection chip; an unreachable API uses the disconnected screen
- streaming cursor blink is acceptable
- tactile button click feedback: `scale(0.96)` with 150ms `ease-out`
- respect `prefers-reduced-motion`
- no motion should be required to understand state

## Web UI Patterns

### Shell

The UI shell is:

1. header
2. tab bar
3. optional chat status row
4. active work area
5. optional session inspector

Do not introduce competing chrome systems inside tabs.

### Header

The header is operational chrome, not marketing chrome.

- show provider/model/tools clearly
- keep mode switching close to the top-level identity
- preserve the compact terminal-console feel
- do not show a connected or disconnected chip; connection failure stays on the disconnected screen

### Tab Bar

Tabs are flat, textual, and grouped by thin dividers.

Current groups in `caliche_web`:

- `chat`, `history`
- `fleet` (fleet, orchestration, workflow)
- `inspect` (metrics, reasoning, documents, chains)
- `plugins`, `settings`

The fleet pane lists Caliche endpoints, external CLI harnesses, and local tasks in one coordination surface. Every row is a worker in this process. Broadcast can select running endpoints, registered harnesses, and active tasks together. Dispatch output stays on the harness row.

Rules:

- active tab uses accent text and accent bottom border
- inactive tabs remain muted
- avoid icon-heavy tab labels
- keep labels lowercase to match the current shell tone

### Chat

Chat is the primary UX surface and should stay the most polished path.

Rules:

- user messages get a subtle blue-tinted background
- assistant messages stay visually lighter and cleaner
- every message shows role and timestamp
- streaming should feel live but not flashy
- token usage stays quiet and secondary
- tool calls are attached inline to the assistant turn, not broken into separate conversation items

### Message Rendering

Markdown rendering should remain terminal-flavored.

- headings use accent color
- headings prepend prompt-like prefixes such as `>` and `>>`
- links use accent with underline
- blockquotes use border-left framing
- code blocks use `--secondary-bg` plus border
- tables should remain readable in constrained widths

### Input And Composer

Inputs should feel like terminal controls, not browser defaults.

- dark fill
- visible focus border in accent green
- square corners
- monospace text
- file/context/media attachments appear as compact pre-send context, not as separate heavy panels

### File Browser And Context

The file browser is an adjunct to chat, not its own product mode.

- keep it collapsible
- treat attachments and `@` references as supporting context
- never let file-picking chrome overpower the message thread

### Status And Session Inspector

The status row and session inspector are secondary operational surfaces.

- status should help orient the user to workspace/session state
- inspector is opt-in and should not compete with the main thread
- inspector affordance should read as a utility toggle, not a primary action

### Fleet And Operational Tabs

Operational tabs should reuse the same tokens and density as chat.

- compact panels
- framed sections
- small status text
- counts and metrics in muted text with selective accent emphasis
- inline endpoint chat should still feel related to the main chat product
- external CLI harnesses use the same worker row, status colors, and form chrome as endpoints

### Connecting And Empty States

Disconnected and empty surfaces should be quiet, clear, and actionable.

- one strong state indicator
- short explanatory text
- no decorative filler
- optional muted line may mention slash commands that work after connect (e.g. `/help`, `/chat`, `/sessions`, `/settings`, `/documents`); keep claims aligned with what the embedded UI actually implements

## REPL Patterns

### Startup Banner

The REPL banner should establish context quickly:

- product name
- provider and model
- tool availability
- session id
- optional indexing or startup notes
- one concise hint line for `/help`, `Tab`, `Ctrl+C`, and multiline input

### Prompt

The prompt is the REPL's primary identity element.

Format:

- dim `[provider/model]`
- one mode glyph
- optional dim `[agent: name]`

Mode glyphs:

- `>` for agent mode in green
- `?` for ask mode in blue
- `#` for plan mode in yellow

### Multiline Input

- trailing `\` continues input
- continuation prompt is dim `...`
- keep multiline behavior predictable and low-ceremony

### Streaming

- show a dim `Thinking...` indicator before first output
- clear it on first real event
- stream raw text with minimal interruption
- do not over-frame normal assistant output

### Tool Rendering

Tool activity is one of the REPL's most important structured surfaces.

Rules:

- group concurrent calls into numbered rounds
- each tool call gets a small badge and ordinal
- summarize arguments instead of dumping full JSON
- show compact completion status and preview
- use color to distinguish `scan`, `read`, `patch`, `write`, `remove`, `exec`, and `fetch` style work

### Status Lines

Status lines are dim and intentionally lightweight.

Use them for:

- phase transitions
- detected skills
- checkpoint creation
- file writes
- rollback notices
- cancellation and timeout guidance
- auto-compact notices

Do not style status lines like success banners unless they are true confirmations.

### Token Footer

At end of turn, render token usage as a quiet footer.

- prompt/completion/embedding/total where available
- context-window utilization where available
- dim horizontal rule after the stats

### Slash Commands

Slash commands are product chrome in text form.

- keep help grouped and scannable
- prefer short command names with predictable nouns and verbs
- completions should feel immediate
- errors should be direct and actionable

## Content Style

### UI Copy

- short
- literal
- operational
- avoid fluff

### Status Copy

- present tense when work is in progress
- past tense for confirmed completion
- mention the concrete object when possible

### Error Copy

- state what failed
- include the next useful action when obvious
- avoid vague apology language

## Accessibility

- maintain high contrast on dark surfaces
- preserve visible keyboard focus for inputs and controls
- support reduced motion
- keep touch targets usable on coarse pointers
- do not rely on color alone when text or framing can help
- ensure mobile layouts keep core chat actions reachable

## Implementation Map

### Web UI Source Of Truth

- `caliche_web/src/styles/mod.rs`

Responsibilities:

- global tokens
- base typography
- message styling
- markdown styling
- tabs
- buttons
- inputs
- streaming mark
- responsive behavior

### Web UI Shell

- `caliche_web/src/components/app_shell.rs`
- `caliche_web/src/components/nav.rs`
- `caliche_web/src/components/chrome.rs`

Responsibilities:

- app shell
- tab structure
- connection states
- session inspector layout
- chat turn locking

### Web UI Core Components

- `caliche_web/src/components/chat_view.rs`
- `caliche_web/src/components/message.rs`
- `caliche_web/src/components/message_list.rs`
- `caliche_web/src/components/chat_input/mod.rs`
- `caliche_web/src/components/tool_call.rs`
- `caliche_web/src/components/file_browser.rs`
- `caliche_web/src/components/session_panel.rs`
- `caliche_web/src/components/orchestration_view.rs`
- `caliche_web/src/harness/catalog.rs`

### REPL Source Of Truth

- `caliche_cli/src/repl/mod.rs`
- `caliche_cli/src/repl/engine.rs`
- `caliche_cli/src/repl/renderer/mod.rs`
- `caliche_cli/src/repl/io.rs`

Responsibilities:

- startup banner
- prompt construction
- multiline input
- streaming output
- tool-call rendering
- status lines
- help and command presentation

## Non-Goals

Caliche should not drift toward:

- glossy SaaS dashboard styling
- large-radius card systems
- decorative gradients as primary identity
- dense iconography replacing clear text labels
- divergent visual languages between web UI and REPL

## Future Rules

When adding or changing UI:

1. Start with the existing tokens and shell patterns before inventing new chrome.
2. Keep the web UI and REPL semantically aligned for mode, status, and tool execution.
3. Prefer monospace rhythm, border hierarchy, and text labels over decorative surfaces.
4. Add or update shared tokens in `caliche_web/src/styles/mod.rs` before repeating raw literals.
5. Keep chat, tools, and session state readable under streaming conditions.
6. Treat `caliche_web/assets/main.css` as fallback/legacy styling, not the primary design source.
7. If a new pattern materially changes runtime flow or visible product behavior, update this guide alongside the implementation.

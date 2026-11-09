//! Messages, media, tools, status, prose, and streaming.

pub const MESSAGES: &str = r#"
/* Message styling */
.message {
    padding: var(--spacing-1) 0;
    border-bottom: 1px dashed var(--border-color);
    animation: fadeIn 0.12s ease-out;
}

.message:last-child {
    border-bottom: none;
}

/* Subtle background tinting to visually distinguish user vs agent messages */
.message.user-msg {
    background: rgba(68, 136, 255, 0.05);
    margin: 0 -1ch;
    padding: 0.75em 1ch;
    border-bottom: 1px dashed var(--border-color);
}

.message.assistant-msg {
    background: transparent;
}

.message-header {
    display: flex;
    gap: var(--spacing-2);
    color: var(--muted-color);
    margin-bottom: 0.5em;
    align-items: center;
}

.message-role {
    font-weight: bold;
}

.message-role.user {
    color: var(--info);
}

.message-role.assistant {
    color: var(--accent);
}

.message-time {
    color: var(--muted-color);
}

.message-actions {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 1ch;
}

.message-content {
    white-space: pre-wrap;
    word-wrap: break-word;
}

/* Code blocks */
pre, code {
    font-family: var(--font-mono);
    background: var(--secondary-bg);
}

pre {
    padding: var(--spacing-1);
    overflow-x: auto;
    border: 1px solid var(--border-color);
}

code {
    padding: 0 0.5ch;
}
"#;

pub const EXTENDED: &str = r#"
/* Media attachments */
.media-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5ch;
    padding: 0.5em 1ch;
    border-bottom: 1px dashed var(--border-color);
    background: var(--secondary-bg);
}

.media-thumb-wrap {
    position: relative;
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    gap: 0.25em;
    font-size: var(--font-size-xs);
}

.media-thumb {
    width: 5ch;
    height: 5ch;
    object-fit: cover;
    border: 1px solid var(--border-color);
    display: block;
}

.media-icon {
    width: 5ch;
    height: 5ch;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid var(--border-color);
    background: var(--card-bg);
    color: var(--muted-color);
    font-size: var(--font-size-xs);
}

.media-remove {
    position: absolute;
    top: -0.5em;
    right: -0.5em;
    background: var(--error);
    color: #fff;
    border: none;
    border-radius: 50%;
    width: 1.4em;
    height: 1.4em;
    padding: 0;
    cursor: pointer;
    font-size: 0.65rem;
    display: flex;
    align-items: center;
    justify-content: center;
    line-height: 1;
    min-height: auto;
    min-width: auto;
}

.media-label {
    color: var(--muted-color);
    max-width: 6ch;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* Inline image in message */
.msg-media-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5ch;
    margin-bottom: 0.5em;
}

.msg-media-img {
    max-width: min(20ch, 100%);
    max-height: 15ch;
    object-fit: contain;
    border: 1px solid var(--border-color);
    display: block;
}

.msg-media-file {
    display: inline-flex;
    align-items: center;
    gap: 0.5ch;
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    padding: 0.25em 0.75ch;
    font-size: var(--font-size-xs);
    color: var(--muted-color);
}

/* Mode toggle buttons */
.mode-btn {
    padding: 0.2em 0.75ch;
    font-size: var(--font-size-xs);
    font-family: var(--font-mono);
    border: 1px solid var(--border-color);
    background: transparent;
    color: var(--muted-color);
    cursor: pointer;
    transition: color 0.1s, border-color 0.1s, background 0.1s;
    min-height: auto;
    min-width: auto;
}

.mode-btn.active {
    background: var(--accent-dim);
    color: #000;
    border-color: var(--accent);
}

/* Running animation for tool call indicators */
@keyframes running {
    0%   { content: "[.  ]"; }
    25%  { content: "[.. ]"; }
    50%  { content: "[...]"; }
    75%  { content: "[ ..]"; }
    100% { content: "[  .]"; }
}

.tool-running-indicator::after {
    content: "[...]";
    animation: blink 1s step-end infinite;
    color: var(--warning);
}

/* Tool calls - collapsible section */
.tool-calls {
    border-top: 1px dashed var(--border-color);
    margin-top: var(--spacing-1);
    padding-top: var(--spacing-1);
}

.tool-call-item {
    border-left: 2px solid var(--border-color);
    padding-left: var(--spacing-1);
    margin: 0.5em 0;
}

.tool-call-item.running {
    border-left-color: var(--warning-color);
}

.tool-call-item.success {
    border-left-color: var(--success-color);
}

.tool-call-item.error {
    border-left-color: var(--error-color);
}

/* Animations */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
}

/* Cursor blink for streaming */
.streaming-cursor::after {
    content: "█";
    animation: blink 1s step-end infinite;
    color: var(--accent);
}

/* Prose/Markdown content */
.prose {
    line-height: var(--lh);
}

.prose h1, .prose h2, .prose h3, .prose h4, .prose h5, .prose h6 {
    margin-top: 1em;
    margin-bottom: 0.5em;
    font-weight: bold;
    color: var(--accent);
}

.prose h1::before { content: "> "; color: var(--muted-color); }
.prose h2::before { content: ">> "; color: var(--muted-color); }
.prose h3::before { content: ">>> "; color: var(--muted-color); }
.prose h4::before { content: ">>>> "; color: var(--muted-color); }
.prose h5::before { content: ">>>>> "; color: var(--muted-color); }
.prose h6::before { content: ">>>>>> "; color: var(--muted-color); }

.prose p {
    margin-bottom: 1em;
}

.prose ul, .prose ol {
    margin-left: 2ch;
    margin-bottom: 1em;
}

.prose li {
    margin-bottom: 0.25em;
}

.prose ul li::marker {
    content: "- ";
}

.prose blockquote {
    border-left: 2px solid var(--border-color);
    padding-left: 2ch;
    color: var(--muted-color);
    margin: 1em 0;
}

.prose a {
    color: var(--accent);
    text-decoration: underline;
}

.prose a:hover {
    color: var(--foreground);
}

.prose strong {
    font-weight: bold;
    color: var(--foreground);
}

.prose em {
    font-style: italic;
}
"#;

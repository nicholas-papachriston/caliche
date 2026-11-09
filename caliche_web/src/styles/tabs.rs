//! Tab bar, session panel, connecting screen, and empty states.

pub const STYLES: &str = r#"
/* Tab bar */
.tab-bar {
    display: flex;
    background: var(--card-bg);
    border-bottom: 1px solid var(--border-color);
    flex-shrink: 0;
    overflow-x: auto;
    scrollbar-width: none;
    align-items: stretch;
    touch-action: pan-x;
}

.tab-bar::-webkit-scrollbar {
    display: none;
}

/* Visual divider between tab groups */
.tab-divider {
    width: 1px;
    background: var(--border-color);
    margin: 0.4em 0.5ch;
    flex-shrink: 0;
    align-self: stretch;
}

.tab-btn {
    padding: 0.4em 2ch;
    background: transparent;
    color: var(--muted-color);
    border: none;
    border-bottom: 2px solid transparent;
    cursor: pointer;
    font-size: var(--font-size-sm);
    font-family: var(--font-mono);
    min-height: auto;
    min-width: auto;
    margin-bottom: -1px;
    transition: color 0.1s, border-color 0.1s;
}

.tab-btn.active {
    color: var(--accent);
    border-bottom-color: var(--accent);
    background: transparent;
}

.pane-bar {
    display: flex;
    background: var(--background);
    border-bottom: 1px solid var(--border-color);
    flex-shrink: 0;
    overflow-x: auto;
    scrollbar-width: none;
    align-items: stretch;
    touch-action: pan-x;
}

.pane-bar::-webkit-scrollbar {
    display: none;
}

.pane-btn {
    padding: 0.3em 1.5ch;
    background: transparent;
    color: var(--muted-color);
    border: none;
    border-bottom: 2px solid transparent;
    cursor: pointer;
    font-size: var(--font-size-xs);
    font-family: var(--font-mono);
    min-height: auto;
    min-width: auto;
    margin-bottom: -1px;
    letter-spacing: 0.04em;
}

.pane-btn.active {
    color: var(--text-color);
    border-bottom-color: var(--text-color);
}

.pane-btn:active:not(:disabled) {
    color: var(--text-color);
}

.section-header {
    display: flex;
    align-items: center;
    gap: 1ch;
    padding: 0.5em 1ch;
    background: var(--card-bg);
    flex-shrink: 0;
    flex-wrap: wrap;
}

.section-title {
    color: var(--accent);
    font-size: var(--font-size-xs);
    letter-spacing: 0.1em;
}

.section-summary {
    color: var(--muted-color);
    font-size: var(--font-size-xs);
    flex: 1;
}

.section-actions {
    display: flex;
    gap: 0.5ch;
    align-items: center;
}

.field-row {
    display: flex;
    flex-direction: column;
    gap: 0.25em;
    margin-bottom: 0.7em;
}

.field-row-head {
    display: flex;
    justify-content: space-between;
    gap: 1ch;
    align-items: baseline;
}

.field-row-label {
    color: var(--text-color);
    font-size: var(--font-size-xs);
}

.field-row-help {
    color: var(--muted-color);
    font-size: var(--font-size-xs);
}

.preset-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(18ch, 1fr));
    gap: 0.5ch;
    margin-bottom: 0.75em;
}

.preset-chip {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.2em;
    text-align: left;
    padding: 0.45em 1ch;
    background: var(--secondary-bg);
    border: 1px solid var(--border-color);
    color: var(--text-color);
    min-height: 44px;
}

.preset-chip.active {
    border-color: var(--accent);
    color: var(--accent);
}

.preset-chip-label {
    font-size: var(--font-size-sm);
}

.preset-chip-help {
    font-size: var(--font-size-xs);
    color: var(--muted-color);
    line-height: 1.3;
}

.preset-chip.active .preset-chip-help {
    color: var(--text-color);
}

.harness-form {
    padding: 0.75em 1ch;
    background: var(--card-bg);
    border-bottom: 1px solid var(--border-color);
    flex-shrink: 0;
}

.form-actions {
    display: flex;
    gap: 1ch;
    align-items: center;
}

.form-error {
    padding: 0.4em 0;
    font-size: var(--font-size-xs);
    color: var(--error);
}

.workers-section {
    flex-shrink: 0;
    border-bottom: 2px solid var(--border-color);
    max-height: 45vh;
    display: flex;
    flex-direction: column;
}

.workers-section.grow {
    flex: 1;
    min-height: 0;
    max-height: none;
    border-bottom: none;
}

.harnesses-section {
    flex-shrink: 0;
    border-bottom: 2px solid var(--border-color);
    max-height: 40vh;
    display: flex;
    flex-direction: column;
}

.broadcast-panel {
    flex-shrink: 0;
    border-bottom: 2px solid var(--accent);
    background: var(--card-bg);
}

.chip-row {
    display: flex;
    gap: 0.6ch;
    flex-wrap: wrap;
    flex-shrink: 0;
}

.chip {
    padding: 0.3em 0.8ch;
    font-size: var(--font-size-xs);
    background: transparent;
    border: 1px solid var(--border-color);
    color: var(--text-color);
    cursor: pointer;
}

.chip.active {
    border-color: var(--accent);
    color: var(--background);
    background: var(--accent);
}

.hint-banner {
    color: var(--muted-color);
    font-size: var(--font-size-xs);
    background: var(--secondary-bg);
    padding: 0.6ch 1ch;
    border: 1px solid var(--border-color);
    flex-shrink: 0;
}

.orch-page {
    display: flex;
    flex-direction: column;
    height: 100%;
    overflow: hidden;
    padding: 1.5ch;
    gap: 1.2ch;
    box-sizing: border-box;
}

.orch-scroll {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 1ch;
    min-height: 0;
}

.agent-grid {
    display: grid;
    grid-template-columns: 8ch 1fr 1fr 1fr 1fr 3ch;
    gap: 0.4ch;
    padding: 0.3ch 0.5ch;
    border: 1px solid var(--border-color);
}

.agent-grid.header {
    background: var(--secondary-bg);
}

.result-block {
    border: 1px solid var(--border-color);
    background: var(--secondary-bg);
    padding: 0.8ch 1ch;
}

.result-block.final {
    border-color: var(--accent);
}

.result-pre {
    margin: 0;
    color: var(--text-color);
    font-size: var(--font-size-xs);
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 12ch;
    overflow-y: auto;
}

.worker-list {
    overflow-y: auto;
    flex: 1;
}

.worker-row {
    border-bottom: 1px solid var(--border-color);
}

.worker-row-main {
    display: flex;
    align-items: center;
    gap: 0.75ch;
    padding: 0.4em 1ch;
    flex-wrap: wrap;
}

.worker-row-dispatch {
    display: flex;
    align-items: center;
    gap: 0.5ch;
    padding: 0 1ch 0.45em 1ch;
}

.worker-name {
    color: var(--text-color);
}

.worker-kind {
    color: var(--accent);
    font-size: var(--font-size-xs);
}

.worker-meta {
    color: var(--muted-color);
    font-size: var(--font-size-xs);
}

.worker-ready {
    color: var(--accent);
    font-size: var(--font-size-xs);
}

.worker-error {
    color: var(--error);
    font-size: var(--font-size-xs);
}

.worker-output {
    margin: 0;
    padding: 0 1ch 0.45em 3ch;
    color: var(--text-color);
    font-size: var(--font-size-xs);
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 12ch;
    overflow-y: auto;
}

.empty-state-body {
    margin-bottom: 0.6em;
}

/* Send button — primary action emphasis */
.send-btn {
    padding: 0.25em 2.5ch;
    letter-spacing: 0.05em;
    font-weight: bold;
}

/* Chat input container streaming state */
.chat-input-streaming {
    border-color: var(--warning) !important;
}

/* Completions dropdown */
.completions-dropdown {
    border-bottom: 2px solid var(--accent);
}

/* History rows - clickable affordance */

/* Session panel toggle button */
.session-panel-btn {
    background: var(--secondary-bg);
    border-left: 1px solid var(--border-color);
    border-top: none;
    border-right: none;
    border-bottom: none;
    color: var(--muted-color);
    padding: 0 1.5ch;
    font-size: var(--font-size-xs);
    cursor: pointer;
    flex-shrink: 0;
    transition: color 0.1s, background 0.1s;
    white-space: nowrap;
}

.session-panel-btn.active {
    color: var(--accent);
    border-left-color: var(--accent);
}

/* Connecting screen */
.connecting-screen {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--muted-color);
}

.connecting-screen-box {
    border: 1px solid var(--border-color);
    padding: 2em 3ch;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1em;
    text-align: center;
    max-width: 40ch;
}

/* Streaming badge in tab */
.tab-streaming-dot {
    display: inline-block;
    width: 0.5ch;
    height: 0.5ch;
    border-radius: 50%;
    background: var(--warning);
    margin-left: 0.5ch;
    animation: pulse 1s infinite;
    vertical-align: middle;
    flex-shrink: 0;
}

/* Empty state */
.empty-state {
    color: var(--muted-color);
    text-align: center;
    padding: 4ch;
}

.empty-state-hint {
    cursor: pointer;
    padding: 0.3em 0.5ch;
    border: 1px solid transparent;
    transition: border-color 0.1s, color 0.1s, background 0.1s;
}

@media (hover: hover) and (pointer: fine) {
    .tab-btn:hover:not(:disabled) {
        color: var(--text-color);
        background: transparent;
        border-bottom-color: var(--border-color);
    }
    .pane-btn:hover:not(:disabled) {
        color: var(--text-color);
    }
    .preset-chip:hover:not(:disabled) {
        border-color: var(--muted-color);
    }
    .chip:hover:not(:disabled):not(.active) {
        border-color: var(--muted-color);
        color: var(--text-color);
    }
    .completions-dropdown > div:hover {
        background: var(--secondary-bg);
    }
    .history-row-btn:hover {
        background: var(--secondary-bg) !important;
    }
    .session-panel-btn:hover:not(:disabled) {
        color: var(--accent);
        background: var(--border-color);
    }
    .empty-state-hint:hover {
        color: var(--accent);
        border-color: var(--border-color);
        background: var(--secondary-bg);
    }
}
"#;

//! Buttons, inputs, focus rings, and accessibility helpers.

pub const STYLES: &str = r#"
/* Buttons - terminal style */
button {
    font-family: var(--font-mono);
    background: var(--secondary-bg);
    color: var(--text-color);
    border: 1px solid var(--border-color);
    padding: 0.25em 1ch;
    cursor: pointer;
    touch-action: manipulation;
    user-select: none;
    -webkit-user-select: none;
    transition-property: scale, background-color, border-color, color;
    transition-duration: 150ms;
    transition-timing-function: ease-out;
    min-height: auto;
    min-width: auto;
}

button:active:not(:disabled) {
    scale: 0.96;
    background: var(--muted-color);
}

button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

button.primary {
    background: var(--accent-dim);
    color: #000;
    border-color: var(--accent);
}

button.primary:active:not(:disabled) {
    background: var(--accent);
}

/* Inputs - terminal style */
input, textarea, select {
    font-family: var(--font-mono);
    font-size: inherit;
    background: var(--input-bg);
    color: var(--text-color);
    border: 1px solid var(--border-color);
    padding: 0.25em 1ch;
    outline: none;
    border-radius: 0;
}

/* Select styling - remove native appearance */
select {
    -webkit-appearance: none;
    -moz-appearance: none;
    appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23666666' d='M2 4l4 4 4-4'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 0.5ch center;
    padding-right: 2.5ch;
    cursor: pointer;
    max-width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
}

input:focus, textarea:focus, select:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 1px var(--accent);
    outline: none;
}

/* Keyboard-only focus ring. The `outline: none` rules above strip the default
   ring for mouse users; `:focus-visible` reinstates a clear high-contrast ring
   so keyboard and screen-reader users can still track focus. */
:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
}
button:focus-visible,
a:focus-visible,
[role="button"]:focus-visible,
[role="tab"]:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
    box-shadow: 0 0 0 1px var(--background);
}
input:focus-visible,
textarea:focus-visible,
select:focus-visible {
    outline: none; /* already handled by :focus rule */
    box-shadow: 0 0 0 2px var(--accent);
}

/* Skip link for keyboard users. Hidden off-screen until focused. */
.skip-link {
    position: absolute;
    top: -40px;
    left: 0;
    background: var(--accent);
    color: #000;
    padding: 0.5em 1ch;
    z-index: 10000;
    text-decoration: none;
    font-weight: bold;
}
.skip-link:focus,
.skip-link:focus-visible {
    top: 0;
}

/* Visually hidden but accessible to assistive tech. */
.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
}

textarea {
    resize: none;
    line-height: var(--lh);
}

@media (hover: hover) and (pointer: fine) {
    button:hover:not(:disabled) {
        background: var(--border-color);
        border-color: var(--muted-color);
    }
    button.primary:hover:not(:disabled) {
        background: var(--accent);
    }
}
"#;

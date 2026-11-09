//! App shell, header, box borders, and scrollbars.

pub const SHELL: &str = r#"
/* App container */
.app {
    height: 100%;
    max-height: 100%;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

/* Header with box-drawing border */
.app-header {
    border-bottom: 1px solid var(--border-color);
    padding: var(--spacing-1) var(--spacing-2);
    background: var(--card-bg);
}

/* Box-drawing style borders */
.box {
    border: 1px solid var(--border-color);
    position: relative;
}

.box::before {
    content: "┌";
    position: absolute;
    top: -1px;
    left: -1px;
    color: var(--border-color);
}
"#;

pub const SCROLLBAR: &str = r#"
/* Scrollbar - minimal */
::-webkit-scrollbar {
    width: 4px;
    height: 4px;
}

::-webkit-scrollbar-track {
    background: var(--background);
}

::-webkit-scrollbar-thumb {
    background: var(--border-color);
}

@media (hover: hover) and (pointer: fine) {
    ::-webkit-scrollbar-thumb:hover {
        background: var(--muted-color);
    }
}
"#;

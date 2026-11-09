//! Design tokens (`:root` CSS variables).

pub const STYLES: &str = r#"
:root {
    /* Monospace font stack */
    --font-mono: "Fira Code", "Fira Mono", "SF Mono", "Monaco", "Inconsolata", "Roboto Mono", "Source Code Pro", "Menlo", "Consolas", "DejaVu Sans Mono", monospace;
    
    /* Character dimensions - base unit for the grid */
    --ch: 1ch;
    --lh: 1.5;
    
    /* Colors - Terminal dark theme */
    --background: #0a0a0a;
    --foreground: #e0e0e0;
    --muted: #666666;
    --border: #333333;
    --accent: #00ff41;
    --accent-dim: #00aa2a;
    --error: #ff4444;
    --warning: #ffaa00;
    --info: #4488ff;
    
    /* Semantic colors */
    --text-color: var(--foreground);
    --muted-color: var(--muted);
    --border-color: var(--border);
    --primary-color: var(--accent);
    --card-bg: #111111;
    --input-bg: #0d0d0d;
    --secondary-bg: #1a1a1a;
    --success-color: var(--accent);
    --warning-color: var(--warning);
    --error-color: var(--error);
    
    /* Spacing in character units */
    --spacing-1: 1ch;
    --spacing-2: 2ch;
    --spacing-4: 4ch;
    
    /* Font sizes */
    --font-size-sm: 0.875rem;
    --font-size-base: 1rem;
    --font-size-lg: 1.125rem;
    --font-size-xs: 0.75rem;
}
"#;

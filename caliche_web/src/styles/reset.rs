//! Global reset and document root (`html`, `body`).

pub const STYLES: &str = r#"
* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

html {
    height: 100%;
    height: -webkit-fill-available;
    overflow: hidden;
    overscroll-behavior: none;
    font-size: 14px;
    -webkit-text-size-adjust: 100%;
}

body {
    font-family: var(--font-mono);
    background: var(--background);
    color: var(--text-color);
    line-height: var(--lh);
    height: 100%;
    height: -webkit-fill-available;
    min-height: 100dvh;
    min-height: -webkit-fill-available;
    overflow: hidden;
    overscroll-behavior: none;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    -webkit-tap-highlight-color: transparent;
    /* iOS safe area support */
    padding-top: env(safe-area-inset-top);
    padding-bottom: env(safe-area-inset-bottom);
    padding-left: env(safe-area-inset-left);
    padding-right: env(safe-area-inset-right);
}
"#;

//! Breakpoints, reduced motion, iOS fixes, and touch targets.

pub const STYLES: &str = r#"
/* Responsive - shrink in character-sized steps */
@media (max-width: 80ch) {
    html {
        font-size: 13px;
    }
}

@media (max-width: 60ch) {
    html {
        font-size: 12px;
    }
    
    .keyboard-hint {
        display: none !important;
    }
}

/* Mobile layout - hide desktop controls, show mobile button */
@media (max-width: 640px) {
    .desktop-controls {
        display: none !important;
    }
    
    .mobile-settings-btn {
        display: flex !important;
        align-items: center;
        justify-content: center;
        min-width: 44px;
        min-height: 44px;
    }
    
    ::-webkit-scrollbar {
        display: none;
    }
    * {
        scrollbar-width: none;
    }
}

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }

    button:active:not(:disabled) {
        scale: 1 !important;
    }
}

/* iOS/Mobile specific fixes */
@supports (-webkit-touch-callout: none) {
    /* iOS only */
    html, body {
        height: -webkit-fill-available;
    }
    
    .app {
        height: -webkit-fill-available;
        max-height: -webkit-fill-available;
    }
    
    .chat-view {
        height: 100% !important;
        max-height: 100% !important;
    }
    
    .chat-input-container {
        flex-shrink: 0 !important;
        position: relative;
        z-index: 10;
    }
    
    /* Prevent iOS bounce/overscroll */
    .message-list {
        -webkit-overflow-scrolling: touch;
        overscroll-behavior: contain;
    }
}

/* Mobile touch targets */
@media (pointer: coarse) {
    button {
        min-height: 44px;
        min-width: 44px;
        padding: 0.5em 1ch;
    }
    
    textarea {
        font-size: 16px; /* Prevent iOS zoom on focus */
    }

    input {
        font-size: 16px;
    }
    
    select {
        font-size: 16px;
        min-height: 44px;
    }
}
"#;

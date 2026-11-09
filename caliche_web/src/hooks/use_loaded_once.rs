//! Dioxus hooks for one-shot async resource loading.

use dioxus::prelude::*;

/// Run `loader` once when `should_load()` is true and `loaded` is still false.
///
/// The caller should set `loaded` to true after a successful fetch inside `loader`.
pub fn use_loaded_once<F, G>(should_load: G, loaded: Signal<bool>, loader: F)
where
    F: FnOnce() + 'static,
    G: Fn() -> bool + 'static,
{
    let mut loader = Some(loader);
    use_effect(move || {
        if !should_load() || *loaded.read() {
            return;
        }
        if let Some(run) = loader.take() {
            run();
        }
    });
}

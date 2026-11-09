fn main() {
    if std::env::var("CARGO_FEATURE_EMBEDDED_UI").is_ok() {
        #[expect(
            clippy::expect_used,
            reason = "Cargo always sets CARGO_MANIFEST_DIR for build scripts"
        )]
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
        let dist_index = std::path::Path::new(&manifest_dir).join("../caliche_web/dist/index.html");
        if !dist_index.exists() {
            #[expect(
                clippy::panic,
                reason = "build scripts must hard-fail when embedded-ui assets are missing"
            )]
            {
                panic!(
                    "embedded-ui is enabled but caliche_web/dist/ is missing. \
                     Run `./caliche web wasm` first. \
                     `./caliche rust install` builds those assets, then cargo-installs the CLI with this feature."
                );
            }
        }
    }
}

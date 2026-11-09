{
  description = "caliche — self-contained dev shell";

  # Enter from the repository root with `nix develop` or via direnv.
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      rust-overlay,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ rust-overlay.overlays.default ];
        };
        # Pin close to rust-toolchain.toml; include wasm32 for Dioxus `dx build`.
        # Prefer rustup (~/.cargo/bin) when both are on PATH — see Makefile PATH.
        rust = pkgs.rust-bin.stable."1.92.0".default.override {
          extensions = [
            "rust-src"
            "rust-analyzer"
            "clippy"
          ];
          targets = [ "wasm32-unknown-unknown" ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            rust
            pkgs.cargo-nextest
            pkgs.pkg-config
            pkgs.openssl
          ];
          shellHook = ''
            # App/dev secrets come from secretspec (provider per your machine
            # config: keyring locally, vault/aws in CI/prod). Run commands that
            # need secrets through it, e.g.:
            #   secretspec run -- cargo run
            echo "caliche :: $(rustc --version)"
          '';
        };

        # If caliche deploys as its own service box, its host definition
        # lives HERE too (one machine per service, provisioned separately from
        # the dev host). cloudflared goes on this host, exposing the service
        # under your Cloudflare-delegated subdomain.
        #
        # nixosConfigurations.caliche-svc = nixpkgs.lib.nixosSystem { ... };
      }
    );
}

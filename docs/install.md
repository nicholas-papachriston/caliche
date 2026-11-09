# Install the CLI

This page explains how to build and install the `caliche` binary, how to confirm that the binary on your `PATH` matches your checkout, and what installation does and does not change.

On macOS and Linux, the supported installation builds the CLI with the web UI embedded. On Windows, you can install the CLI with Cargo alone; that binary does not include the web UI unless you also follow the UI steps below.

## Supported installation methods

Install Caliche from a clone of this repository:

- On macOS and Linux, clone `nicholas-papachriston/caliche` with `--recurse-submodules` (this fetches the `shotput` submodule), then run `./caliche rust install`. The resulting binary serves the web UI at `/`.
- On Windows, for the CLI without the web UI, run `cargo install --path caliche_cli --locked --force`.

The following methods do not install this project:

- `brew install`. Homebrew has no formula or cask for Caliche.
- A GitHub release download. This repository does not publish release binaries.
- `cargo install caliche` from crates.io. Caliche is not published to crates.io; the binary's package is `caliche_cli`.
- `./caliche install`. That command installs the JavaScript and Python SDK dependencies, not the CLI binary.
- `cargo install --path caliche_cli` without `--features embedded-ui`, when you want the web UI. A binary built that way returns `404` for `GET /`.

Installation copies a binary onto your `PATH` and nothing else. It does not start a server or change the [HTTP server defaults](../SECURITY.md#http-server-defaults).

## Prerequisites

- Git.
- [rustup](https://rustup.rs) with the toolchain pinned in `rust-toolchain.toml` (`1.92.0`). Any build inside this checkout selects that toolchain automatically.
- A C++ linker. On macOS, install the Xcode Command Line Tools (`xcode-select --install`). On Windows, install Visual Studio 2022 Build Tools with the "Desktop development with C++" workload, so that `link.exe` is on `PATH`.
- Cargo's bin directory on the `PATH` of the shell you will use after installation.
- For the web UI build that `./caliche rust install` performs:
  - [dioxus-cli](https://crates.io/crates/dioxus-cli) (`dx`) at the same version as the `dioxus` crate used by `caliche_web`. The command exits if `dx` is missing.
  - The `wasm32-unknown-unknown` target. The command adds it when `rustup` is on `PATH`.
  - `wasm-opt` from [binaryen](https://github.com/WebAssembly/binaryen/releases) version 117 or newer. On macOS, run `brew install binaryen`, or `brew upgrade binaryen` if `wasm-opt --version` reports a version older than 117.

`./caliche web wasm` runs `wasm-opt -Oz` with the WebAssembly features that Rust 1.92 enables by default, including `--enable-bulk-memory` (needed for `table.fill`, which is misc prefix opcode 17). The installation stops if the optimizer is missing, if binaryen is version 116 or older, or if `wasm-opt` fails. In that case the unoptimized module is left on disk and is not embedded.

The first installation compiles native C dependencies, including libgit2, SQLite, and `ring`, so expect a long compile. The default CLI does not link the local inference libraries; they are added by the `local-inference` feature. The UI installation also compiles the Dioxus WebAssembly bundle before the CLI.

## Install on macOS or Linux

From a new clone:

```bash
git clone --recurse-submodules https://github.com/nicholas-papachriston/caliche.git
cd caliche
git rev-parse HEAD
./caliche rust install
```

`./caliche rust install` runs these two commands:

```bash
./caliche web wasm
cargo install --path caliche_cli --locked --force --features embedded-ui
```

If you already have a checkout, update it to the latest `main` first. A checkout that is behind `origin/main` installs an old build onto your `PATH`.

```bash
git fetch origin
git checkout main
git pull --ff-only origin main
git submodule update --init shotput
git rev-parse HEAD
./caliche rust install
```

The flags have these effects:

- `--locked` builds with the versions in `Cargo.lock`.
- `--force` replaces a `caliche` binary that Cargo installed earlier.
- `--features embedded-ui` builds `caliche_web/dist` into the binary.

To check that the installed binary is the one you built, see [Confirm which build is on your PATH](#confirm-which-build-is-on-your-path).

The `embedded-ui` feature is not part of the `caliche_cli` default features. `cargo build -p caliche_cli`, `cargo test`, and `./caliche build-dev` therefore do not need `dx`, and a plain `cargo install --path caliche_cli --locked --force` produces a CLI whose `GET /` returns `404`. The release builds `./caliche rust build` and `./caliche rust build-with-ui` enable the feature after the WebAssembly build. `./caliche ci` runs `./caliche rust test-embedded-ui` as part of `./caliche rust test`; that test compiles the API with the feature and checks that `GET /` returns HTML. If `caliche_web/dist/index.html` is missing, the test writes a placeholder page and deletes it afterwards. It does not replace `./caliche web wasm`.

## Install on Windows

The current `main` branch is the version to install on Windows. Run the `git` and `cargo` commands from **Developer PowerShell for VS 2022**, or from any shell in which both `where.exe link` and `where.exe cargo` resolve. If the shell cannot find Cargo, add it to `PATH` first:

```powershell
$env:Path = "$env:USERPROFILE\.cargo\bin;" + $env:Path
where.exe cargo
where.exe link
git rev-parse HEAD
cargo install --path caliche_cli --locked --force
```

This command does not pass `--features embedded-ui`, so `GET /` returns `404`. The API still answers `GET /health`. To embed the web UI, install `dx` at the version that matches the `dioxus` crate in `caliche_web`, then run `./caliche rust install` from Git Bash in this checkout.

The `./caliche` script is a Unix shell entry point. The CLI-only installation does not use it; the UI installation does.

Notes for Windows:

- The default build does not compile `tokenizers`. The features that use it (local embedding and inference in `dripstone`) build `tokenizers` without its default features, so its `esaxx-rs` dependency compiles without its C++ code and does not request the static MSVC runtime (`/MT`), which conflicts with the dynamic runtime (`/MD`) that `ort-sys` uses (`LNK2038`). A binary built earlier and still in `%USERPROFILE%\.cargo\bin` does not include this change.
- Some shells, including non-interactive shells used by coding agents, leave `%USERPROFILE%\.cargo\bin` off `PATH` even when rustup is installed. `where.exe caliche` then misses the binary you just installed, or finds a different copy first. Windows uses the first match.
- When the CLI creates its configuration and SQLite metadata file, it puts them under `%USERPROFILE%\.caliche`. Installation does not create that directory and does not write credentials.

A successful Windows build means the CLI compiled. It does not change the [HTTP server defaults](../SECURITY.md#http-server-defaults).

## Confirm which build is on your PATH

Every build in this release line reports package version `0.1.0`, so a binary built months ago reports the same version as one built today. The `build_id` value tells them apart.

`caliche --version` prints the package version, followed by `build_id` on its own line:

```text
0.1.0
build_id=<40-character commit>
```

`caliche doctor` prints the same `build_id=<commit>` line. `build_id` identifies the commit a binary was built from. It is not a security property.

Leave `CALICHE_BUILD_ID` unset for a normal installation. If it holds a 40-character hexadecimal commit (optionally followed by `-dirty`), that value is embedded instead of the commit from Git. Any other value is ignored, and the commit from Git is used.

Compare `build_id` with `git rev-parse HEAD` in this repository. `caliche_cli` is a folder in this repository, so there is no separate CLI commit to compare. If this checkout is itself a Git submodule of another repository, `build_id` is the commit of that outer repository.

On macOS or Linux:

```bash
echo "tip $(git rev-parse HEAD)"
caliche --version
command -v caliche
```

On Windows:

```powershell
Write-Output "tip $(git rev-parse HEAD)"
caliche --version
where.exe caliche
```

The installed binary matches your checkout when `build_id` equals the commit and the resolved path is in Cargo's bin directory (`~/.cargo/bin/caliche`, or `%USERPROFILE%\.cargo\bin\caliche.exe` on Windows).

The installed binary does not match when any of the following is true:

- `build_id` differs from `git rev-parse HEAD`.
- `build_id=unknown`. Git was not available at compile time, and `CALICHE_BUILD_ID` was unset or was not a commit. `unknown` is not a commit and never matches `git rev-parse HEAD`.
- `build_id` ends in `-dirty`. The working tree had uncommitted changes at compile time. Commit or discard the changes, then install again from a clean checkout.
- `command -v` or `where.exe` shows a path outside Cargo's bin directory. Another copy of `caliche` comes earlier on `PATH`. Move Cargo's bin directory to the front of `PATH`, or remove the other copy, and run `--version` again.

`cargo install --list` should include `caliche_cli v0.1.0`. That entry confirms that Cargo installed the package, but it does not prove that the first `caliche` on `PATH` is that installation. The `build_id` comparison is the reliable check.

## After installation

Start a local server from the installed binary:

```bash
caliche --server --bind 127.0.0.1 --port 3000
```

The server keeps its security defaults, described in [HTTP server defaults](../SECURITY.md#http-server-defaults). To create sessions with `curl`, follow [HTTP sessions](sessions/sessions.md).

On a binary installed with `./caliche rust install`, confirm that the web UI is embedded:

```bash
curl -sS -D- -o /tmp/caliche-web.html http://127.0.0.1:3000/
```

The response status is `200`, `Content-Type` starts with `text/html`, and the body is the UI page. The server log includes `Embedded UI at http://127.0.0.1:3000/`; open that address in a browser. A binary without the UI logs `No embedded UI in this binary.` at startup instead. If you run `caliche` without `--server` in a second terminal, the REPL attaches to the running server through `server.lock`. The browser, the CLI, and `curl` then share the sessions of that one process.

`GET /health` checks the API and returns `200` whether or not the UI is embedded. `GET /` returns `404` only when the binary was built without `--features embedded-ui`, for example with the Windows CLI-only command or with `cargo run` without the feature. That `404` means the UI assets are absent from the binary.

If you need provider keys in a file, copy `.env.example` to `.env` and pass it with `--env-file .env`. Installation does not require a `.env` file.

## What installation does not change

- The security defaults of the HTTP server stay as they are; see [HTTP server defaults](../SECURITY.md#http-server-defaults).
- Installation does not set `VECTOR_BACKEND`. The retrieval status values are described in [Environment variables](reference/environment-variables.md#vector-backends).

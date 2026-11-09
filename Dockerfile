# syntax=docker/dockerfile:1.4
# Multi-stage build for the caliche_server image.
# Requires Docker BuildKit (DOCKER_BUILDKIT=1).
#
# Workspace copy lists match [workspace].members in Cargo.toml. Members the
# server does not compile are stubbed so cargo can load the workspace.
#
# The runtime stage sets CALICHE_ENV=production. The process refuses to
# start until JWT auth is configured.
#
# Local open bind (still publish on loopback):
#   docker build -t caliche-server .
#   docker run --rm -p 127.0.0.1:3000:3000 \
#     -e CALICHE_ALLOW_INSECURE_PRODUCTION=1 \
#     caliche-server
#
# Production:
#   docker run --rm -p 127.0.0.1:3000:3000 \
#     -e JWT_ENABLED=true \
#     -e JWT_PUBLIC_KEY="<pem>" \
#     -e JWT_AUDIENCE="<audience>" \
#     -e JWT_ISSUER="<issuer>" \
#     -e CALICHE_REQUIRE_AUTH=true \
#     -e PLUGIN_HTTP_ALLOWLIST="<hosts>" \
#     caliche-server
#
# Inside the container the process listens on 0.0.0.0 so Docker can publish
# the port. Publish it on 127.0.0.1 unless the host is meant to be public.

# ==============================================================================
# Stage 1: Chef - Prepare cargo-chef for dependency caching
# ==============================================================================
FROM rust:1.92-slim AS chef

WORKDIR /app

RUN cargo install cargo-chef --locked

# ==============================================================================
# Stage 2: Planner - Generate recipe.json (dependency manifest)
# ==============================================================================
FROM chef AS planner

COPY Cargo.toml Cargo.lock README.md ./
COPY dripstone/Cargo.toml dripstone/
COPY caliche_js/Cargo.toml caliche_js/
COPY caliche_py/Cargo.toml caliche_py/
COPY caliche_server/Cargo.toml caliche_server/
COPY caliche_cli/Cargo.toml caliche_cli/
COPY caliche_web/Cargo.toml caliche_web/
COPY caliche_sandbox/Cargo.toml caliche_sandbox/
COPY crates/caliche_acp_client/Cargo.toml crates/caliche_acp_client/
COPY crates/caliche_contracts/Cargo.toml crates/caliche_contracts/
COPY crates/caliche_orchestration/Cargo.toml crates/caliche_orchestration/
COPY crates/caliche_mcp/Cargo.toml crates/caliche_mcp/
COPY crates/caliche_policy/Cargo.toml crates/caliche_policy/

COPY src src
COPY dripstone/src dripstone/src
COPY dripstone/tests dripstone/tests
COPY dripstone/examples dripstone/examples
COPY caliche_server/src caliche_server/src
COPY caliche_server/build.rs caliche_server/build.rs
COPY caliche_sandbox/src caliche_sandbox/src
COPY caliche_sandbox/build.rs caliche_sandbox/build.rs
COPY crates/caliche_contracts/src crates/caliche_contracts/src
COPY crates/caliche_orchestration/src crates/caliche_orchestration/src
COPY crates/caliche_mcp/src crates/caliche_mcp/src
COPY crates/caliche_policy/src crates/caliche_policy/src

# Stub workspace members this image does not compile, and root example/test
# paths cargo checks when it loads the manifests.
RUN mkdir -p \
        caliche_js/src \
        caliche_py/src \
        caliche_web/src \
        caliche_cli/src \
        crates/caliche_acp_client/src \
        examples \
        tests && \
    printf '%s\n' '' > caliche_js/src/lib.rs && \
    printf '%s\n' 'fn main() {}' > caliche_js/build.rs && \
    printf '%s\n' '' > caliche_py/src/lib.rs && \
    printf '%s\n' 'pub fn _dummy() {}' > caliche_web/src/lib.rs && \
    printf '%s\n' 'fn main() {}' > caliche_web/src/main.rs && \
    printf '%s\n' 'pub fn _dummy() {}' > caliche_cli/src/lib.rs && \
    printf '%s\n' 'fn main() {}' > caliche_cli/src/main.rs && \
    printf '%s\n' 'pub fn _dummy() {}' > crates/caliche_acp_client/src/lib.rs && \
    awk -F'"' ' \
      function emit() { \
        if (name == "") return; \
        if (path != "") print path; \
        else if (kind == "test") print "tests/" name ".rs"; \
        else print "examples/" name ".rs"; \
      } \
      /^\[\[example\]\]/ { if (kind != "") emit(); kind = "example"; name = ""; path = ""; next } \
      /^\[\[test\]\]/ { if (kind != "") emit(); kind = "test"; name = ""; path = ""; next } \
      kind != "" && /^[[:space:]]*name[[:space:]]*=/ { name = $2; next } \
      kind != "" && /^[[:space:]]*path[[:space:]]*=/ { path = $2; next } \
      kind != "" && /^\[/ { emit(); kind = ""; next } \
      END { if (kind != "") emit() } \
    ' Cargo.toml | while read -r ex_path; do \
      mkdir -p "$(dirname "$ex_path")" && printf '%s\n' 'fn main() {}' > "$ex_path"; \
    done

RUN cargo chef prepare --recipe-path recipe.json

# ==============================================================================
# Stage 3: Builder - Build dependencies and application
# ==============================================================================
FROM chef AS builder

ARG RELEASE_FAST=0

WORKDIR /app

ENV CARGO_BUILD_JOBS=1

RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libasound2-dev \
    binutils \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY --from=planner /app/recipe.json recipe.json

COPY Cargo.toml Cargo.lock README.md ./
COPY dripstone/Cargo.toml dripstone/
COPY caliche_js/Cargo.toml caliche_js/
COPY caliche_py/Cargo.toml caliche_py/
COPY caliche_server/Cargo.toml caliche_server/
COPY caliche_cli/Cargo.toml caliche_cli/
COPY caliche_web/Cargo.toml caliche_web/
COPY caliche_sandbox/Cargo.toml caliche_sandbox/
COPY crates/caliche_acp_client/Cargo.toml crates/caliche_acp_client/
COPY crates/caliche_contracts/Cargo.toml crates/caliche_contracts/
COPY crates/caliche_orchestration/Cargo.toml crates/caliche_orchestration/
COPY crates/caliche_mcp/Cargo.toml crates/caliche_mcp/
COPY crates/caliche_policy/Cargo.toml crates/caliche_policy/

RUN mkdir -p \
        caliche_js/src \
        caliche_py/src \
        caliche_web/src \
        caliche_cli/src \
        caliche_server/src \
        caliche_sandbox/src \
        crates/caliche_acp_client/src \
        crates/caliche_contracts/src \
        crates/caliche_orchestration/src \
        crates/caliche_mcp/src \
        crates/caliche_policy/src \
        dripstone/src \
        dripstone/examples \
        dripstone/tests \
        examples \
        tests && \
    printf '%s\n' '' > caliche_js/src/lib.rs && \
    printf '%s\n' 'fn main() {}' > caliche_js/build.rs && \
    printf '%s\n' '' > caliche_py/src/lib.rs && \
    printf '%s\n' 'pub fn _dummy() {}' > caliche_web/src/lib.rs && \
    printf '%s\n' 'fn main() {}' > caliche_web/src/main.rs && \
    printf '%s\n' 'pub fn _dummy() {}' > caliche_cli/src/lib.rs && \
    printf '%s\n' 'fn main() {}' > caliche_cli/src/main.rs && \
    printf '%s\n' 'pub fn _dummy() {}' > caliche_server/src/lib.rs && \
    printf '%s\n' 'fn main() {}' > caliche_server/src/main.rs && \
    printf '%s\n' 'fn main() {}' > caliche_server/build.rs && \
    printf '%s\n' 'pub fn _dummy() {}' > caliche_sandbox/src/lib.rs && \
    printf '%s\n' 'fn main() {}' > caliche_sandbox/build.rs && \
    printf '%s\n' 'pub fn _dummy() {}' > crates/caliche_acp_client/src/lib.rs && \
    printf '%s\n' 'pub fn _dummy() {}' > crates/caliche_contracts/src/lib.rs && \
    printf '%s\n' 'pub fn _dummy() {}' > crates/caliche_orchestration/src/lib.rs && \
    printf '%s\n' 'pub fn _dummy() {}' > crates/caliche_mcp/src/lib.rs && \
    printf '%s\n' 'pub fn _dummy() {}' > crates/caliche_policy/src/lib.rs && \
    printf '%s\n' '' > dripstone/src/lib.rs && \
    awk -F'"' ' \
      function emit() { \
        if (name == "") return; \
        if (path != "") print prefix path; \
        else if (kind == "test") print prefix "tests/" name ".rs"; \
        else print prefix "examples/" name ".rs"; \
      } \
      /^\[\[example\]\]/ { if (kind != "") emit(); kind = "example"; name = ""; path = ""; next } \
      /^\[\[test\]\]/ { if (kind != "") emit(); kind = "test"; name = ""; path = ""; next } \
      kind != "" && /^[[:space:]]*name[[:space:]]*=/ { name = $2; next } \
      kind != "" && /^[[:space:]]*path[[:space:]]*=/ { path = $2; next } \
      kind != "" && /^\[/ { emit(); kind = ""; next } \
      END { if (kind != "") emit() } \
    ' Cargo.toml | while read -r ex_path; do \
      mkdir -p "$(dirname "$ex_path")" && printf '%s\n' 'fn main() {}' > "$ex_path"; \
    done && \
    awk -F'"' -v prefix='dripstone/' ' \
      function emit() { \
        if (name == "") return; \
        if (path != "") print prefix path; \
        else if (kind == "test") print prefix "tests/" name ".rs"; \
        else print prefix "examples/" name ".rs"; \
      } \
      /^\[\[example\]\]/ { if (kind != "") emit(); kind = "example"; name = ""; path = ""; next } \
      /^\[\[test\]\]/ { if (kind != "") emit(); kind = "test"; name = ""; path = ""; next } \
      kind != "" && /^[[:space:]]*name[[:space:]]*=/ { name = $2; next } \
      kind != "" && /^[[:space:]]*path[[:space:]]*=/ { path = $2; next } \
      kind != "" && /^\[/ { emit(); kind = ""; next } \
      END { if (kind != "") emit() } \
    ' dripstone/Cargo.toml | while read -r ex_path; do \
      mkdir -p "$(dirname "$ex_path")" && printf '%s\n' 'fn main() {}' > "$ex_path"; \
    done

# The image builds the server with `app` (cloud providers, vector stores, Redis
# and PostgreSQL session stores). The route groups (`tools-rag`, `document-loaders`,
# `chains`, `retrieval-chains`, `reasoning-patterns`) stay off.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target,sharing=locked \
    if [ "$RELEASE_FAST" = "1" ]; then \
        echo "Building dependencies (fast mode)..." && \
        cargo chef cook --release --recipe-path recipe.json --package caliche_server --features app; \
    else \
        echo "Building dependencies (optimized mode)..." && \
        CARGO_PROFILE_RELEASE_DEBUG=0 \
        CARGO_PROFILE_RELEASE_OPT_LEVEL="z" \
        CARGO_PROFILE_RELEASE_LTO=true \
        cargo chef cook --release --recipe-path recipe.json --package caliche_server --features app; \
    fi

COPY src src
COPY dripstone/src dripstone/src
COPY caliche_server/src caliche_server/src
COPY caliche_server/build.rs caliche_server/build.rs
COPY caliche_sandbox/src caliche_sandbox/src
COPY caliche_sandbox/build.rs caliche_sandbox/build.rs
COPY crates/caliche_contracts/src crates/caliche_contracts/src
COPY crates/caliche_orchestration/src crates/caliche_orchestration/src
COPY crates/caliche_mcp/src crates/caliche_mcp/src
COPY crates/caliche_policy/src crates/caliche_policy/src

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target,sharing=locked \
    if [ "$RELEASE_FAST" = "1" ]; then \
        echo "Building application (fast mode)..." && \
        cargo build --release --package caliche_server --features app --bin caliche_server && \
        cp /app/target/release/caliche_server /app/caliche_server; \
    else \
        echo "Building application (optimized mode)..." && \
        CARGO_PROFILE_RELEASE_DEBUG=0 \
        CARGO_PROFILE_RELEASE_OPT_LEVEL="z" \
        CARGO_PROFILE_RELEASE_LTO=true \
        cargo build --release --package caliche_server --features app --bin caliche_server && \
        cp /app/target/release/caliche_server /app/caliche_server; \
    fi && \
    strip /app/caliche_server

# ==============================================================================
# Stage 4: Runtime - Minimal production image
# ==============================================================================
FROM debian:trixie-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

COPY --from=builder /app/caliche_server ./caliche_server

RUN chown appuser:appuser /app/caliche_server

USER appuser

EXPOSE 3000

# Fail closed until JWT_ENABLED, CALICHE_REQUIRE_AUTH, JWT_AUDIENCE, and
# JWT_ISSUER are set. CALICHE_ALLOW_INSECURE_PRODUCTION=1 is the local override.
ENV CALICHE_ENV=production

ENTRYPOINT ["./caliche_server"]
CMD ["--port", "3000", "--bind", "0.0.0.0"]

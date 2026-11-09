# Multi-stage build for Caliche Web (Dioxus/WASM)
# Build context must be repo root so the workspace resolves.

FROM rust:1.92-bookworm AS builder

WORKDIR /app

# dioxus-cli matches the dioxus crate major.minor in AGENTS.md.
# No nginx image digest is recorded in this repo.
RUN rustup target add wasm32-unknown-unknown && \
    cargo install dioxus-cli --version 0.7.6 --locked

COPY . .

# Build Dioxus UI for web (output under target/dx/caliche_web/release/web/public)
RUN cd caliche_web && dx build --release --platform web

# nginx-unprivileged style: listen on 8080 and run as the nginx user.
FROM nginx:alpine

RUN printf '%s\n' \
    'pid /tmp/nginx.pid;' \
    'worker_processes auto;' \
    'error_log /tmp/nginx-error.log warn;' \
    'events { worker_connections 1024; }' \
    'http {' \
    '  include /etc/nginx/mime.types;' \
    '  default_type application/octet-stream;' \
    '  access_log /tmp/nginx-access.log;' \
    '  client_body_temp_path /tmp/nginx-client;' \
    '  proxy_temp_path /tmp/nginx-proxy;' \
    '  fastcgi_temp_path /tmp/nginx-fastcgi;' \
    '  uwsgi_temp_path /tmp/nginx-uwsgi;' \
    '  scgi_temp_path /tmp/nginx-scgi;' \
    '  server {' \
    '    listen 8080;' \
    '    root /usr/share/nginx/html;' \
    '    index index.html;' \
    '    location / { try_files $uri $uri/ /index.html; }' \
    '  }' \
    '}' \
    > /etc/nginx/nginx.conf && \
    mkdir -p /tmp/nginx-client /tmp/nginx-proxy /tmp/nginx-fastcgi /tmp/nginx-uwsgi /tmp/nginx-scgi && \
    chown -R nginx:nginx /usr/share/nginx/html /tmp/nginx-client /tmp/nginx-proxy /tmp/nginx-fastcgi /tmp/nginx-uwsgi /tmp/nginx-scgi

COPY --from=builder --chown=nginx:nginx /app/target/dx/caliche_web/release/web/public /usr/share/nginx/html

USER nginx

EXPOSE 8080

CMD ["nginx", "-g", "daemon off;"]

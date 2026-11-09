# Production authentication checklist

The defaults that this checklist changes are in [HTTP server defaults](../../SECURITY.md#http-server-defaults). The production profile, which refuses to start with an incomplete configuration, is active only when `CALICHE_ENV` is `production` or `prod`.

This checklist names environment variables only. It does not contain values for any real host. Confirming the settings of a real deployment is the operator's responsibility.

## Required settings for every public host

| Item | Variable or action | Behavior when missing |
| --- | --- | --- |
| Production profile | `CALICHE_ENV=production` | The server refuses to start without JWT and required authentication, unless `CALICHE_ALLOW_INSECURE_PRODUCTION` is set. |
| JWT validation | `JWT_ENABLED=true`, plus a public key or a JWKS URL, and non-empty `JWT_AUDIENCE` and `JWT_ISSUER` | The server refuses to start if any of these is missing. |
| Required authentication | `CALICHE_REQUIRE_AUTH=true` | With JWT enabled, most `/api/*` routes require a bearer token. |
| Insecure override | Leave `CALICHE_ALLOW_INSECURE_PRODUCTION` unset. | Set it only to override the profile on a local machine. |
| Unauthenticated orchestration | Leave `CALICHE_ALLOW_UNAUTH_ORCHESTRATION` unset. The production profile ignores it. | Not applicable. |
| Listen address | Listen on a private address, or put an authenticating reverse proxy in front of the server. | The server logs a warning and still starts. It does not refuse. |
| Plugin egress | Set `PLUGIN_HTTP_ALLOWLIST`. | The production profile refuses to start without it. |
| Public chat | Set `CALICHE_PUBLIC_CHAT=true` only when anonymous chat is intended. | Operator decision. |
| Metrics | When authentication is required, `/metrics` accepts a JWT or a `CALICHE_METRICS_TOKEN` bearer token. | Operator decision. |
| Trusted proxy | Set `CALICHE_TRUSTED_PROXY` only behind a real proxy, together with its CIDR ranges. | The server refuses to start if the trusted proxy setting is on without CIDR ranges. |

`JWT_AUDIENCE` and `JWT_ISSUER` are values that you supply. Empty values stop the server at startup, and the server never substitutes a built-in audience or issuer.

## Listen address

| Setting | What it controls |
| --- | --- |
| CLI `--bind` | The address that `caliche --server` listens on. The default is `127.0.0.1`. It does not read `API_HOST`. Authentication is configured separately; see [Security](../../SECURITY.md). |
| `BIND_ADDRESS` | The address that the standalone `caliche_server` binary listens on (its `--bind` option). The default is `127.0.0.1` when the variable is unset. It does not read `API_HOST`. |
| `API_HOST` | Documented in `.env.example`. The router reads it at startup only to decide whether to log the warning described below. It is not used as the CLI listen address. |

`caliche doctor` prints the address that `--server` would listen on, together with the opt-in settings:

```text
build_id=<git commit>
jwt_enabled=true|false
require_auth=true|false
listen_bind=<--bind>:<--port>
api_host_is_cli_listen=false
open_bind_warn=true|false
```

- `listen_bind` is the CLI `--bind` and `--port` (by default `127.0.0.1:3000`).
- `api_host_is_cli_listen` is always `false`, because `API_HOST` does not set that address.
- `jwt_enabled` reports `JWT_ENABLED`, and `require_auth` reports `CALICHE_REQUIRE_AUTH`. Both default to `false`. These lines report the current settings; they do not change them.
- `open_bind_warn` is `true` when `listen_bind` is an every-interface address (`0.0.0.0`, `::`, or `[::]`). It is a warning only; the server still starts. An explicit non-loopback `--bind` is an opt-in choice and is not refused. The only refusal is the production profile check described above, when the JWT and required authentication settings are incomplete.
- `build_id` is the same line that `caliche --version` prints. A packager building without a `.git` directory can set `CALICHE_BUILD_ID` to the Caliche commit at compile time. When `build_id` is a 40-character commit, you can compare a binary on `PATH` with your checkout. `build_id=unknown` means Git was not available at compile time and `CALICHE_BUILD_ID` was unset or was not a commit; `unknown` never matches `git rev-parse HEAD`. `build_id` is not a security property. [Install](../install.md) explains how to replace a stale `0.1.0` binary on `PATH` with one whose `build_id` equals `git rev-parse HEAD`.

The addresses `0.0.0.0` and `::` listen on every network interface. Listening on them is unsafe while `JWT_ENABLED` and `CALICHE_REQUIRE_AUTH` keep their default values and `CALICHE_ENV` is not `production`.

Local development without authentication remains available on a loopback address (`127.0.0.1`, `::1`, or `localhost`) when the production profile is not set.

## Warning for an open listen address

The router logs a warning at startup, and still starts, when all of the following are true:

- `API_HOST` or `BIND_ADDRESS` is `0.0.0.0`, `::`, or `[::]`.
- The production profile is off.
- `CALICHE_REQUIRE_AUTH` is off.

`caliche --server` and `caliche_server` also log the warning when the resolved listen address is one of those addresses. The warning does not replace the production profile check.

## Checks before deployment

Do not paste keys, audience values, or issuer values into tickets or chat.

```text
# The server must refuse to start when the production profile is incomplete:
CALICHE_ENV=production JWT_ENABLED=false CALICHE_REQUIRE_AUTH=false \
  <binary-or-cargo-run>   # must refuse to start

# The server must start only with the complete JWT and required authentication
# settings (keys, audience, and issuer from your secret store).
```

## Embedded web UI and session routes

On macOS and Linux, `./caliche rust install` builds a binary that serves the web UI at `/` on the same origin as `/api/*`. That command runs `./caliche web wasm`, then `cargo install --path caliche_cli --locked --force --features embedded-ui`. A binary built without that feature or without the UI assets returns `404` for `GET /`. That `404` means only that the UI is absent from the binary: it does not mean that the API is down (check `GET /health`), and it does not make the server any more secure.

With an open listen address and authentication off, the server accepts session creation, the process-wide `GET /api/agent/sessions` list, and `POST /api/agent/sessions/{id}/cancel` from every network interface. When the UI is embedded, the same socket also serves the operator UI. A bearer token pasted into the UI is an operator credential. The JWT `aud` and `iss` values are ones that you supply.

`GET /api/agent/sessions` returns every live session in the process. The list is not filtered by the JWT `sub` claim or by roles. Cancel uses the same route authentication as a turn, so with authentication off, any caller can cancel any session in the process.

Example `curl` commands are in [HTTP sessions](../sessions/sessions.md).

## Remaining risk

Treat a public listen address on a server whose `CALICHE_ENV` is unset, or set to something other than `production`, as a critical misconfiguration. The startup warning makes this configuration visible, but it does not prove that a live host is protected.

On each public host, confirm the following without pasting secret values:

- `CALICHE_ENV=production`
- `JWT_ENABLED=true` and `CALICHE_REQUIRE_AUTH=true`
- `CALICHE_ALLOW_INSECURE_PRODUCTION` is unset
- The listen address is private, or an authenticating reverse proxy is in front of the server

# HTTP sessions

A [session](../concepts.md#session) is one conversation held in the memory of the server process. This page shows how to create a session over HTTP, run a [turn](../concepts.md#turn), read the event stream, and cancel or delete the session. It also describes how sessions leave the cache and how JWT authentication applies to them.

The examples use a server with the defaults described in [HTTP server defaults](../../SECURITY.md#http-server-defaults): it listens on `127.0.0.1`, authentication is off, and all callers share one session list.

The CLI, the HTTP API, and the embedded web UI share the sessions of the process they connect to. There is no other terminal UI or IDE extension in this repository that uses them. The session routes on this page do not call Dripstone; the retrieval status is described separately under [Vector backends](../reference/environment-variables.md#vector-backends).

The authentication and listen address settings for a public host are in [Security](../../SECURITY.md) and the [production authentication checklist](../ops/production-auth-checklist.md).

## Starting the server

The server does not need a model API key to start. Creating, listing, deleting, and cancelling sessions do not call a model; running a turn or a task does.

```bash
caliche --server --bind 127.0.0.1 --port 3000
```

Use the binary from [Install](../install.md) (`./caliche rust install` on macOS and Linux). That binary serves the web UI at `/` and the API on the same origin, and its startup log includes `Embedded UI at http://127.0.0.1:3000/`. If you run the REPL (`caliche`, without `--server`) in another terminal, it attaches to the server through `server.lock` and uses the same sessions.

`cargo run -p caliche_cli` without `--features embedded-ui` builds a different binary, whose `GET /` returns `404`. It does not replace the `caliche` that is already on `PATH`.

Every build reports package version `0.1.0`. Before you assume that the binary on `PATH` was built from your checkout, compare the `build_id=` line of `caliche --version` and `caliche doctor` with `git rev-parse HEAD`. `build_id=unknown` means Git was not available at compile time and `CALICHE_BUILD_ID` was unset or was not a 40-character commit; it never matches `HEAD`.

`--bind` and `--port` set the listen address, which defaults to `127.0.0.1:3000`. They do not read `API_HOST`. With an explicit `--bind 0.0.0.0` or `--bind ::`, the server logs an `open_bind_warn` warning and still starts. A loopback address produces no warning.

Check that the API is up:

```bash
curl -sf -o /dev/null -w 'health %{http_code}\n' http://127.0.0.1:3000/health
```

A `200` response means the API is running. On a binary installed with `./caliche rust install`, `GET /` also returns `200` with `text/html`. On a binary built without `embedded-ui`, `GET /` returns `404` while `GET /health` still returns `200`.

```bash
curl -sS -D- -o /tmp/caliche-web.html http://127.0.0.1:3000/
```

## Which provider a turn uses

At startup, the server chooses a provider in this order:

1. `--provider`, when it is set.
2. `settings.json` in the same directory as `mcp.json`, when that file names both a provider and a model.
3. The `last_provider` value stored in SQLite, when `last_model` is also stored. This takes precedence over environment variables.
4. `AGENT_PROVIDER`.
5. `DEFAULT_PROVIDER`.
6. The built-in default, `openrouter`.

When `--provider` names the same provider as `settings.json`, the model recorded in that file is still used. `--model` and `--temperature` override the values in the file.

The standalone `caliche_server` binary uses the same order without step 3, because it does not open the SQLite store. With no flag, settings file or environment variable, it also uses `openrouter`.

`caliche doctor` prints `default_provider` and `default_provider_source`, which is one of `flag`, `last`, `env`, or `default`. A plain `--server` does not probe for a local model runtime. Local runtime detection runs only when `--provider` is not set and either `--prefer-local` is set or the process is opening the REPL. `doctor` does not probe either.

The body of a create request does not select a provider. Unknown JSON fields are ignored, so a body such as `{"provider": ..., "model": ...}` still creates a session but does not change the model.

If the selected provider has no API key, the turn fails, and the response body contains `No credentials found for provider '<name>'`. The server does not switch to another provider; a missing key is reported as a credential error.

If the egress policy blocks the provider URL, for example a restored local Ollama base URL, the server still starts and does not select another provider. Turns then fail with that provider's egress error (`egress provider url rejected`, or an error naming the blocked host).

## Creating a session, running a turn, and deleting it

In this example, `BASE` is the loopback server. The field that carries the prompt in a turn request is `input`. A body that sends `message` instead returns `422` with a plain text error.

```bash
BASE=http://127.0.0.1:3000

SID=$(curl -sS -X POST "$BASE/api/agent/sessions" \
  -H 'Content-Type: application/json' \
  -d '{"max_turns":1}' | sed -n 's/.*"session_id":"\([^"]*\)".*/\1/p')
test -n "$SID"
echo "session $SID"

curl -sS "$BASE/api/agent/sessions"

curl -N "$BASE/api/agent/sessions/$SID/events" >/tmp/caliche-events.txt &
SSE_PID=$!
sleep 0.2

curl -sS -X POST "$BASE/api/agent/sessions/$SID/turn" \
  -H 'Content-Type: application/json' \
  -d '{"input":"Reply with the single word pong."}'

echo '--- sse ---'
head -n 40 /tmp/caliche-events.txt

curl -sS -o /tmp/caliche-turn-err.txt -w 'bad-body %{http_code}\n' \
  -X POST "$BASE/api/agent/sessions/$SID/turn" \
  -H 'Content-Type: application/json' \
  -d '{"message":"Reply with the single word pong."}'

curl -sS -o /dev/null -w 'delete %{http_code}\n' \
  -X DELETE "$BASE/api/agent/sessions/$SID"

kill "$SSE_PID" 2>/dev/null || true
```

A create request returns `{"session_id":"<id>","created_at":"<rfc3339>"}`. Error responses from the handlers (`404`, `409`, and `422`) are plain text. Successful responses to create, list, turn, task, and cancel requests are JSON.

Subscribe to the event stream before you send `POST .../turn`; otherwise you miss events that were already sent. The `-N` option turns off output buffering in `curl`. When the process has no live sessions, `GET /api/agent/sessions` returns `200` with `[]`.

A task runs on the same session id with a different request body. Run it before the `DELETE`:

```bash
curl -sS -X POST "$BASE/api/agent/sessions/$SID/task" \
  -H 'Content-Type: application/json' \
  -d '{"task":"Reply with the single word pong."}'
```

## Events

`GET /api/agent/sessions/{id}/events` returns a `text/event-stream` (server-sent events). A `: keep-alive` comment line arrives about every 15 seconds.

The `event:` name of a frame and the `type` field in its JSON data are different strings. Clients should match on the `event:` line.

| `event:` | JSON `type` |
| --- | --- |
| `turn.start` | `turn_start` |
| `turn.complete` | `turn_complete` |
| `agent.thinking` | `thinking` |
| `tool.start` | `tool_executing` |
| `tool.end` | `tool_complete` |
| `session.updated` | `session_start` or `session_end` |
| `text_chunk` | `text_chunk` |
| `done` | `done` |
| `error` | `error` |
| `resync` | `resync` |

Frames look like this:

```text
event: turn.start
data: {"version":"1","type":"turn_start","payload":{"type":"turn_start","turn_id":1,"input":"...","timestamp":"..."}}

event: turn.complete
data: {"version":"1","type":"turn_complete","payload":{"type":"turn_complete","turn_id":1,"response":"...","usage":{},"duration_ms":0,"timestamp":"..."}}
```

`version` is `"1"`, and `payload` is the harness event. Every other event (`planning`, `tool_error`, `file_read`, `checkpoint_created`, and the rest) uses the same snake_case name on both lines.

### Missed events

A `resync` frame means that this subscriber fell behind the live broadcast and missed events. The number of missed broadcasts is in `payload.skipped`, inside the same envelope as the other frames on this stream. The task and chat streams send raw harness events instead, so their resync data is `{"type":"resync","skipped":n}` without an envelope. The frame after a `resync` is later than the last frame you handled, so read the session again (its message history and latest checkpoint) before you act on later frames.

The task and chat streams can resume: reconnect with `after` set to the last durable event id you stored. `GET /api/agent/sessions/{id}/events` is live only. Its frames have no SSE id, and the handler does not read `Last-Event-ID`, so reconnecting starts a new live stream and does not replay the missed events.

JSON-RPC session subscriptions report the same gap as a `sessions.resync` notification with `session_id` and `skipped`. If sending that notification fails, the subscription stops, and later `sessions.event` notifications are not delivered. `caliche code` prints the same count on a single `[resync]` line.

## Cancelling a session

`POST /api/agent/sessions/{id}/cancel` and `POST /api/agent/sessions/{id}/stop` are the same handler, and both take an empty body. Cancel stops a turn that is in progress and removes the session, so a later turn on that id returns `404`.

To cancel a turn in progress, call cancel from a second client. The turn request may return `409` with the plain text `Session turn cancelled: <id>`. If the cancel is processed before the turn handler starts, the turn request returns `404` instead. In both cases the session is removed.

Cancelling an idle session also returns `cancelled: true` and removes the id. A second cancel or stop returns `404`.

Removal of idle sessions from the cache uses a separate timer, described in [Empty sessions and the live cache](#empty-sessions-and-the-live-cache); it does not go through this route.

```bash
SID=$(curl -sS -X POST "$BASE/api/agent/sessions" \
  -H 'Content-Type: application/json' \
  -d '{"max_turns":1}' | sed -n 's/.*"session_id":"\([^"]*\)".*/\1/p')
test -n "$SID"

curl -sS -X POST "$BASE/api/agent/sessions/$SID/turn" \
  -H 'Content-Type: application/json' \
  -d '{"input":"Count slowly to twenty."}' >/tmp/caliche-inflight.txt &

curl -sS -X POST "$BASE/api/agent/sessions/$SID/cancel"
# {"session_id":"<id>","cancelled":true}

curl -sS -o /tmp/caliche-after-cancel.txt -w '%{http_code}\n' \
  -X POST "$BASE/api/agent/sessions/$SID/turn" \
  -H 'Content-Type: application/json' \
  -d '{"input":"again"}'
# 404
```

`DELETE /api/agent/sessions/{id}` removes a live session and returns `204`. Use it when no turn needs to be stopped. After a cancel, `DELETE` returns `404` because the id is already gone.

With authentication off, any caller that can reach the server can list, cancel, and delete every session.

## Listing sessions

`GET /api/agent/sessions` returns `200` with a JSON array. Each element has `session_id`, `created_at`, `last_activity`, `turn_count`, and `total_tokens`. `title` and `name_slug` appear only when they are set. `name_slug` is a slug used by the session tracker, and `title` is the label meant for people.

The JWT `sub` claim and roles do not filter this list.

## Session titles

`title` is optional on `POST /api/agent/sessions`. If you omit it, send `null`, or send an empty string, the request still returns `200`, and the session has no `title` field in the list. A non-empty title is trimmed and stored. A title longer than 80 characters returns `400` with the plain text `session title is longer than 80 characters`.

```bash
SID=$(curl -sS -X POST "$BASE/api/agent/sessions" \
  -H 'Content-Type: application/json' \
  -d '{"title":"desk","max_turns":1}' | sed -n 's/.*"session_id":"\([^"]*\)".*/\1/p')
test -n "$SID"

curl -sS "$BASE/api/agent/sessions"
```

The session panel of the embedded web UI shows the title of the active session in its header. `GET /` serves the panel on a binary installed with `./caliche rust install`, or on a debug build made with `./caliche web wasm` followed by `cargo build -p caliche_cli --features embedded-ui`. The panel and `GET /api/agent/sessions` read the same process.

## Empty sessions and the live cache

A live session is empty while its `turn_count` is `0`. Creating a session does not count as a turn. Cancelling an empty session still returns `cancelled: true` and removes the id.

Empty sessions and sessions with turns share one cache. A session leaves the cache when any of the following happens:

| Trigger | What you observe |
| --- | --- |
| `DELETE /api/agent/sessions/{id}` | `204`, and the id no longer appears in the list. |
| `POST .../cancel` or `POST .../stop` | `{"cancelled":true}`, and the id no longer appears in the list. |
| Idle time | The cache removes a session 3600 seconds after the last insert or read of its entry. Creating a session inserts it. `GET /api/agent/sessions` and `GET /api/agent/sessions/{id}` read the entry, so they restart the 3600 seconds. A session that is created and then never listed, read, or used for a turn or task can therefore leave the cache after that time. The same timer applies to sessions whose `turn_count` is greater than `0`. |
| Capacity | The cache holds at most 1000 sessions and evicts entries beyond that. Capacity eviction uses the same listener as idle eviction. |
| Process exit | The in-memory cache is lost. |

When the cache evicts a session, the id is removed from the process runtime, and any turn in progress is signaled to cancel. This is separate from the cancel route.

A router task that runs about every 300 seconds removes stale document collections and orphaned gateway bindings. For a session that has a gateway binding, that task reads the cache entry and so restarts the session's idle timer. It does not delete sessions.

`SESSION_STORE_URL` configures a separate session tracker. The list reads only the live cache, so a tracker row for an evicted session does not appear in the list. `GET /api/agent/sessions/{id}` can still return that tracker row, with `turn_count` `0` and no `title`, but a later turn on that id returns `404`, because turns read the live cache. When `SESSION_STORE_URL` is unset, there is no tracker, and `GET /api/agent/sessions/{id}` also returns `404` after eviction.

## JWT authentication

Leave JWT off for the examples above. The CLI has no option that issues a token. `caliche doctor` prints `jwt_enabled` and `require_auth`, which report the environment variables below; it does not change them.

The operator supplies tokens from their own issuer. The server reads these variables:

| Variable | Behavior |
| --- | --- |
| `JWT_ENABLED` | Accepts only `true` or `false`. Unset, `1`, `yes`, and `TRUE` all leave JWT off. |
| `JWT_PUBLIC_KEY`, `JWT_PUBLIC_KEY_PATH`, or `JWT_JWKS_URL` | One of these is required when JWT is on. The server refuses to start if all three are missing. |
| `JWT_AUDIENCE` | Required when JWT is on. A comma-separated list. An empty value stops the server at startup. There is no built-in audience. |
| `JWT_ISSUER` | Required when JWT is on. A comma-separated list. An empty value stops the server at startup. There is no built-in issuer. |
| `JWT_ALGORITHM` | Optional. The default is `RS256`. |
| `CALICHE_REQUIRE_AUTH` | `true`, `1`, `yes`, or `on` requires a bearer token on most `/api/*` routes. The server refuses to start if this is on and `JWT_ENABLED` is not `true`. |
| `JWT_TENANT_CLAIM` | Optional. The default is `tenant_id`. |
| `JWT_REQUIRED_CLAIMS` | Optional. The default is `sub`. |
| `JWT_VALIDATE_EXP` | Optional. The default is `true`. Expiry validation stays on when required authentication or the production profile is set. |

```bash
JWT_ENABLED=true \
JWT_PUBLIC_KEY_PATH=/path/to/public.pem \
JWT_AUDIENCE=operators \
JWT_ISSUER=<issuer> \
CALICHE_REQUIRE_AUTH=true \
cargo run -p caliche_cli -- --server --bind 127.0.0.1 --port 3000
```

Callers send `Authorization: Bearer <token>`. This repository does not issue tokens.

Setting `JWT_ENABLED=true` already requires a bearer token on `/api/agent/sessions`, even while `CALICHE_REQUIRE_AUTH` is `false`, because that route prefix is on the list of routes that always check a token when JWT is on.

### Sessions owned by another caller

When caller A creates a session while JWT is on, the session records A's `sub` claim (or `user_id` claim) as its owner.

Caller B has a valid token with a different `sub`:

- `GET /api/agent/sessions` returns `200` and includes A's session, because the list always contains every live session in the process.
- `GET /api/agent/sessions/{id}`, `DELETE`, `POST .../turn`, `POST .../task`, `GET .../events`, `POST .../cancel`, and `POST .../stop` on A's session return `403` with the plain text `session access denied`.
- A missing or invalid bearer token on those routes returns `401` with `authentication required` or `invalid or missing authentication token`.

A caller with the same `sub` as A can use A's session. If both the stored owner and B have a tenant claim and the values differ, the per-session check also returns `403`. If either side has no tenant claim, the tenant comparison is skipped.

The `401` and `403` responses on a single session come from the owner check. B still sees A's session in the list. The runtime, the tool sandbox, and the session map are shared within the one process.

With both `JWT_ENABLED` and `CALICHE_REQUIRE_AUTH` off, B can list A's session, run turns on it, cancel it, and delete it.

## OpenAPI description

This repository does not generate an OpenAPI document. [openapi-sessions.yaml](openapi-sessions.yaml) is a hand-written description of the routes on this page, checked against `caliche_server/src/handlers/sessions.rs` and `caliche_server/src/router/mod.rs`. It does not describe the context, memory, share, mode, plan, chat, or gateway routes, which exist but are outside the scope of this page.

## Lean profile

`--profile lean` is opt-in and does not change the default agent. Its tool list, its system prompt, and what it skips are described in [Configuration](../configuration.md#lean-profile).

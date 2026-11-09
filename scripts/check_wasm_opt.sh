#!/usr/bin/env bash
# Gate for the UI wasm-opt bake: feature flags, binaryen 117 floor, fail closed.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/caliche/lib.sh"

fail() {
  printf 'check_wasm_opt: %s\n' "$*" >&2
  exit 1
}

assert_contains() {
  local haystack="$1"
  local needle="$2"
  local label="$3"
  if [[ "$haystack" != *"$needle"* ]]; then
    printf '%s\n' "$haystack" >&2
    fail "${label}: missing [${needle}]"
  fi
}

write_fake() {
  local dir="$1"
  local body="$2"
  mkdir -p "$dir"
  printf '%s\n' '#!/bin/sh' "$body" >"${dir}/wasm-opt"
  chmod +x "${dir}/wasm-opt"
}

feature_guard() {
  cat <<'EOF'
if [ "${1:-}" = "--version" ]; then
  printf '%s\n' "wasm-opt version 120 (version_120)"
  exit 0
fi
out=""
prev=""
for arg in "$@"; do
  if [ "$prev" = "-o" ]; then
    out=$arg
  fi
  prev=$arg
done
for need in -Oz --enable-bulk-memory --enable-multivalue --enable-mutable-globals --enable-nontrapping-float-to-int --enable-reference-types --enable-sign-ext; do
  ok=0
  for arg in "$@"; do
    if [ "$arg" = "$need" ]; then
      ok=1
    fi
  done
  if [ "$ok" -ne 1 ]; then
    printf 'missing %s\n' "$need" >&2
    exit 1
  fi
done
printf 'optimized' >"$out"
exit 0
EOF
}

tmp="$(mktemp -d "${TMPDIR:-/tmp}/caliche-wasm-opt-check.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

printf '%s\n' " ${CALICHE_WASM_OPT_FEATURES[*]} " | grep -q ' --enable-bulk-memory ' \
  || fail "feature list dropped --enable-bulk-memory"

write_fake "${tmp}/v116" '
if [ "${1:-}" = "--version" ]; then
  printf "%s\n" "wasm-opt version 116 (version_116)"
  exit 0
fi
printf "should-not-run\n" >&2
exit 99
'
write_fake "${tmp}/v120" "$(feature_guard)"
write_fake "${tmp}/v120-fail" '
if [ "${1:-}" = "--version" ]; then
  printf "%s\n" "wasm-opt version 120 (version_120)"
  exit 0
fi
printf "%s\n" "[parse exception: invalid code after misc prefix: 17 (at 0:72)]" >&2
exit 1
'

ver116="$(caliche_wasm_opt_version_number "${tmp}/v116/wasm-opt")"
[[ "$ver116" == "116" ]] || fail "parsed 116 as [${ver116}]"
ver120="$(caliche_wasm_opt_version_number "${tmp}/v120/wasm-opt")"
[[ "$ver120" == "120" ]] || fail "parsed 120 as [${ver120}]"

printf 'original' >"${tmp}/old.wasm"
hash -r
set +e
old_err="$(PATH="${tmp}/v116:/usr/bin:/bin" caliche_optimize_wasm "${tmp}/old.wasm" 2>&1)"
old_status=$?
set -e
hash -r
[[ "$old_status" -eq 1 ]] || fail "binaryen 116 status ${old_status}"
[[ "$(cat "${tmp}/old.wasm")" == "original" ]] || fail "binaryen 116 replaced the module"
assert_contains "$old_err" "older than binaryen 117" "binaryen 116"
assert_contains "$old_err" "table.fill" "binaryen 116"
assert_contains "$old_err" "brew install binaryen" "binaryen 116"
if [[ "$old_err" == *"should-not-run"* ]]; then
  fail "binaryen 116 still invoked wasm-opt -Oz"
fi

printf 'original' >"${tmp}/ok.wasm"
hash -r
PATH="${tmp}/v120:/usr/bin:/bin" caliche_optimize_wasm "${tmp}/ok.wasm"
hash -r
[[ "$(cat "${tmp}/ok.wasm")" == "optimized" ]] || fail "binaryen 120 did not optimize"

printf 'original' >"${tmp}/bad.wasm"
hash -r
set +e
bad_err="$(PATH="${tmp}/v120-fail:/usr/bin:/bin" caliche_optimize_wasm "${tmp}/bad.wasm" 2>&1)"
bad_status=$?
set -e
hash -r
[[ "$bad_status" -eq 1 ]] || fail "failed opt status ${bad_status}"
[[ "$(cat "${tmp}/bad.wasm")" == "original" ]] || fail "failed opt replaced the module"
assert_contains "$bad_err" "invalid code after misc prefix: 17" "failed opt"
assert_contains "$bad_err" "exit 1" "failed opt"
assert_contains "$bad_err" "left unchanged" "failed opt"
assert_contains "$bad_err" "--enable-bulk-memory" "failed opt"
assert_contains "$bad_err" "table.fill" "failed opt"

hash -r
set +e
missing_err="$(PATH="/usr/bin:/bin" caliche_optimize_wasm "${tmp}/old.wasm" 2>&1)"
missing_status=$?
set -e
hash -r
[[ "$missing_status" -eq 1 ]] || fail "missing wasm-opt status ${missing_status}"
assert_contains "$missing_err" "missing command: wasm-opt" "missing wasm-opt"
assert_contains "$missing_err" "brew upgrade binaryen" "missing wasm-opt"

printf 'ok wasm-opt bake gate\n'

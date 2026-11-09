# Shared helpers for the Caliche ./caliche CLI. Source this file. Do not execute it.

if [[ -n "${CALICHE_LIB_LOADED:-}" ]]; then
  return 0
fi
CALICHE_LIB_LOADED=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CALICHE_SCRIPTS="${ROOT}/scripts/caliche"
cd "$ROOT"

export PATH="${HOME}/.cargo/bin:${PATH}"

if [[ "$(uname -s)" == "Darwin" ]]; then
  export CC="${CC:-/usr/bin/cc}"
  export CXX="${CXX:-/usr/bin/c++}"
fi

CALICHE_WITH_UI="${CALICHE_WITH_UI:-0}"

PY_DIR="${ROOT}/caliche_py"
JS_DIR="${ROOT}/caliche_js"
UI_DIR="${ROOT}/caliche_web"
CLI_DIR="${ROOT}/caliche_cli"
API_DIR="${ROOT}/caliche_server"
DRIPSTONE_DIR="${ROOT}/dripstone"

# binaryen 116 and older parse `table.fill` as "invalid code after misc
# prefix: 17" and ignore feature flags. 117+ accepts it when bulk-memory
# and reference-types are enabled. These flags are the wasm32-unknown-unknown
# defaults from `rustc --print cfg` on the toolchain in rust-toolchain.toml
# (bulk-memory, multivalue, mutable-globals, nontrapping-fptoint,
# reference-types, sign-ext).
CALICHE_WASM_OPT_MIN_VERSION=117
CALICHE_WASM_OPT_FEATURES=(
  --enable-bulk-memory
  --enable-multivalue
  --enable-mutable-globals
  --enable-nontrapping-float-to-int
  --enable-reference-types
  --enable-sign-ext
)

if [[ -t 1 ]]; then
  C_RED=$'\033[0;31m'
  C_GREEN=$'\033[0;32m'
  C_YELLOW=$'\033[0;33m'
  C_BLUE=$'\033[0;34m'
  C_CYAN=$'\033[0;36m'
  C_RESET=$'\033[0m'
else
  C_RED=
  C_GREEN=
  C_YELLOW=
  C_BLUE=
  C_CYAN=
  C_RESET=
fi

step() {
  printf '%s==>%s %s\n' "${C_BLUE}" "${C_RESET}" "$*"
}

ok() {
  printf '%sok %s%s\n' "${C_GREEN}" "$*" "${C_RESET}"
}

warn() {
  printf '%swarn:%s %s\n' "${C_YELLOW}" "${C_RESET}" "$*"
}

die() {
  printf '%serror:%s %s\n' "${C_RED}" "${C_RESET}" "$*" >&2
  exit 1
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    die "missing command: $1"
  fi
}

caliche_run() {
  local name="$1"
  shift
  if [[ ! -x "${CALICHE_SCRIPTS}/${name}" ]]; then
    die "internal: missing command script ${name}"
  fi
  "${CALICHE_SCRIPTS}/${name}" "$@"
}

caliche_want_help() {
  case "${1:-}" in
    -h | --help | help) return 0 ;;
    *) return 1 ;;
  esac
}

run_py_cargo() {
  if [[ "$(uname -s)" == "Darwin" ]]; then
    env RUSTFLAGS= "$@"
  else
    "$@"
  fi
}

toml_version() {
  grep '^version' "$1" | sed -n '1s/version = "\(.*\)"/\1/p'
}

npm_version() {
  grep '"version"' "$1" | sed -n '1s/.*"version": "\(.*\)".*/\1/p'
}

caliche_wasm_opt_version_number() {
  local raw num
  raw="$("$@" --version 2>&1 || true)"
  num="$(printf '%s\n' "$raw" | sed -n 's/.*[Vv]ersion[^0-9]*\([0-9][0-9]*\).*/\1/p' | sed -n '1p')"
  printf '%s\n' "$num"
}

caliche_wasm_opt_install_hint() {
  cat <<EOF
Install binaryen ${CALICHE_WASM_OPT_MIN_VERSION} or newer (provides wasm-opt).
  macOS: brew install binaryen
         brew upgrade binaryen
  https://github.com/WebAssembly/binaryen/releases
EOF
}

caliche_require_wasm_opt() {
  local bin ver
  if ! command -v wasm-opt >/dev/null 2>&1; then
    printf '%s\n' "wasm-opt is required to optimize the UI bundle." >&2
    caliche_wasm_opt_install_hint >&2
    die "missing command: wasm-opt (binaryen ${CALICHE_WASM_OPT_MIN_VERSION}+)"
  fi
  bin="$(command -v wasm-opt)"
  ver="$(caliche_wasm_opt_version_number wasm-opt)"
  if [[ -n "$ver" && "$ver" -lt "$CALICHE_WASM_OPT_MIN_VERSION" ]]; then
    printf '%s\n' \
      "wasm-opt ${ver} at ${bin} is older than binaryen ${CALICHE_WASM_OPT_MIN_VERSION}." \
      "Rust's wasm32-unknown-unknown target emits table.fill (misc prefix opcode 17, bulk-memory)." \
      "binaryen ${CALICHE_WASM_OPT_MIN_VERSION}+ parses that opcode when --enable-bulk-memory is set." \
      "Upgrade wasm-opt, then rerun ./caliche web wasm (or ./caliche rust install)." >&2
    caliche_wasm_opt_install_hint >&2
    die "wasm-opt ${ver} cannot optimize the UI bundle"
  fi
}

# Optimize one wasm module in place. On failure the input bytes stay put
# and the process exits. Callers embed this file into the CLI, so a failed
# optimizer must not continue into cargo install.
caliche_optimize_wasm() {
  local wasm="$1"
  local tmp err status ver bin
  caliche_require_wasm_opt
  tmp="$(mktemp "${wasm}.caliche-opt.XXXXXX")"
  err="$(mktemp "${TMPDIR:-/tmp}/caliche-wasm-opt-err.XXXXXX")"
  # RETURN is function-local. exit via die does not run it, so failure
  # also deletes temps explicitly below.
  trap 'rm -f "$tmp" "$err"' RETURN
  status=0
  wasm-opt -Oz "${CALICHE_WASM_OPT_FEATURES[@]}" "$wasm" -o "$tmp" 2>"$err" && status=0 || status=$?
  if [[ "$status" -eq 0 ]]; then
    mv -f "$tmp" "$wasm"
    return 0
  fi
  bin="$(command -v wasm-opt)"
  ver="$(caliche_wasm_opt_version_number wasm-opt)"
  printf '%s\n' "wasm-opt failed (exit ${status}) on ${wasm}." >&2
  if [[ -s "$err" ]]; then
    cat "$err" >&2
  fi
  printf '%s\n' \
    "Command: wasm-opt -Oz ${CALICHE_WASM_OPT_FEATURES[*]} ${wasm} -o <temp>" \
    "wasm-opt: ${bin}${ver:+ (version ${ver})}" \
    "The module was left unchanged. This bake stops so cargo install does not embed an unoptimized UI bundle." >&2
  if grep -q "invalid code after misc prefix: 17" "$err"; then
    printf '%s\n' \
      "misc prefix opcode 17 is table.fill. That parse error means this wasm-opt does not implement bulk-memory." \
      "Install binaryen ${CALICHE_WASM_OPT_MIN_VERSION} or newer." >&2
  fi
  rm -f "$tmp" "$err"
  caliche_wasm_opt_install_hint >&2
  die "wasm-opt failed on ${wasm}"
}

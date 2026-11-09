#!/usr/bin/env bash
# Fail when the dripstone default build, or a build with no default features,
# pulls a remote backend, an ONNX/model stack, an HTTP client, or the HNSW index.
set -euo pipefail

# The workspace root is two levels up from this script. Deriving it from the
# script's location works in a clone and in a source archive with no .git.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

deny=(
  reqwest
  ureq
  hf-hub
  ort
  candle-core
  candle-nn
  candle-transformers
  tokenizers
  redis
  sqlx
  qdrant-client
  aws-sdk-s3
  aws-config
  aws-sdk-sts
  aws-sdk-s3vectors
  neo4rs
  llama-cpp-2
  image
  hdf5-reader
  parquet
  pgvector
  mlx-rs
  hnsw_rs
  anndists
  hyper
  http
)

check_tree() {
  local label="$1"
  shift
  local tree count package
  tree="$(cargo tree -p dripstone "$@" --prefix none -e normal)"
  count="$(printf '%s\n' "$tree" | sed '/^$/d' | sort -u | wc -l | tr -d ' ')"
  echo "${label} dependency count: ${count}"
  for package in "${deny[@]}"; do
    if printf '%s\n' "$tree" | awk '{print $1}' | grep -qx "$package"; then
      echo "forbidden dependency ${package} in ${label}" >&2
      return 1
    fi
  done
}

check_tree "no-default-features" --no-default-features
check_tree "default"

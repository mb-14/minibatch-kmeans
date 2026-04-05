#!/usr/bin/env bash
# Setup for clustering quality evaluation: pip deps and clustering-data-v1 (suite v1.1.0).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TAG="v1.1.0"
CLUSTERING_DATA_URL="https://github.com/gagolews/clustering-data-v1.git"
DATA_DIR="$ROOT/clustering-data-v1"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

  From the repo root:
    - pip install -e ".[eval]" (see pyproject.toml optional-dependencies eval)
    - clone clustering-data-v1 at tag ${TAG} (if missing), or git checkout ${TAG} inside an existing clone

  Then run e.g.:
    python -m evaluation.run --data-path ./clustering-data-v1 --device cuda:0 -o results_eval.json

Options:
  -h, --help     Show this help.
EOF
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage 0 ;;
    *)
      echo "$(basename "$0"): unknown option: $1" >&2
      usage 1
      ;;
  esac
  shift
done

echo "Installing Python dependencies..."
pip install -e ".[eval]"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Cloning clustering-data-v1 (${TAG})..."
  git clone --depth 1 --branch "$TAG" "$CLUSTERING_DATA_URL" "$DATA_DIR"
elif [[ -d "$DATA_DIR/.git" ]]; then
  echo "Using existing ${DATA_DIR}; checking out ${TAG}..."
  if ! git -C "$DATA_DIR" checkout "$TAG" 2>/dev/null; then
    echo "warning: checkout ${TAG} failed; try: git -C ${DATA_DIR} fetch --depth 1 origin tag ${TAG} && git -C ${DATA_DIR} checkout ${TAG}" >&2
  fi
else
  echo "warning: ${DATA_DIR} exists but is not a git clone; not modifying it." >&2
fi

echo "Evaluation setup complete. Data path for --data-path: ${DATA_DIR}"

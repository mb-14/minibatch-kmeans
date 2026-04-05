#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "usage: $0 OUTPUT_PREFIX [extra args for: python -m evaluation.perf_test ...]" >&2
  echo "  Example: $0 ./profiles/run1 --device cuda:0 --n 65536 --k 1024 --d 128 --niter 1 --batch-size 1024" >&2
  exit 1
}

if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
  usage
fi
if [[ $# -lt 1 ]]; then
  usage
fi

OUT="$1"
shift


if ! command -v nsys >/dev/null 2>&1; then
  echo "profile_nsys.sh: 'nsys' not found on PATH. Install Nsight Systems / CUDA toolkit profiling tools." >&2
  exit 127
fi

nsys profile \
  -o "$OUT" \
  --trace=cuda,nvtx,osrt \
  --force-overwrite=true \
  -- \
  python -m evaluation.perf_test "$@" --method minibatchkmeans


report="${OUT}.nsys-rep"
echo "Nsight Systems report: ${report}" >&2

#!/usr/bin/env bash
# Generate static HTML API docs for MiniBatchKMeans only (pdoc).
# Documents the public package ``minibatch_kmeans``; the page is the class API
# (constructor, properties, fit / partial_fit / predict / fit_predict / save / load) with
# source hidden so only docstrings appear.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m pdoc \
  -d markdown \
  --no-show-source \
  --footer-text "minibatch-kmeans" \
  -t "$ROOT/pdoc_templates" \
  -o pdoc_html \
  minibatch_kmeans

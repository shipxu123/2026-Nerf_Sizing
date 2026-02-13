#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FALCON_DIR="${FALCON_DIR:-$ROOT_DIR/data/FALCON}"

if [[ ! -d "$FALCON_DIR" ]]; then
  echo "FALCON directory not found: $FALCON_DIR" >&2
  exit 1
fi

cd "$FALCON_DIR"
python scripts/example_mlp_minimal.py "$@"

#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/minimal.yaml}"
DATA_OUT="${DATA_OUT:-data/minimal_train_data.npz}"
NUM_SAMPLES="${NUM_SAMPLES:-0}"
DEVICE="${DEVICE:-auto}"
CHECKPOINT="${CHECKPOINT:-checkpoints/minimal/final.pt}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v python >/dev/null 2>&1; then
  echo "python not found in PATH" >&2
  exit 1
fi

gen_args=(--config "$CONFIG" --output "$DATA_OUT")
if [[ "$NUM_SAMPLES" -gt 0 ]]; then
  gen_args+=(--num-samples "$NUM_SAMPLES")
fi

echo "Generating data..."
python scripts/generate_data.py "${gen_args[@]}"

echo "Training model..."
python scripts/train.py --config "$CONFIG" --data "$DATA_OUT" --device "$DEVICE"

echo "Running optimization..."
python scripts/optimize.py --config "$CONFIG" --checkpoint "$CHECKPOINT" --device "$DEVICE"

echo "Done. Check outputs/, checkpoints/, and data/."

#!/bin/bash
set -euo pipefail

# Move to repo root
cd "$(dirname "$0")/../.."

# Ensure Python can import 'scripts'
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Run as module (preferred)
python -m scripts.check_transforms \
  --root data/augmented/train \
  --num_samples 8 \
  --policy randaugment \
  --seed 42
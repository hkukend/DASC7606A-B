#!/bin/bash
set -euo pipefail

# Move to repo root
cd "$(dirname "$0")/../.."

# Ensure Python can import 'scripts'
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Run diagnostics analyzer
python -m scripts.analyze_diagnostics "$@"

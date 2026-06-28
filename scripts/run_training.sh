#!/usr/bin/env bash
# COM668 AI-IDS — Training script
# Usage: bash scripts/run_training.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "  AI-IDS Training Pipeline"
echo "========================================"

cd "$PROJECT_ROOT"

# Activate virtualenv if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Activated .venv"
fi

# Run training
python src/train_pipeline.py "$@"

echo ""
echo "Training complete. Run the API:"
echo "  python src/api.py"

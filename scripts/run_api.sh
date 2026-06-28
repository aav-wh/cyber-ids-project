#!/usr/bin/env bash
# COM668 AI-IDS — API startup script
# Usage: bash scripts/run_api.sh [--prod]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROD=false

for arg in "$@"; do
    if [ "$arg" == "--prod" ]; then PROD=true; fi
done

cd "$PROJECT_ROOT"
if [ -d ".venv" ]; then source .venv/bin/activate; fi

echo "========================================"
echo "  AI-IDS REST API"
echo "========================================"

if $PROD; then
    echo "  Mode: Production (gunicorn, 4 workers)"
    gunicorn 'src.api:app' \
        --workers 4 \
        --bind 0.0.0.0:5000 \
        --timeout 30 \
        --access-logfile logs/access.log \
        --error-logfile logs/error.log
else
    echo "  Mode: Development (Flask built-in)"
    python src/api.py
fi

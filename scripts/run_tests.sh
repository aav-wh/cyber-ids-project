#!/usr/bin/env bash
# COM668 AI-IDS — Test runner script
# Usage: bash scripts/run_tests.sh [--coverage]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COVERAGE=false

for arg in "$@"; do
    if [ "$arg" == "--coverage" ]; then COVERAGE=true; fi
done

cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then source .venv/bin/activate; fi

echo "========================================"
echo "  AI-IDS Test Suite"
echo "========================================"

if $COVERAGE; then
    pytest tests/ -v \
        --cov=src/ids \
        --cov-report=html:htmlcov \
        --cov-report=term-missing
    echo "Coverage report: htmlcov/index.html"
else
    pytest tests/ -v
fi

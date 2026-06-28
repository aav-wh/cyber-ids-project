#!/usr/bin/env bash
# COM668 AI-IDS — Docker build and run script
# Usage: bash scripts/build_docker.sh [--clean]

set -euo pipefail

IMAGE_NAME="ai-ids"
TAG="latest"
CLEAN=false

for arg in "$@"; do
    if [ "$arg" == "--clean" ]; then CLEAN=true; fi
done

echo "========================================"
echo "  AI-IDS Docker Build"
echo "========================================"

if $CLEAN; then
    echo "Removing existing image..."
    docker rmi "${IMAGE_NAME}:${TAG}" 2>/dev/null || true
fi

echo "Building image: ${IMAGE_NAME}:${TAG}"
docker build -t "${IMAGE_NAME}:${TAG}" .

echo ""
echo "Build complete. To run:"
echo "  docker-compose up"
echo "  # or: docker run -p 5000:5000 ${IMAGE_NAME}:${TAG}"

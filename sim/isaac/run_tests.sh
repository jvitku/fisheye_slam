#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CONTAINER="isaac-pegasus"

# Ensure container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "Starting ${CONTAINER} container..."
    xhost +local:docker > /dev/null 2>&1 || true
    docker compose up -d
    sleep 3
fi

echo "=== Tier 1: PX4 SITL tests (no GPU) ==="
docker exec "$CONTAINER" python3 -m pytest /workspace/tests/test_px4_sitl.py -v

if [[ "${1:-}" == "--full" ]]; then
    echo ""
    echo "=== Tier 2: Integration tests (GPU required) ==="
    docker exec "$CONTAINER" /isaac-sim/python.sh -m pytest /workspace/tests/test_integration.py -v
fi

echo ""
echo "All requested tests completed."

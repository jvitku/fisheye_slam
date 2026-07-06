#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
xhost +local:docker > /dev/null 2>&1 || true
exec docker compose up "$@"
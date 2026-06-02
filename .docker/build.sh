#!/usr/bin/env bash
# Build the public DynamicCodec image from .docker/Dockerfile.
#
# Most users should run scripts/setup_container.sh, which calls this when the
# Tencent internal image is unreachable. This script is the explicit entry
# point if you only want to build the image.
#
# Usage:
#   bash .docker/build.sh                    # build :local tag
#   bash .docker/build.sh dynamiccodec:dev   # custom tag

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TAG="${1:-dynamiccodec:local}"

docker build -t "$TAG" -f "$HERE/Dockerfile" "$HERE"
echo "built: $TAG"

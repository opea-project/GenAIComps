#!/usr/bin/env bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# End-to-end test – Router micro-service, RouteLLM controller (CPU/Xeon)
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKPATH="$(cd "$SCRIPT_DIR/../.." && pwd)"
host=127.0.0.1
LOG_PATH="$WORKPATH/tests"
ROUTER_PORT=6000
CONTAINER=opea_router

# Required secrets
: "${HF_TOKEN:?Need HF_TOKEN}"
: "${OPENAI_API_KEY:=}"

REGISTRY_AND_REPO=${REGISTRY_AND_REPO:-opea/router}
TAG=${TAG:-latest}

export HF_TOKEN OPENAI_API_KEY REGISTRY_AND_REPO TAG

build_image() {
  cd "$WORKPATH"
  docker build --no-cache -t "${REGISTRY_AND_REPO}:${TAG}" \
    -f comps/router/src/Dockerfile .
}

start_router() {
  cd "$WORKPATH/comps/router/deployment/docker_compose"

  export CONTROLLER_TYPE=routellm

  docker compose -f compose.yaml up router_service -d
  sleep 20
}

validate() {
  # weak route
  rsp=$(
    curl -s --noproxy localhost,127.0.0.1 \
      -X POST http://${host}:${ROUTER_PORT}/v1/route \
      -H 'Content-Type: application/json' \
      -d '{"text":"What is 2 + 2?"}'
  )
  [[ $rsp == *"weak"* ]] || { echo "weak routing failed ($rsp)"; exit 1; }

  # strong route
  hard='Given a 100x100 grid where each cell is independently colored black or white such that for every cell the sum of black cells in its row, column, and both main diagonals is a distinct prime number, determine whether there exists a unique configuration of the grid that satisfies this condition and, if so, compute the total number of black cells in that configuration.'
  rsp=$(
    curl -s --noproxy localhost,127.0.0.1 \
      -X POST http://${host}:${ROUTER_PORT}/v1/route \
      -H 'Content-Type: application/json' \
      -d "{\"text\":\"$hard\"}"
  )
  [[ $rsp == *"strong"* ]] || { echo "strong routing failed ($rsp)"; exit 1; }
}

cleanup() {
  cd "$WORKPATH/comps/router/deployment/docker_compose"
  docker compose -f compose.yaml down --remove-orphans
}

trap cleanup EXIT
cleanup
build_image
start_router
validate

echo "✅ RouteLLM controller test passed."

#!/usr/bin/env bash
# End-to-end test – Router micro-service, RouteLLM controller (CPU/Xeon)
set -xeuo pipefail

WORKPATH=$(dirname "$PWD")
host_ip=$(hostname -I | awk '{print $1}')
LOG_PATH="$WORKPATH/tests"
ROUTER_PORT=6000
CONTAINER=opea_router

# Required secrets
: "${HF_TOKEN:?Need HF_TOKEN}"
: "${OPENAI_API_KEY:?Need OPENAI_API_KEY}"
: "${REGISTRY_AND_REPO:?Need REGISTRY_AND_REPO}"
: "${TAG:?Need TAG}"

build_image() {
  cd "$WORKPATH"
  docker build --no-cache -t "${REGISTRY_AND_REPO}:${TAG}" \
    -f comps/router/src/Dockerfile .
}

start_router() {
  export HF_TOKEN OPENAI_API_KEY REGISTRY_AND_REPO TAG
  unset CONTROLLER_TYPE                      
  cd "$WORKPATH/comps/router/deployment/docker_compose"
  docker compose -f compose.yaml up router_service -d
  sleep 20
}

validate() {

  # weak route
  rsp=$(curl -s http://${host_ip}:${ROUTER_PORT}/v1/route \
        -X POST -H 'Content-Type: application/json' \
        -d '{"text":"What is 2 + 2?"}')
  [[ $rsp == *"weak"* ]] || { echo "weak routing failed ($rsp)"; exit 1; }

  # strong route
  hard='Explain Gödel’s incompleteness theorem in formal terms.'
  rsp=$(curl -s http://${host_ip}:${ROUTER_PORT}/v1/route \
        -X POST -H 'Content-Type: application/json' \
        -d "{\"text\":\"$hard\"}")
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

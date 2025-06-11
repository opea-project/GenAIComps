# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/bin/bash

set -euo pipefail
set -x

WORKPATH=$(dirname "$PWD")
COMPOSE_FILE="$WORKPATH/../../GenAIComps/comps/third_parties/nebula/deployment/docker_compose/docker-compose.yaml"

stop_services() {
  echo "Stopping and cleaning up any existing Nebula services..."
  docker compose -f "$COMPOSE_FILE" down -v || true
  echo "Cleanup complete."
}

start_services() {
  echo "Starting Nebula services..."
  docker compose -f "$COMPOSE_FILE" up -d

  echo "Waiting for services to become healthy..."
  local services=("metad0" "metad1" "metad2" "storaged0" "storaged1" "storaged2" "graphd")

  for svc in "${services[@]}"; do
    local container_id
    container_id=$(docker compose -f "$COMPOSE_FILE" ps -q "$svc")
    echo -n "Waiting for $svc to be healthy"
    until [ "$(docker inspect -f '{{.State.Health.Status}}' "$container_id")" == "healthy" ]; do
      echo -n "."
      sleep 2
    done
    echo "Start services complete"
  done

  echo "All services are healthy."
}

run_query() {
  local query="$1"
  local network_name="$2"

  docker run --rm --network "$network_name" docker.io/vesoft/nebula-console:v3.5 \
    -addr graphd -port 9669 -u root -p nebula -e "$query"
}

validate_database() {
  echo "Running validation queries..."

  # Get network name from graphd container
  local container_id
  container_id=$(docker compose -f "$COMPOSE_FILE" ps -q graphd)
  local network_name
  network_name=$(docker inspect "$container_id" \
    --format '{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{end}}')

  if [ -z "$network_name" ]; then
    echo "Could not detect the Docker network from container."
    exit 1
  fi

  echo "[Step 1] Creating space..."
  run_query "CREATE SPACE my_space(partition_num=10, replica_factor=1, vid_type=FIXED_STRING(32));" "$network_name"
  echo "[Step 1] Space created successfully."
  sleep 2

  echo "[Step 2] Creating tag..."
  run_query "USE my_space; CREATE TAG person(name string, age int);" "$network_name"
  echo "[Step 2] Tag created successfully."
  sleep 2

  echo "[Step 3] Inserting data..."
  run_query "USE my_space; INSERT VERTEX person(name, age) VALUES 'person1':('Alice', 30), 'person2':('Bob', 25);" "$network_name"
  echo "[Step 3] Data inserted successfully."
  sleep 2

  echo "[Step 4] Querying data..."
  run_query "USE my_space; MATCH (p:person) RETURN p;" "$network_name"
  echo "[Step 4] Data queried successfully."

  echo "All tests successful."
}

# Main flow

stop_services
start_services
validate_database
stop_services

echo "Test script completed successfully."

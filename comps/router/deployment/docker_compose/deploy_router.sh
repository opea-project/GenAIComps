#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ========================
# OPEA Router Deploy Script
# ========================

# Load environment variables from a .env file if present
if [ -f .env ]; then
  echo "[INFO] Loading environment variables from .env"
  export $(grep -v '^#' .env | xargs)
fi

# Required variables
REQUIRED_VARS=("HF_TOKEN")

# Validate that all required variables are set
for VAR in "${REQUIRED_VARS[@]}"; do
  if [ -z "${!VAR}" ]; then
    echo "[ERROR] $VAR is not set. Please set it in your environment or .env file."
    exit 1
  fi
done

export HUGGINGFACEHUB_API_TOKEN="$HF_TOKEN"

# Default values for Docker image
REGISTRY_AND_REPO=${REGISTRY_AND_REPO:-opea/router}
TAG=${TAG:-latest}

# Export them so Docker Compose can see them
export REGISTRY_AND_REPO
export TAG

# Print summary
echo "[INFO] Starting deployment with the following config:"
echo "  Image: ${REGISTRY_AND_REPO}:${TAG}"
echo "  HF_TOKEN: ***${HF_TOKEN: -4}"
echo "  OPENAI_API_KEY: ***${OPENAI_API_KEY: -4}"
echo ""

# Compose up
echo "[INFO] Launching Docker Compose service..."
docker compose -f compose.yaml up --build

# Wait a moment then check status
sleep 2
docker ps --filter "name=opea-router"

echo "[SUCCESS] Router service deployed and running on http://localhost:6000"

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#!/bin/bash
#######################################################################
# Proxy
#######################################################################
export https_proxy=${https_proxy}
export http_proxy=${http_proxy}
export no_proxy=${no_proxy}
export your_ip=${your_ip}
################################################################
# Configure LLM Parameters based on the model selected.
################################################################

export HF_TOKEN=${HF_TOKEN}

export LLM_ID=${LLM_ID:-"HuggingFaceH4/zephyr-7b-alpha"}
export LLM_MODEL_ID=${LLM_MODEL_ID:-"HuggingFaceH4/zephyr-7b-alpha"}
export LLM_ENDPOINT_PORT=${LLM_ENDPOINT_PORT:-"9001"}

export TGI_PORT=8008
export PYTHONPATH="/home/user/"
export TGI_LLM_ENDPOINT="http://${your_ip}:${TGI_PORT}"

export NEO4J_USERNAME=${NEO4J_USERNAME:-"neo4j"}
export NEO4J_PASSWORD=${NEO4J_PASSWORD:-"neo4j_password"}
export NEO4J_URL=${NEO4J_URL:-"neo4j://localhost:7687"}
export NEO4J_URI=${NEO4J_URI:-"neo4j://localhost:7687"}
export NEO4J_PORT1={$NEO4J_PORT1:-7474}:7474
export NEO4J_PORT2={$NEO4J_PORT2:-7687}:7687
export NEO4J_AUTH=neo4j/password
export NEO4J_PLUGINS=\[\"apoc\"\]
export NEO4J_HEALTH_URL="http://localhost:7474"
export DATA_DIRECTORY=$(pwd)
export ENTITIES="PERSON,PLACE,ORGANIZATION"
export RELATIONS="HAS,PART_OF,WORKED_ON,WORKED_WITH,WORKED_AT"
export VALIDATION_SCHEMA='{
    "PERSON": ["HAS", "PART_OF", "WORKED_ON", "WORKED_WITH", "WORKED_AT"],
    "PLACE": ["HAS", "PART_OF", "WORKED_AT"],
    "ORGANIZATION": ["HAS", "PART_OF", "WORKED_WITH"]
}'
export TEXT2KG_PORT=8090

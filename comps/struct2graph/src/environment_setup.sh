# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#######################################################################
# Proxy
#######################################################################
export https_proxy=${https_proxy}
export http_proxy=${http_proxy}
export no_proxy=${no_proxy}
################################################################
# Configure LLM Parameters based on the model selected.
################################################################
export INDEX_NAME=${INDEX_NAME:-"graph_store"}
export PYTHONPATH="/home/user/"
export NEO4J_USERNAME=${NEO4J_USERNAME:-"neo4j"}
export NEO4J_PASSWORD=${NEO4J_PASSWORD:-"neo4j_password"}
export NEO4J_URL=${NEO4J_URL:-"neo4j://neo4j-apoc:7687"}
export DATA_DIRECTORY=${DATA_DIRECTORY:-data}
export FILENAME=${FILENAME:-test1.csv}
export LOAD_FORMAT=${LOAD_FORMAT:-"CSV"}

export STRUCT2GRAPH_PORT=${STRUCT2GRAPH_PORT:-8090}
export NEO4J_URL=bolt://neo4j-apoc:7687
export NEO4J_URI=bolt://neo4j-apoc:7687
export NEO4J_PORT1=7474
export NEO4J_PORT2=7687
export NEO4J_HEALTH_URL="http://localhost:7474"


export CYPHER_CSV_CMD="LOAD CSV WITH HEADERS FROM 'file:////test1.csv' AS row \
CREATE (:Person {ID: toInteger(row.ID), Name: row.Name, Age: toInteger(row.Age), City: row.City});"
export CYPHER_JSON_CMD=" \
CALL apoc.load.json("file:///test1.json") YIELD value \
UNWIND value.table AS row \
CREATE (:Person { \
          ID: row.ID, \
          Name: row.Name, \
          Age: row.Age, \
          City: row.City \
       }); \
 "

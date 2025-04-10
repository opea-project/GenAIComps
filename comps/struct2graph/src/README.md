# Struct2Graph Microservice

The Struct2Graph Microservice represents a powerful solution for transforming structured data formats like csv and json into Neo4j graph structures, serving as a crucial bridge between traditional data sources and modern graph-based systems. This process allows for enriching existing graphs, performing advanced data analysis, and constructing comprehensive knowledge graphs.
By importing structured data, users can integrate it into RAG flows, enhance querying capabilities to uncover patterns and relationships across large datasets. It's particularly useful for populating databases, creating hierarchical structures, and enabling cross-document querying. Furthermore, this approach supports data integration to provide a solid foundation for developing sophisticated graph-based applications that can exploit the rich relationships and properties inherent in graph data structures.

## Features

To convert structured data from CSV and JSON we provide the following interface -
Input:

```
{
  "input_text": "string",
  "task": "string",
  "cypher_cmd": "string"
}
```

Output: Directory with results to query.

The task can be set to the following -

1. Index - generates index based on the cypher command (Output: Generated index)
2. Query - queries the index based on the input text (Output: Directory with results to query)

## Implementation

The struct2graph microservice is able to load and query structured data through neo4j.  
The service is hosted in a docker. The mode of operation is through docker build + run or using docker compose.

## 🚀1. Start Microservice with docker run

### Install Requirements

```bash
pip install -r requirements.txt
```

### Export environment variables

```
cd comps/struct2graph/src/
source environment_setup.sh
```

OR

```
export https_proxy=${https_proxy}
export http_proxy=${http_proxy}
export no_proxy=${no_proxy}
export INDEX_NAME=${INDEX_NAME:-"graph_store"}
export PYTHONPATH="/home/user/"
export NEO4J_USERNAME=${NEO4J_USERNAME:-"neo4j"}
export NEO4J_PASSWORD=${NEO4J_PASSWORD:-"neo4j_password"}
export NEO4J_URL=${NEO4J_URL:-"neo4j://neo4j-apoc:7687"}
export DATA_DIRECTORY=${DATA_DIRECTORY:-data}
export STRUCT2GRAPH_PORT=8090
export LOAD_FORMAT="CSV" # or JSON
```

### Launch Neo4j Service

Refer to [this link](https://github.com/opea-project/GenAIComps/blob/main/comps/third_parties/neo4j/src/README.md) to start and verify the neo4j microservice.

### Verify the Neo4j Service

```bash
curl -v http://localhost:7474
```

If the Neo4j server is running correctly, the response should include an HTTP status code of 200 OK. Any other status code or an error message indicates that the server is not running or is not accessible. If the port 7474 is mapped to another port, you should change the port in the command accordingly.

### Start struct2graph Microservice with Docker

Command to build struct2graph microservice -

```bash
docker build -f Dockerfile -t opea/struct2graph:latest ../../../
```

Command to run struct2graph microservice -

```bash
docker run -i -t --net=host --ipc=host -p STRUCT2GRAPH_PORT opea/struct2graph:latest
```

The docker launches the struct2graph microservice interactively.

## 🚀2. Start Microservice with docker compose

Export environment variables as mentioned in option 1.

Command to run docker compose -

```bash
cd GenAIComps/tests/struct2graph/deployment/docker_compose

docker compose -f struct2graph-compose.yaml up
```

## 3. Validate the service using API endpoint

Example for "index" task -

```bash
curl -X POST http://localhost:$STRUCT2GRAPH_PORT/v1/struct2graph \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-d '{
  "input_text": "",
  "task": "Index",
  "cypher_cmd": "LOAD CSV WITH HEADERS FROM \'file:///$DATA_DIRECTORY/test1.csv\' AS row CREATE (:Person {ID: toInteger(row.ID), Name: row.Name, Age: toInteger(row.Age), City: row.City})"
}'
```

Example for "query" task -

```bash
curl -X POST http://localhost:$STRUCT2GRAPH_PORT/v1/struct2graph \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-d '{
  "input_text": "MATCH (p:Person {Name:\'Alice\'}) RETURN p",
  "task": "Query",
  "cypher_cmd": ""
}'
```

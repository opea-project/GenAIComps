# Retriever Microservice with Neo4J

This retrieval microservice is intended for use in GraphRAG pipeline and assumes a GraphRAGStore containing document graph, entity_info and Community Symmaries already exist. Please refer to the GenAIExamples/GraphRAG example.

Retrieval follows these steps:

- Uses similarty to find the relevant entities to the input query. Retrieval is done over the neo4j index that natively supports embeddings.
- Uses Cypher queries to retrieve the community summaries for all the communities the entities belong to.
- Generates a partial answer to the query for each community summary. This will later be used as context to generate a final query response. Please refer to [GenAIExamples/GraphRAG](https://github.com/opea-project/GenAIExamples).

## 🚀Start Microservice with Docker

### 1. Build Docker Image

```bash
cd ../../../
docker build -t opea/retriever:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/retrievers/src/Dockerfile .
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Start Neo4j VectorDB Service

```bash
docker run \
    -p 7474:7474 -p 7687:7687 \
    -v $PWD/data:/data -v $PWD/plugins:/plugins \
    --name neo4j-apoc \
    -d \
    -e NEO4J_AUTH=neo4j/password \
    -e NEO4J_PLUGINS=\[\"apoc\"\]  \
    neo4j:latest
```

### 2. Setup Environment Variables

```bash
# Set private environment settings
export host_ip=${your_hostname IP}  # local IP
export no_proxy=$no_proxy,${host_ip}  # important to add {host_ip} for containers communication
export http_proxy=${your_http_proxy}
export https_proxy=${your_http_proxy}
export NEO4J_USERNAME=${your_neo4j_username}
export NEO4J_PASSWORD=${your_neo4j_password}
export PYTHONPATH=${path_to_comps}
export OPENAI_KEY=${your_openai_api_key}  # optional, when not provided will use smaller models TGI/TEI
export HUGGINGFACEHUB_API_TOKEN=${your_hf_token}
export DATA_PATH=${host_path_to_volume_mnt}

# set additional environment settings
export EMBEDDING_MODEL_ID="BAAI/bge-base-en-v1.5"
export OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
export LLM_MODEL_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
export MAX_INPUT_TOKENS=4096
export MAX_TOTAL_TOKENS=8192
export OPENAI_LLM_MODEL="gpt-4o"
export TEI_EMBEDDER_PORT=11633
export TEI_EMBEDDING_ENDPOINT="http://${host_ip}:${TEI_EMBEDDER_PORT}"
export LLM_ENDPOINT_PORT=11634
export TGI_LLM_ENDPOINT="http://${host_ip}:${LLM_ENDPOINT_PORT}"
export TEI_EMBEDDING_ENDPOINT="http://${host_ip}:6006"
export TGI_LLM_ENDPOINT="http://${host_ip}:6005"
export NEO4J_PORT1=7474   # 11631
export NEO4J_PORT2=7687   # 11632
export NEO4J_URI="bolt://${host_ip}:${NEO4J_PORT2}"
export NEO4J_URL="bolt://${host_ip}:${NEO4J_PORT2}"
export DATAPREP_SERVICE_ENDPOINT="http://${host_ip}:6004/v1/dataprep"
export RETRIEVER_PORT=11635
export LOGFLAG=True
```

### 3. Run Docker with Docker Compose

Docker compose will start 5 microservices: retriever-neo4j-llamaindex, dataprep-neo4j-llamaindex, neo4j-apoc, tgi-gaudi-service and tei-embedding-service. Neo4j database supports embeddings natively so we do not need a separate vector store. Checkout the blog [Introducing the Property Graph Index: A Powerful New Way to Build Knowledge Graphs with LLMs](https://www.llamaindex.ai/blog/introducing-the-property-graph-index-a-powerful-new-way-to-build-knowledge-graphs-with-llms) for a better understanding of Property Graph Store and Index.

```bash
cd ../deployment/docker_compose
export service_name="retriever-neo4j"
docker compose -f compose.yaml up ${service_name} -d
```

## Invoke Microservice

### 3.1 Check Service Status

```bash
curl http://${host_ip}:7000/v1/health_check \
  -X GET \
  -H 'Content-Type: application/json'
```

### 3.2 Consume Retriever Service

If OPEN_AI_KEY is provided it will use OPENAI endpoints for LLM and Embeddings otherwise will use TGI and TEI endpoints. If a model name not provided in the request it will use the default specified by the set_env.sh script.

```bash
curl -X POST http://${host_ip}:7000/v1/retrieval \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-3.5-turbo","messages": [{"role": "user","content": "Who is John Brady and has he had any confrontations?"}]}'
```

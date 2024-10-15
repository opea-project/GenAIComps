# Retriever Microservice with Neo4J

Retrieval assumes a GraphRAGStore exists.
Retreval follows these steps:
-Perform hierarchical_leiden to identify communities in the knowledge graph
-Performs similarty to find the relevant entities to the input query
-Generates a community symmary for each community
-Generates an answer to the query for each community summary (this will later be aggregated into a single anwser)

## 🚀Start Microservice with Python

### Install Requirements

```bash
pip install -r requirements.txt
```

### Start Neo4J Server

To launch Neo4j locally, first ensure you have docker installed. Then, you can launch the database with the following docker command.

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
For the retrieval to work assumes the data exists already in the Neo4j GraphPropertyStore. You can use `/comps/data_prep/neo4j/llama-index/extract_graph_neo4j.py` to create the graph using your input documents.

### Setup Environment Variables

```bash
export no_proxy=${your_no_proxy}
export http_proxy=${your_http_proxy}
export https_proxy=${your_http_proxy}
export NEO4J_URI=${your_neo4j_url}
export NEO4J_USERNAME=${your_neo4j_username}
export NEO4J_PASSWORD=${your_neo4j_password}
```

### Start Retriever Service

```bash
python retriever_community_answers_neo4j.py
```

## 🚀Start Microservice with Docker

### Build Docker Image

```bash
cd ../../
docker build -t opea/retriever-community-answers-neo4j:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/retrievers/neo4j/llama-index/Dockerfile .
```

### Run Docker with CLI

```bash
docker run -d --name="retriever-graphRAG" -p 7000:7000 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e NEO4J_URI=${your_neo4j_host_ip}  opea/retriever-community-answers-neo4j:latest
```

## 🚀3. Consume Retriever Service

### 3.1 Check Service Status

```bash
curl http://${your_ip}:7000/v1/health_check \
  -X GET \
  -H 'Content-Type: application/json'
```

### 3.2 Consume Retriever Service

```bash
curl -X POST http://10.165.9.52:7000/v1/retrieval \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-3.5-turbo","messages": [{"role": "user","content": "Who is John Brady and has he had any confrontations?"}]}'
```

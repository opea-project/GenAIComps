# Retriever Microservice

This retriever microservice is a highly efficient search service designed for handling and retrieving embedding vectors. It operates by receiving an embedding vector as input and conducting a similarity search against vectors stored in a VectorDB database. Users must specify the VectorDB's URL and the index name, and the service searches within that index to find documents with the highest similarity to the input vector.

The service primarily utilizes similarity measures in vector space to rapidly retrieve contentually similar documents. The vector-based retrieval approach is particularly suited for handling large datasets, offering fast and accurate search results that significantly enhance the efficiency and quality of information retrieval.

Overall, this microservice provides robust backend support for applications requiring efficient similarity searches, playing a vital role in scenarios such as recommendation systems, information retrieval, or any other context where precise measurement of document similarity is crucial.

### 1.1 Build Docker Image

```bash
cd GenAIComps
docker build -t opea/retriever:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/retrievers/src/Dockerfile .
```

### 1.2 Run Docker with CLI (Option A)

#### 1.2.1 Start MariaDB Server
Please refer to this [readme](../../third_parties/mariadb/src/README.md).
You need to ingest your knowledge documents into the vector database.

#### 1.2.2 Start the retriever service
```bash
export HOST_IP=$(hostname -I | awk '{print $1}')
# If you've configured the server with the default env values then:
export MARIADB_CONNECTION_URL=mariadb+mariadbconnector://dbuser:password@${HOST_IP}$:3306/vectordb

docker run  -d --rm --name="retriever-mariadb-vector" -p 7000:7000 --ipc=host -e MARIADB_CONNECTION_URL=$MARIADB_CONNECTION_URL -e RETRIEVER_COMPONENT_NAME="OPEA_RETRIEVER_MARIADBVECTOR" opea/retriever:latest
```

### 1.3 Run with Docker Compose (Option B)

```bash
cd comps/retrievers/deployment/docker_compose
docker compose -f compose.yaml up retriever-mariadb-vector -d
```

## 🚀2. Consume Retriever Service

### 2.1 Check Service Status

```bash
curl http://${HOST_IP}:7000/v1/health_check \
  -X GET \
  -H 'Content-Type: application/json'
```

### 2.2 Consume Embedding Service

To consume the Retriever Microservice, you can generate a mock embedding vector of length 768 with Python.

```bash
export your_embedding=$(python -c "import random; embedding = [random.uniform(-1, 1) for _ in range(768)]; print(embedding)")
curl http://${HOST_IP}:7000/v1/retrieval \
  -X POST \
  -d "{\"text\":\"What is the revenue of Nike in 2023?\",\"embedding\":${your_embedding}}" \
  -H 'Content-Type: application/json'
```

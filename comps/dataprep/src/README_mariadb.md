# Dataprep Microservice with MariaDB Vector

## 🚀1. Start Microservice with Docker

### 1.1 Build Docker Image

```bash
cd GenAIComps
docker build -t opea/dataprep:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/src/Dockerfile .
```

### 1.2 Run Docker with CLI (Option A)

#### 1.2.1 Start MariaDB Server
Please refer to this [readme](../../third_parties/mariadb/src/README.md).

#### 1.2.2 Start the data preparation service
```bash

export HOST_IP=$(hostname -I | awk '{print $1}')
# If you've configured the server with the default env values then:
export MARIADB_CONNECTION_URL: mariadb+mariadbconnector://dbuser:password@${HOST_IP}$:3306/vectordb

docker run  -d --rm --name="dataprep-mariadb-vector" -p 5000:5000 --ipc=host -e MARIADB_CONNECTION_URL=$MARIADB_CONNECTION_URL -e DATAPREP_COMPONENT_NAME="OPEA_DATAPREP_MARIADBVECTOR" opea/dataprep:latest
```

### 1.3 Run with Docker Compose (Option B)

```bash
cd comps/dataprep/deployment/docker_compose
docker compose -f compose.yaml up dataprep-mariadb-vector -d
```

## 🚀2. Consume Microservice

### 2.1 Consume Upload API

Once the data preparation microservice for MariaDB Vector is started, one can use the below command to invoke the microservice to convert documents/links to embeddings and save them to the vector store.

```bash
export document="/path/to/document"
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"path":"${document}"}' \
    http://localhost:6007/v1/dataprep/ingest
```

### 2.2 Consume get API

To get the structure of the uploaded files, use the `get` API endpoint:

```bash
curl -X POST \
    -H "Content-Type: application/json" \
    http://localhost:6007/v1/dataprep/get
```

A JSON formatted response similar to the one below will follow:

```json
[
  {
    "name": "uploaded_file_1.txt",
    "id": "uploaded_file_1.txt",
    "type": "File",
    "parent": ""
  },
  {
    "name": "uploaded_file_2.txt",
    "id": "uploaded_file_2.txt",
    "type": "File",
    "parent": ""
  }
]
```

### 2.3 Consume delete API

To delete uploaded files/links, use the `delete` API endpoint.

The `file_path` is the `id` returned by the `/v1/dataprep/get` API.

```bash
# delete link
curl -X POST "http://${HOST_IP}:5000/v1/dataprep/delete"
    -H "Content-Type: application/json" \
    -d '{"file_path": "https://www.ces.tech/.txt"}'
    
# delete file
curl -X POST "http://${HOST_IP}:5000/v1/dataprep/delete"
    -H "Content-Type: application/json" \
    -d '{"file_path": "uploaded_file_1.txt"}'

# delete all files and links
curl -X POST "http://${HOST_IP}:5000/v1/dataprep/delete"
    -H "Content-Type: application/json" \
    -d '{"file_path": "all"}'
```

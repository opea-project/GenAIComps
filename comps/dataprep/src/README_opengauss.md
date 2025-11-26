# Dataprep Microservice with openGauss

## Table of contents

1. [🚀1. Start Microservice with Docker](#1-start-microservice-with-docker)
2. [🚀2. Consume Microservice](#2-consume-microservice)

## 🚀1. Start Microservice with Docker

### 1.1 Start openGauss

Please refer to this [readme](../../third_parties/opengauss/src/README.md).

### 1.2 Setup Environment Variables

```bash
export GS_CONNECTION_STRING=opengauss+psycopg2://gaussdb:openGauss@123@${your_ip}:5432/postgres
export INDEX_NAME=${your_index_name}
export TEI_EMBEDDING_ENDPOINT=${your_tei_embedding_endpoint}
export HF_TOKEN=${your_hf_api_token}
```

### 1.3 Build Docker Image

```bash
cd GenAIComps
docker build -t opea/dataprep:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/src/Dockerfile .
```

### 1.4 Run Docker with CLI (Option A)

```bash
docker run  --name="dataprep-opengauss" -p 6007:6007 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e GS_CONNECTION_STRING=$GS_CONNECTION_STRING  -e INDEX_NAME=$INDEX_NAME -e EMBED_MODEL=${EMBED_MODEL} -e TEI_EMBEDDING_ENDPOINT=$TEI_EMBEDDING_ENDPOINT -e HF_TOKEN=${HF_TOKEN} -e DATAPREP_COMPONENT_NAME="OPEA_DATAPREP_OPENGAUSS" opea/dataprep:latest
```

### 1.5 Run with Docker Compose (Option B)

```bash
cd comps/dataprep/deployment/docker_compose
docker compose -f compose.yaml up dataprep-opengauss -d
```

## 🚀2. Consume Microservice

### 2.1 Consume Upload API

Once document preparation microservice for openGauss is started, user can use below command to invoke the microservice to convert the document to embedding and save to the database.

```bash
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"path":"/path/to/document"}' \
    http://localhost:6007/v1/dataprep/ingest
```

### 2.2 Consume get API

To get uploaded file structures, use the following command:

```bash
curl -X POST \
    -H "Content-Type: application/json" \
    http://localhost:6007/v1/dataprep/get
```

Then you will get the response JSON like this:

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

To delete uploaded file/link, use the following command.

The `file_path` here should be the `id` get from `/v1/dataprep/get` API.

```bash
# delete link
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"file_path": "https://www.ces.tech/.txt"}' \
    http://localhost:6007/v1/dataprep/delete

# delete file
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"file_path": "uploaded_file_1.txt"}' \
    http://localhost:6007/v1/dataprep/delete

# delete all files and links
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"file_path": "all"}' \
    http://localhost:6007/v1/dataprep/delete
```

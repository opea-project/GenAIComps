# Dataprep Microservice with Qdrant

## Table of contents

1. [Start Microservice with Docker](#start-microservice-with-docker)
2. [Invoke Microservice](#invoke-microservice)
3. [Running in the air gapped environment](#running-in-the-air-gapped-environment)

## 🚀Start Microservice with Docker

### Start Qdrant Server

docker run -p 6333:6333 -p 6334:6334 -v ./qdrant_storage:/qdrant/storage:z qdrant/qdrant

### Setup Environment Variables

```bash
export no_proxy=${your_no_proxy}
export http_proxy=${your_http_proxy}
export https_proxy=${your_http_proxy}
export QDRANT_HOST=${host_ip}
export QDRANT_PORT=6333
export COLLECTION_NAME=${your_collection_name}
export PYTHONPATH=${path_to_comps}
```

### Build Docker Image

```bash
cd ../../../../
docker build -t opea/dataprep:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/src/Dockerfile .
```

### Run Docker with CLI

```bash
docker run -d --name="dataprep-qdrant-server" -p 6007:6007 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e DATAPREP_COMPONENT_NAME="OPEA_DATAPREP_QDRANT" opea/dataprep:latest
```

### Run Docker with Docker Compose

```bash
cd comps/dataprep/deployment/docker_compose
docker compose -f compose_qdrant.yaml up -d
```

## Invoke Microservice

Once document preparation microservice for Qdrant is started, user can use below command to invoke the microservice to convert the document to embedding and save to the database.

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./file1.txt" \
    http://localhost:6007/v1/dataprep/ingest
```

You can specify chunk_size and chunk_size by the following commands.

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./file1.txt" \
    -F "chunk_size=1500" \
    -F "chunk_overlap=100" \
    http://localhost:6007/v1/dataprep/ingest
```

We support table extraction from pdf documents. You can specify process_table and table_strategy by the following commands. "table_strategy" refers to the strategies to understand tables for table retrieval. As the setting progresses from "fast" to "hq" to "llm," the focus shifts towards deeper table understanding at the expense of processing speed. The default strategy is "fast".

Note: If you specify "table_strategy=llm", You should first start TGI Service, please refer to 1.2.1, 1.3.1 in https://github.com/opea-project/GenAIComps/tree/main/comps/llms/README.md, and then `export TGI_LLM_ENDPOINT="http://${your_ip}:8008"`.

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./your_file.pdf" \
    -F "process_table=true" \
    -F "table_strategy=hq" \
    http://localhost:6007/v1/dataprep/ingest
```

## Running in the air gapped environment

Please follow the [common guide](../README.md#running-in-the-air-gapped-environment) to run dataprep microservice in the air gapped environment.

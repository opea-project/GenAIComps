# Dataprep Microservice with Pinecone

## 🚀Start Microservice with Docker

### Start Pinecone Server

1. Create Pinecone account from the below link

https://app.pinecone.io/

More details from Pinecone quick start guide https://docs.pinecone.io/guides/get-started/quickstart

2. Get API key

API Key is needed to make the API calls. API key can get it from the Project -> Manage -> API keys

3. Create the index in https://app.pinecone.io/

Following details are to be provided

    - Index name
    - Based on the embedding model selected, following has to be provided
        a. Dimensions
        b. Metric

### Setup Environment Variables

```bash
export http_proxy=${your_http_proxy}
export https_proxy=${your_http_proxy}
export PINECONE_API_KEY=${PINECONE_API_KEY}
export PINECONE_INDEX_NAME=${PINECONE_INDEX_NAME}
```

### Build Docker Image

```bash
cd ../../../../
docker build -t opea/dataprep:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/src/Dockerfile .
```

### Run Docker with CLI

```bash
docker run -d --name="dataprep-pinecone-server" -p 6007:6007 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e DATAPREP_COMPONENT_NAME="OPEA_DATAPREP_PINECONE" opea/dataprep:latest
```

### Run Docker with Docker Compose

```bash
cd comps/dataprep/deployment/docker_compose
docker compose -f compose_pipecone.yaml up -d
```

## Invoke Microservice

Once document preparation microservice for Pinecone is started, user can use below command to invoke the microservice to convert the document to embedding and save to the database.

```bash
curl -X POST -H "Content-Type: application/json" -d '{"path":"/path/to/document"}' http://localhost:6007/v1/dataprep/ingest
```

# Dataprep Microservice with Seahorse Cloud

## 0. Prerequisites

Before using this microservice, you need a Seahorse Cloud table with an API endpoint:

1. Sign up at [Seahorse Console](https://console.seahorse.dnotitia.ai)
2. Create a table — an API endpoint (`SEAHORSE_BASE_URL`) and API key (`SEAHORSE_API_KEY`) will be issued upon creation
3. Use the issued endpoint and key as environment variables below

## Table of contents

1. [Start Microservice with Docker](#start-microservice-with-docker)
2. [Invoke Microservice](#invoke-microservice)

## 🚀 Start Microservice with Docker

### Setup Environment Variables

```bash
export SEAHORSE_BASE_URL="https://<table-uuid>.api.seahorse.dnotitia.ai"
export SEAHORSE_API_KEY="sk_xxx"
export SEAHORSE_EMBEDDING_MODE="builtin"  # or "external"
export DATAPREP_COMPONENT_NAME="OPEA_DATAPREP_SEAHORSE"
```

> **⚠️ Important**: `SEAHORSE_EMBEDDING_MODE` must be the same value for both Retriever and Dataprep services.
> If Dataprep uses `SEAHORSE_EMBEDDING_MODE="external"`, Retriever must also use `SEAHORSE_EMBEDDING_MODE="external"` and `SEAHORSE_SEARCH_MODE="dense"`.

For `external` mode, also set:

```bash
export TEI_EMBEDDING_ENDPOINT="http://${your_ip}:6060"
export HF_TOKEN=${your_huggingface_token}
```

If `TEI_EMBEDDING_ENDPOINT` is not set, Dataprep falls back to local HuggingFace embeddings.
The bundled `dataprep-seahorse` Docker Compose service forwards both `TEI_EMBEDDING_ENDPOINT`
and `HF_TOKEN` into the container, so export them before `docker compose up dataprep-seahorse`
when you want TEI-backed external embeddings.

In `external` mode, external embeddings apply to dense vectors only; sparse always uses the built-in embedding path.

### Build Docker Image

```bash
cd ../../../../
docker build -t opea/dataprep:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/src/Dockerfile .
```

### Run Docker with CLI

```bash
docker run -d --name="dataprep-seahorse-server" -p 5000:5000 --ipc=host -e SEAHORSE_BASE_URL=$SEAHORSE_BASE_URL -e SEAHORSE_API_KEY=$SEAHORSE_API_KEY -e SEAHORSE_EMBEDDING_MODE=$SEAHORSE_EMBEDDING_MODE -e TEI_EMBEDDING_ENDPOINT=$TEI_EMBEDDING_ENDPOINT -e HF_TOKEN=$HF_TOKEN -e DATAPREP_COMPONENT_NAME=$DATAPREP_COMPONENT_NAME opea/dataprep:latest
```

### Run Docker Compose Service

```bash
cd ../deployment/docker_compose
docker compose up dataprep-seahorse
```

## Invoke Microservice

### Ingest a file

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./file1.txt" \
    http://localhost:5000/v1/dataprep/ingest
```

### Ingest with custom chunk size

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./file1.txt" \
    -F "chunk_size=1500" \
    -F "chunk_overlap=100" \
    http://localhost:5000/v1/dataprep/ingest
```

### Get ingested files

```bash
curl -X POST http://localhost:5000/v1/dataprep/get
```

### Delete all files

```bash
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"file_path": "all"}' \
    http://localhost:5000/v1/dataprep/delete
```

## Embedding Modes

| Mode       | Env Var                            | Behavior                                            | TEI Required?                        |
| ---------- | ---------------------------------- | --------------------------------------------------- | ------------------------------------ |
| `builtin`  | `SEAHORSE_EMBEDDING_MODE=builtin`  | Seahorse server generates embeddings server-side    | No                                   |
| `external` | `SEAHORSE_EMBEDDING_MODE=external` | TEI or local HuggingFace model generates embeddings | No (falls back to local HuggingFace) |

> `SEAHORSE_EMBEDDING_MODE` is normalized to lowercase and trimmed at startup, so values like `Builtin`, `EXTERNAL`, or `builtin` are all accepted. Any other value falls back to the `external` code path.

> Once documents are ingested with a specific mode, do **not** switch modes without deleting all existing data first.

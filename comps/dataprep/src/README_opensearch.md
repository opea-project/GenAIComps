# Dataprep Microservice with OpenSearch

For dataprep microservice for text input, we provide here the `Langchain` framework.

## 🚀1. Start Microservice with Docker

### 1.1 Start OpenSearch Stack Server

Please refer to this [readme](../../third_parties/opensearch/src/README.md).

### 1.2 Setup Environment Variables

```bash
export your_ip=$(hostname -I | awk '{print $1}')
export EMBEDDING_MODEL_ID="BAAI/bge-base-en-v1.5"
export TEI_EMBEDDING_ENDPOINT="http://${your_ip}:6006"
export OPENSEARCH_URL="http://${your_ip}:9200"
export INDEX_NAME=${your_index_name}
export HUGGINGFACEHUB_API_TOKEN=${your_hf_api_token}
```

### 1.3 Start Embedding Service

First, you need to start a TEI service.

```bash
your_port=6006
model="BAAI/bge-base-en-v1.5"
docker run -p $your_port:80 -v ./data:/data --name tei_server -e http_proxy=$http_proxy -e https_proxy=$https_proxy --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-1.5 --model-id $model
```

Then you need to test your TEI service using the following commands:

```bash
curl localhost:$your_port/embed \
    -X POST \
    -d '{"inputs":"What is Deep Learning?"}' \
    -H 'Content-Type: application/json'
```

After checking that it works, set up environment variables.

```bash
export TEI_EMBEDDING_ENDPOINT="http://localhost:$your_port"
```

### 1.4 Build Docker Image

- Build docker image with langchain

- option 1: Start single-process version (for processing up to 10 files)

```bash
cd ../../
docker build -t opea/dataprep:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/src/Dockerfile .
```

### 1.5 Run Docker with CLI (Option A)

- option 1: Start single-process version (for processing up to 10 files)

```bash
docker run -d --name="dataprep-opensearch-server" -p 6007:6007 --runtime=runc --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e OPENSEARCH_URL=$OPENSEARCH_URL -e INDEX_NAME=$INDEX_NAME -e EMBED_MODEL=${EMBED_MODEL} -e TEI_EMBEDDING_ENDPOINT=$TEI_EMBEDDING_ENDPOINT -e HUGGINGFACEHUB_API_TOKEN=$HUGGINGFACEHUB_API_TOKEN -e DATAPREP_COMPONENT_NAME="OPEA_DATAPREP_OPENSEARCH" opea/dataprep:latest
```

### 1.6 Run with Docker Compose (Option B - deprecated, will move to genAIExample in future)

```bash
# for langchain
cd comps/dataprep/deployment/docker_compose
# common command
docker compose -f compose_opensearch.yaml up -d
```

## 🚀2. Status Microservice

```bash
docker container logs -f dataprep-opensearch-server
```

## 🚀3. Consume Microservice

### 3.1 Consume Upload API

Once document preparation microservice for OpenSearch is started, user can use below command to invoke the microservice to convert the document to embedding and save to the database.

Make sure the file path after `files=@` is correct.

- Single file upload

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

- Multiple file upload

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./file1.txt" \
    -F "files=@./file2.txt" \
    -F "files=@./file3.txt" \
    http://localhost:6007/v1/dataprep/ingest
```

- Links upload (not supported for llama_index now)

```bash
curl -X POST \
    -F 'link_list=["https://www.ces.tech/"]' \
    http://localhost:6007/v1/dataprep/ingest
```

or

```python
import requests
import json

proxies = {"http": ""}
url = "http://localhost:6007/v1/dataprep/ingest"
urls = [
    "https://towardsdatascience.com/no-gpu-no-party-fine-tune-bert-for-sentiment-analysis-with-vertex-ai-custom-jobs-d8fc410e908b?source=rss----7f60cf5620c9---4"
]
payload = {"link_list": json.dumps(urls)}

try:
    resp = requests.post(url=url, data=payload, proxies=proxies)
    print(resp.text)
    resp.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes
    print("Request successful!")
except requests.exceptions.RequestException as e:
    print("An error occurred:", e)
```

### 3.2 Consume get_file API

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

### 3.3 Consume delete_file API

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

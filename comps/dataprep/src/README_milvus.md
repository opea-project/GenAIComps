# Dataprep Microservice with Milvus

## 🚀1. Start Microservice with Docker

### 1.1 Start Milvus Server

Please refer to this [readme](../../third_parties/milvus/src/README.md).

### 1.2 Setup Environment Variables

```bash
export no_proxy=${your_no_proxy}
export http_proxy=${your_http_proxy}
export https_proxy=${your_http_proxy}
export MILVUS_HOST=${your_host_ip}
export MILVUS_PORT=19530
export COLLECTION_NAME=${your_collection_name}
export HUGGINGFACEHUB_API_TOKEN=${your_hf_api_token}
export EMBEDDING_MODEL_ID=${your_embedding_model_id}
```

### 1.3 Start TEI Embedding Service

First, start the TEI embedding server.

```bash
your_port=6010
model="BAAI/bge-base-en-v1.5"
docker run -p $your_port:80 -v ./data:/data --name tei_server -e http_proxy=$http_proxy -e https_proxy=$https_proxy --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-1.5 --model-id $model
export TEI_EMBEDDING_ENDPOINT="http://localhost:$your_port"
```

### 1.4 Build Docker Image

```bash
cd ../../..
# build dataprep milvus docker image
docker build -t opea/dataprep:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy --build-arg no_proxy=$no_proxy -f comps/dataprep/src/Dockerfile .
```

### 1.5 Run Docker with CLI (Option A)

```bash
docker run -d --name="dataprep-milvus-server" -p 6010:6010 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy -e TEI_EMBEDDING_ENDPOINT=${TEI_EMBEDDING_ENDPOINT} -e MILVUS_HOST=${MILVUS_HOST} -e HUGGINGFACEHUB_API_TOKEN=${HUGGINGFACEHUB_API_TOKEN} -e DATAPREP_COMPONENT_NAME="OPEA_DATAPREP_MILVUS" opea/dataprep:latest
```

### 1.5 Run with Docker Compose (Option B)

```bash
mkdir model
cd model
git clone https://huggingface.co/BAAI/bge-base-en-v1.5
cd ../
# Update `host_ip` and  `HUGGINGFACEHUB_API_TOKEN` in set_env.sh
. set_env.sh
docker compose -f compose_milvus.yaml up -d
```

## 🚀2. Consume Microservice

### 2.1 Consume Upload API

Once document preparation microservice for Milvus is started, user can use below command to invoke the microservice to convert the document to embedding and save to the database.

Make sure the file path after `files=@` is correct.

- Single file upload

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./file.pdf" \
    http://localhost:6010/v1/dataprep/ingest
```

You can specify chunk_size and chunk_size by the following commands. To avoid big chunks, pass a small chun_size like 500 as below (default 1500).

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./file.pdf" \
    -F "chunk_size=500" \
    -F "chunk_overlap=100" \
    http://localhost:6010/v1/dataprep/ingest
```

- Multiple file upload

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./file1.pdf" \
    -F "files=@./file2.pdf" \
    -F "files=@./file3.pdf" \
    http://localhost:6010/v1/dataprep/ingest
```

- Links upload (not supported for llama_index now)

```bash
curl -X POST \
    -F 'link_list=["https://www.ces.tech/"]' \
    http://localhost:6010/v1/dataprep/ingest
```

or

```python
import requests
import json

proxies = {"http": ""}
url = "http://localhost:6010/v1/dataprep/ingest"
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

We support table extraction from pdf documents. You can specify process_table and table_strategy by the following commands. "table_strategy" refers to the strategies to understand tables for table retrieval. As the setting progresses from "fast" to "hq" to "llm," the focus shifts towards deeper table understanding at the expense of processing speed. The default strategy is "fast".

Note: If you specify "table_strategy=llm", You should first start TGI Service, please refer to 1.2.1, 1.3.1 in https://github.com/opea-project/GenAIComps/tree/main/comps/llms/README.md, and then `export TGI_LLM_ENDPOINT="http://${your_ip}:8008"`.

```bash
curl -X POST -H "Content-Type: application/json" -d '{"path":"/home/user/doc/your_document_name","process_table":true,"table_strategy":"hq"}' http://localhost:6010/v1/dataprep
```

We support table extraction from pdf documents. You can specify process_table and table_strategy by the following commands. "table_strategy" refers to the strategies to understand tables for table retrieval. As the setting progresses from "fast" to "hq" to "llm," the focus shifts towards deeper table understanding at the expense of processing speed. The default strategy is "fast".

Note: If you specify "table_strategy=llm", You should first start TGI Service, please refer to 1.2.1, 1.3.1 in https://github.com/opea-project/GenAIComps/tree/main/comps/llms/README.md, and then `export TGI_LLM_ENDPOINT="http://${your_ip}:8008"`.

```bash
curl -X POST -H "Content-Type: application/json" -d '{"path":"/home/user/doc/your_document_name","process_table":true,"table_strategy":"hq"}' http://localhost:6010/v1/dataprep/ingest
```

### 2.2 Consume get API

To get uploaded file structures, use the following command:

```bash
curl -X POST \
    -H "Content-Type: application/json" \
    http://localhost:6010/v1/dataprep/get
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
    http://localhost:6010/v1/dataprep/delete

# delete file
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"file_path": "uploaded_file_1.txt"}' \
    http://localhost:6010/v1/dataprep/delete

# delete all files and links, will drop the entire db collection
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"file_path": "all"}' \
    http://localhost:6010/v1/dataprep/delete
```

## 🚀3. Troubleshooting

1. If you get errors from TEI Embedding Endpoint like `cannot find this task, maybe it has expired` while uploading files, try to reduce the `chunk_size` in the curl command like below (the default chunk_size=1500).

   ```bash
   curl -X POST \
       -H "Content-Type: multipart/form-data" \
       -F "files=@./file.pdf" \
       -F "chunk_size=500" \
       http://localhost:6010/v1/dataprep/ingest
   ```

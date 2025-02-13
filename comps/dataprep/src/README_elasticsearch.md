# Dataprep Microservice with Elasticsearch

## 🚀1. Start Microservice with Python（Option 1）

### 1.1 Install Requirements

```bash
pip install -r requirements.txt
```

### 1.2 Setup Environment Variables

```bash
export ES_CONNECTION_STRING=http://localhost:9200
export INDEX_NAME=${your_index_name}
```

### 1.3 Start Elasticsearch

Please refer to this [readme](../../third_parties/elasticsearch/src/README.md).

### 1.4 Start Document Preparation Microservice for Elasticsearch with Python Script

Start document preparation microservice for Elasticsearch with below command.

```bash
python prepare_doc_elastic.py
```

## 🚀2. Start Microservice with Docker (Option 2)

### 2.1 Start Elasticsearch

Please refer to this [readme](../../third_parties/elasticsearch/src/README.md).

### 2.2 Setup Environment Variables

```bash
export ES_CONNECTION_STRING=http://localhost:9200
export INDEX_NAME=${your_index_name}
```

### 2.3 Build Docker Image

```bash
cd GenAIComps
docker build -t opea/dataprep:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/src/Dockerfile .
```

### 2.4 Run Docker with CLI (Option A)

```bash
docker run  --name="dataprep-elasticsearch" -p 6011:6011 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e ES_CONNECTION_STRING=$ES_CONNECTION_STRING  -e INDEX_NAME=$INDEX_NAME -e TEI_EMBEDDING_ENDPOINT=$TEI_EMBEDDING_ENDPOINT -e HUGGINGFACEHUB_API_TOKEN=$HUGGINGFACEHUB_API_TOKEN -e DATAPREP_COMPONENT_NAME="OPEA_DATAPREP_ELASTICSEARCH" opea/dataprep:latest
```

### 2.5 Run with Docker Compose (Option B)

```bash
cd comps/dataprep/deployment/docker_compose/
docker compose -f compose_elasticsearch.yaml up -d
```

## 🚀3. Consume Microservice

### 3.1 Consume Upload API

Once document preparation microservice for Elasticsearch is started, user can use below command to invoke the
microservice to convert the document to embedding and save to the database.

```bash
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"path":"/path/to/document"}' \
    http://localhost:6011/v1/dataprep/ingest
```

### 3.2 Consume get API

To get uploaded file structures, use the following command:

```bash
curl -X POST \
    -H "Content-Type: application/json" \
    http://localhost:6011/v1/dataprep/get
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

### 4.3 Consume delete API

To delete uploaded file/link, use the following command.

The `file_path` here should be the `id` get from `/v1/dataprep/get` API.

```bash
# delete link
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"file_path": "https://www.ces.tech/.txt"}' \
    http://localhost:6011/v1/dataprep/delete

# delete file
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"file_path": "uploaded_file_1.txt"}' \
    http://localhost:6011/v1/dataprep/delete

# delete all files and links
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"file_path": "all"}' \
    http://localhost:6011/v1/dataprep/delete
```

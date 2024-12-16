# Dataprep Microservice with ArangoDB

## 🚀Start Microservice with Python

### Install Requirements

```bash
pip install -r requirements.txt
apt-get install libtesseract-dev -y
apt-get install poppler-utils -y
```

### Start ArangoDB Server

To launch ArangoDB locally, first ensure you have docker installed. Then, you can launch the database with the following docker command.

```bash
docker run -d --name arangodb -p 8529:8529 -e ARANGO_ROOT_PASSWORD=password arangodb/arangodb:latest
```

### Setup Environment Variables

```bash
export no_proxy=${your_no_proxy}
export http_proxy=${your_http_proxy}
export https_proxy=${your_http_proxy}
export ARANGO_URL=${your_arango_url}
export ARANGO_USERNAME=${your_arango_username}
export ARANGO_PASSWORD=${your_arango_password}
export ARANGO_DB_NAME=${your_db_name}
export PYTHONPATH=${path_to_comps}
```

### Start Document Preparation Microservice for ArangoDB with Python Script

Start document preparation microservice for ArangoDB with below command.

```bash
python prepare_doc_arango.py
```

## 🚀Start Microservice with Docker

### Build Docker Image

```bash
cd ../../../../
docker build -t opea/dataprep-arango:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/arango/langchain/Dockerfile .
```

### Run Docker with CLI

```bash
docker run -d --name="dataprep-arango-server" -p 6007:6007 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy opea/dataprep-arango:latest
```

### Run Docker with Docker Compose

```bash
cd comps/dataprep/arango/langchain
docker compose -f docker-compose-dataprep-arango.yaml up -d
```

## Invoke Microservice

Once document preparation microservice for ArangoDB is started, user can use below command to invoke the microservice to convert the document to embedding and save to the database.

After the service is complete a Graph is created in ArangoDB. The default graph name is `NewGraph`, you can specify the graph name by `-F "graph_name=${your_graph_name}"` in the curl command.

By default, the microservice will create embeddings for the documents. You can specify `-F "create_embeddings=false"` to skip the embedding creation.

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./file1.txt" \
    -F "graph_name=${your_graph_name}" \
    http://localhost:6007/v1/dataprep
```

You can specify chunk_size and chunk_size by the following commands.

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./file1.txt" \
    -F "chunk_size=1500" \
    -F "chunk_overlap=100" \
    -F "graph_name=${your_graph_name}" \
    http://localhost:6007/v1/dataprep
```

Additional options that can be specified from the environment variables are as follows (default values are in the config.py file):

OpenAI Configuration:
- `OPENAI_EMBED_MODEL`: The embedding model to use for the OpenAI service.
- `OPENAI_EMBED_DIMENSIONS`: The dimensions for the OpenAI service.

ArangoDB Configuration:
- `USE_ONE_ENTITY_COLLECTION`: If set to True, the microservice will use a single entity collection for all nodes. If set to False, the microservice will use a separate collection for each node type.
- `INSERT_ASYNC`: If set to True, the microservice will insert the data asynchronously.
- `ARANGO_BATCH_SIZE`: The batch size for the microservice to insert the data.
- `INCLUDE_SOURCE`: If set to True, the microservice will include the source in the data.

We support table extraction from pdf documents. You can specify process_table and table_strategy by the following commands. "table_strategy" refers to the strategies to understand tables for table retrieval. As the setting progresses from "fast" to "hq" to "llm," the focus shifts towards deeper table understanding at the expense of processing speed. The default strategy is "fast".

Note: If you specify "table_strategy=llm", You should first start TGI Service, please refer to 1.2.1, 1.3.1 in https://github.com/opea-project/GenAIComps/tree/main/comps/llms/README.md, and then `export TGI_LLM_ENDPOINT="http://${your_ip}:8008"`.

For ensure the quality and comprehensiveness of the extracted entities, we recommend to use `gpt-4o` as the default model for parsing the document. To enable the openai service, please `export OPENAI_API_KEY=xxxx` before using this services.

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./your_file.pdf" \
    -F "process_table=true" \
    -F "table_strategy=hq" \
    -F "graph_name=${your_graph_name}" \
    http://localhost:6007/v1/dataprep
```

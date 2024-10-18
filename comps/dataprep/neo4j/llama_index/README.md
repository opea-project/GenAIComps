# Dataprep Microservice with Neo4J

## 🚀Start Microservice with Python

### Install Requirements

```bash
pip install -r requirements.txt
apt-get install libtesseract-dev -y
apt-get install poppler-utils -y
```

### Start Neo4J Server

To launch Neo4j locally, first ensure you have docker installed. Then, you can launch the database with the following docker command. Substitute with the password you want to set

```bash
docker run \
    -p 7474:7474 -p 7687:7687 \
    -v $PWD/data:/data -v $PWD/plugins:/plugins \
    --name neo4j-apoc \
    -d \
    -e NEO4J_AUTH=neo4j/password \
    -e NEO4J_PLUGINS=\[\"apoc\"\]  \
    neo4j:latest
```

### Setup Environment Variables

```bash
#Manually set private environment settings
export host_ip=${your_hostname IP} #local IP
export no_proxy=$no_proxy,${host_ip} #important to add {host_ip} for containers communication
export http_proxy=${your_http_proxy}
export https_proxy=${your_http_proxy}
export NEO4J_URI=${your_neo4j_url}
export NEO4J_USERNAME=${your_neo4j_username}
export NEO4J_PASSWORD=${your_neo4j_password}
export PYTHONPATH=${path_to_comps}
export OPENAI_KEY=${your_openai_api_key} #optional, when not provided will use smaller models TGI/TEI
export HUGGINGFACEHUB_API_TOKEN=${your_hf_token}
#set additional environment settings
source ./set_env.sh
```

### Start Document Preparation Microservice for Neo4J with Python Script

Start document preparation microservice for Neo4J with below command.

```bash
python extract_graph_neo4j.py
```

## 🚀Start Microservice with Docker

### Build Docker Image

```bash
cd ../../../../
docker build -t opea/dataprep-neo4j-llama_index:latest --build-arg no_proxy=$no_proxy --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/neo4j/llama-index/Dockerfile .
```

### Run Docker with CLI

```bash
docker run -d --name="dataprep-neo4j-server" -p 6004:6004 --ipc=host -e no_proxy=$no_proxy -e http_proxy=$http_proxy -e https_proxy=$https_proxy opea/dataprep-neo4j-llama_index:latest
```

### Setup Environment Variables

```bash
#Set private environment settings
export host_ip=${your_hostname IP} #local IP
export no_proxy=$no_proxy,${host_ip} #important to add {host_ip} for containers communication
export http_proxy=${your_http_proxy}
export https_proxy=${your_http_proxy}
export NEO4J_URI=${your_neo4j_url}
export NEO4J_USERNAME=${your_neo4j_username}
export NEO4J_PASSWORD=${your_neo4j_password}
export PYTHONPATH=${path_to_comps}
export OPENAI_KEY=${your_openai_api_key} #optional, when not provided will use smaller models TGI/TEI
export HUGGINGFACEHUB_API_TOKEN=${your_hf_token}
#set additional environment settings
source ./set_env.sh
```

### Run Docker with Docker Compose

```bash
cd comps/dataprep/neo4j/llama-index
docker compose -f compose.yaml up -d
```

## Invoke Microservice

Once document preparation microservice for Neo4J is started, user can use below command to invoke the microservice to convert the document to embedding and save to the database.

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./file1.txt" \
    http://localhost:6004/v1/dataprep
```

You can specify chunk_size and chunk_size by the following commands.

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./file1.txt" \
    -F "chunk_size=1500" \
    -F "chunk_overlap=100" \
    http://localhost:6004/v1/dataprep
```

We support table extraction from pdf documents. You can specify process_table and table_strategy by the following commands. "table_strategy" refers to the strategies to understand tables for table retrieval. As the setting progresses from "fast" to "hq" to "llm," the focus shifts towards deeper table understanding at the expense of processing speed. The default strategy is "fast".

Note: If you specify "table_strategy=llm", You should first start TGI Service, please refer to 1.2.1, 1.3.1 in https://github.com/opea-project/GenAIComps/tree/main/comps/llms/README.md, and then `export TGI_LLM_ENDPOINT="http://${your_ip}:8008"`.

For ensure the quality and comprehensiveness of the extracted entities, we recommend to use `gpt-4o` as the default model for parsing the document. To enable the openai service, please `export OPENAI_KEY=xxxx` before using this services.

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./your_file.pdf" \
    -F "process_table=true" \
    -F "table_strategy=hq" \
    http://localhost:6004/v1/dataprep
```

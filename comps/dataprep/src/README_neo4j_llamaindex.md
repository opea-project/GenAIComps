# Dataprep Microservice with Neo4J

This Dataprep microservice performs:

- Graph extraction (entities, relationships and descriptions) using LLM
- Performs hierarchical_leiden clustering to identify communities in the knowledge graph
- Generates a community symmary for each community
- Stores all of the above in Neo4j Graph DB

This microservice follows the graphRAG approached defined by Microsoft paper ["From Local to Global: A Graph RAG Approach to Query-Focused Summarization"](https://www.microsoft.com/en-us/research/publication/from-local-to-global-a-graph-rag-approach-to-query-focused-summarization/) with some differences such as: 1) no node degree prioritization is used in populating the LLM context window for community summaries, 2) no ranking of sub-communities is applied in generating higher level communities summaries.

This dataprep microservice ingests the input files and uses LLM (TGI, VLLM or OpenAI model when OPENAI_API_KEY is set) to extract entities, relationships and descriptions of those to build a graph-based text index. Compose yaml file deploys TGI but works also with vLLM inference endpoint.

## Setup Environment Variables

```bash
# Manually set private environment settings
export host_ip=${your_hostname IP}  # local IP
export no_proxy=$no_proxy,${host_ip}  # important to add {host_ip} for containers communication
export http_proxy=${your_http_proxy}
export https_proxy=${your_http_proxy}
export NEO4J_USERNAME=${your_neo4j_username}
export NEO4J_PASSWORD=${your_neo4j_password}  # should match what was used in NEO4J_AUTH when running the neo4j-apoc
export PYTHONPATH=${path_to_comps}
export OPENAI_KEY=${your_openai_api_key}  # optional, when not provided will use open models TGI/TEI
export HUGGINGFACEHUB_API_TOKEN=${your_hf_token}
export DATA_PATH=${host_path_to_volume_mnt}

# set additional environment settings
export EMBEDDING_MODEL_ID="BAAI/bge-base-en-v1.5"
export EMBED_MODEL=${EMBEDDING_MODEL_ID}
export OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
export LLM_MODEL_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
export MAX_INPUT_TOKENS=4096
export MAX_TOTAL_TOKENS=8192
export OPENAI_LLM_MODEL="gpt-4o"
export TEI_EMBEDDER_PORT=11633
export TEI_EMBEDDING_ENDPOINT="http://${host_ip}:${TEI_EMBEDDER_PORT}"
export LLM_ENDPOINT_PORT=11634
export TGI_LLM_ENDPOINT="http://${host_ip}:${LLM_ENDPOINT_PORT}"
export NEO4J_AUTH="${NEO4J_USERNAME}/${NEO4J_PASSWORD}"
export NEO4J_PORT1=7474   # 11631
export NEO4J_PORT2=7687   # 11632
export NEO4J_URI="bolt://${host_ip}:${NEO4J_PORT2}"
export NEO4J_URL="bolt://${host_ip}:${NEO4J_PORT2}"
export DATAPREP_SERVICE_ENDPOINT="http://${host_ip}:6004/v1/dataprep"
export LOGFLAG=True
```

## 🚀Start Microservice with Docker

### 1. Build Docker Image

```bash
cd ../../../../
docker build -t opea/dataprep-neo4j-llamaindex:latest --build-arg no_proxy=$no_proxy --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/src/Dockerfile .
```

### 2. Setup Environment Variables

```bash
# Set private environment settings
export host_ip=${your_hostname IP}  # local IP
export no_proxy=$no_proxy,${host_ip}  # important to add {host_ip} for containers communication
export http_proxy=${your_http_proxy}
export https_proxy=${your_http_proxy}
export NEO4J_URI=${your_neo4j_url}
export NEO4J_USERNAME=${your_neo4j_username}
export NEO4J_PASSWORD=${your_neo4j_password}
export PYTHONPATH=${path_to_comps}
export OPENAI_KEY=${your_openai_api_key}  # optional, when not provided will use smaller models TGI/TEI
export HUGGINGFACEHUB_API_TOKEN=${your_hf_token}
# set additional environment settings
source ./set_env.sh
```

### 3. Run Docker with Docker Compose

Docker compose will start 4 microservices: dataprep-neo4j-llamaindex, neo4j-apoc, tgi-gaudi-service and tei-embedding-service. The reason TGI and TEI are needed is because dataprep relies on LLM to extract entities and relationships from text to build the graph and Neo4j Property Graph Index. Neo4j database supports embeddings natively so we do not need a separate vector store. Checkout the blog [Introducing the Property Graph Index: A Powerful New Way to Build Knowledge Graphs with LLMs](https://www.llamaindex.ai/blog/introducing-the-property-graph-index-a-powerful-new-way-to-build-knowledge-graphs-with-llms) for a better understanding of Property Graph Store and Index.

```bash
cd comps/dataprep/deployment/docker_compose
docker compose -f compose.yaml up -d
```

## Invoke Microservice

Once document preparation microservice for Neo4J is started, user can use below command to invoke the microservice to convert the document to embedding and save to the database.

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./file1.txt" \
    http://${host_ip}:6004/v1/dataprep/ingest
```

You can specify chunk_size and chunk_size by the following commands.

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./file1.txt" \
    -F "chunk_size=1500" \
    -F "chunk_overlap=100" \
    http://${host_ip}:6004/v1/dataprep/ingest
```

Please note that clustering of extracted entities and summarization happens in this data preparation step. The result of this is:

- Large processing time for large dataset. An LLM call is done to summarize each cluster which may result in large volume of LLM calls
- Need to clean graph GB entity_info and Cluster if dataprep is run multiple times since the resulting cluster numbering will differ between consecutive calls and will corrupt the results.

We support table extraction from pdf documents. You can specify process_table and table_strategy by the following commands. "table_strategy" refers to the strategies to understand tables for table retrieval. As the setting progresses from "fast" to "hq" to "llm," the focus shifts towards deeper table understanding at the expense of processing speed. The default strategy is "fast".

Note: If you specify "table_strategy=llm" TGI service will be used.

For ensure the quality and comprehensiveness of the extracted entities, we recommend to use `gpt-4o` as the default model for parsing the document. To enable the openai service, please `export OPENAI_KEY=xxxx` before using this services.

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./your_file.pdf" \
    -F "process_table=true" \
    -F "table_strategy=hq" \
    http://localhost:6004/v1/dataprep/ingest
```

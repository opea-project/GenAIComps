# Retriever Microservice with ArangoDB

## 🚀Start Microservice with Docker

### Start ArangoDB Server

To launch ArangoDB locally, first ensure you have docker installed. Then, you can launch the database with the following docker command.

```bash
docker run -d -p 8529:8529 -e ARANGO_ROOT_PASSWORD=${ARANGO_ROOT_PASSWORD} arangodb/arangodb:latest
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
export TEI_EMBEDDING_ENDPOINT=${your_tei_embedding_endpoint}
export HUGGINGFACEHUB_API_TOKEN=${your_huggingface_api_token}
```

### Build Docker Image

```bash
cd ~/GenAIComps/
docker build -t opea/retriever:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/retrievers/src/Dockerfile .
```

### Run via CLI

```bash
docker run -d --name="retriever-arango-server" -p 7000:7000 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e ARANGODB_URL="http://localhost:8529"  opea/retriever:latest -e RETRIEVER_COMPONENT_NAME="OPEA_RETRIEVER_ARANGODB"
```

### Run Docker with Docker Compose

```bash
cd ~/GenAIComps/comps/retrievers/deployment/docker_compose/
docker compose up retriever-arangodb -d
```

See below for additional environment variables that can be set.

## 🚀3. Consume Retriever Service

```bash
curl http://${your_ip}:7000/v1/health_check \
  -X GET \
  -H 'Content-Type: application/json'
```

### 3.2 Consume Embedding Service

To consume the Retriever Microservice, you can generate a mock embedding vector of length 768 with Python.

```bash
export your_embedding=$(python -c "import random; embedding = [random.uniform(-1, 1) for _ in range(768)]; print(embedding)")
curl http://${your_ip}:7000/v1/retrieval \
  -X POST \
  -d "{\"input\":\"What is the revenue of Nike in 2023?\",\"embedding\":${your_embedding}}" \
  -H 'Content-Type: application/json'
```

```bash
export your_embedding=$(python -c "import random; embedding = [random.uniform(-1, 1) for _ in range(768)]; print(embedding)")
curl http://localhost:7000/v1/retrieval \
  -X POST \
  -d "{\"input\":\"What is the revenue of Nike in 2023?\",\"embedding\":${your_embedding},\"search_type\":\"similarity\", \"k\":4}" \
  -H 'Content-Type: application/json'
```

```bash
export your_embedding=$(python -c "import random; embedding = [random.uniform(-1, 1) for _ in range(768)]; print(embedding)")
curl http://localhost:7000/v1/retrieval \
  -X POST \
  -d "{\"input\":\"What is the revenue of Nike in 2023?\",\"embedding\":${your_embedding},\"search_type\":\"similarity_distance_threshold\", \"k\":4, \"distance_threshold\":1.0}" \
  -H 'Content-Type: application/json'
```

```bash
export your_embedding=$(python -c "import random; embedding = [random.uniform(-1, 1) for _ in range(768)]; print(embedding)")
curl http://localhost:7000/v1/retrieval \
  -X POST \
  -d "{\"input\":\"What is the revenue of Nike in 2023?\",\"embedding\":${your_embedding},\"search_type\":\"similarity_score_threshold\", \"k\":4, \"score_threshold\":0.2}" \
  -H 'Content-Type: application/json'
```

```bash
export your_embedding=$(python -c "import random; embedding = [random.uniform(-1, 1) for _ in range(768)]; print(embedding)")
curl http://localhost:7000/v1/retrieval \
  -X POST \
  -d "{\"input\":\"What is the revenue of Nike in 2023?\",\"embedding\":${your_embedding},\"search_type\":\"mmr\", \"k\":4, \"fetch_k\":20, \"lambda_mult\":0.5}" \
  -H 'Content-Type: application/json'
```

---

Additional options that can be specified from the environment variables are as follows (default values are in the `config.py` file):

ArangoDB Connection configuration

- `ARANGO_URL`: The URL for the ArangoDB service.
- `ARANGO_USERNAME`: The username for the ArangoDB service.
- `ARANGO_PASSWORD`: The password for the ArangoDB service.
- `ARANGO_DB_NAME`: The name of the database to use for the ArangoDB service.

ArangoDB Vector configuration

- `ARANGO_GRAPH_NAME`: The name of the graph to use for the ArangoDB service. Defaults to `GRAPH`.
- `ARANGO_DISTANCE_STRATEGY`: The distance strategy to use for the ArangoDB service. Defaults to `COSINE`. Other option could be `"EUCLIDEAN_DISTANCE"`.
- `ARANGO_USE_APPROX_SEARCH`: If set to True, the microservice will use the approximate nearest neighbor search for as part of the retrieval step. Defaults to `False`, which means the microservice will use the exact search.
- `ARANGO_NUM_CENTROIDS`: The number of centroids to use for the approximate nearest neighbor search. Defaults to `1`.
- `ARANGO_SEARCH_START`: The starting point for the search. Defaults to `node`. Other option could be `"edge"`, or `"chunk"`.

ArangoDB Traversal configuration

- `ARANGO_TRAVERSAL_ENABLED`: If set to True, the microservice will perform a traversal of the graph on the documents matched by similarity and return additional context (i.e nodes, edges, or chunks) from the graph. Defaults to `False`. See the `fetch_neighborhoods` method in the `arangodb.py` file for more details.
- `ARANGO_TRAVERSAL_MAX_DEPTH`: The maximum depth for the traversal. Defaults to `1`.
- `ARANGO_TRAVERSAL_MAX_RETURNED`: The maximum number of nodes/edges/chunks to return per matched document from the traversal. Defaults to `3`.
- `ARANGO_TRAVERSAL_SCORE_THRESHOLD`: The score threshold for the traversal. Defaults to `0.5`.
- `ARANGO_TRAVERSAL_QUERY`: An optional query to define custom traversal logic. This can be used to specify a custom traversal query for the ArangoDB service. If not set, the default traversal logic will be used. See the `fetch_neighborhoods` method in the `arangodb.py` file for more details.

Embedding configuration

- `TEI_EMBEDDING_ENDPOINT`: The endpoint for the TEI service.
- `TEI_EMBED_MODEL`: The model to use for the TEI service. Defaults to `BAAI/bge-base-en-v1.5`.
- `HUGGINGFACEHUB_API_TOKEN`: The API token for Hugging Face access.

Summarizer Configuration

- `SUMMARIZER_ENABLED`: If set to True, the microservice will apply summarization after retrieval. Defaults to `False`. Requires the `VLLM` service to be running or a valid `OPENAI_API_KEY` to be set. See the `VLLM Configuration` section or the `OpenAI Configuration` section below.

vLLM Configuration

- `VLLM_API_KEY`: The API key for the vLLM service. Defaults to `"EMPTY"`.
- `VLLM_ENDPOINT`: The endpoint for the VLLM service. Defaults to `http://localhost:80`.
- `VLLM_MODEL_ID`: The model ID for the VLLM service. Defaults to `Intel/neural-chat-7b-v3-3`.
- `VLLM_MAX_NEW_TOKENS`: The maximum number of new tokens to generate. Defaults to `512`.
- `VLLM_TOP_P`: If set to < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation. Defaults to `0.9`.
- `VLLM_TEMPERATURE`: The temperature for the sampling. Defaults to `0.8`.
- `VLLM_TIMEOUT`: The timeout for the VLLM service. Defaults to `600`.

OpenAI Configuration:
**Note**: This configuration can replace the VLLM and TEI services for text generation and embeddings.

- `OPENAI_API_KEY`: The API key for the OpenAI service. If not set, the microservice will not use the OpenAI service.
- `OPENAI_CHAT_MODEL`: The chat model to use for the OpenAI service. Defaults to `gpt-4o`.
- `OPENAI_CHAT_TEMPERATURE`: The temperature for the OpenAI service. Defaults to `0`.
- `OPENAI_EMBED_MODEL`: The embedding model to use for the OpenAI service. Defaults to `text-embedding-3-small`.
- `OPENAI_EMBED_DIMENSION`: The embedding dimension for the OpenAI service. Defaults to `768`.
- `OPENAI_CHAT_ENABLED`: If set to True, the microservice will use the OpenAI service for text generation, as long as `OPENAI_API_KEY` is also set. Defaults to `True`.
- `OPENAI_EMBED_ENABLED`: If set to True, the microservice will use the OpenAI service for text embeddings, as long as `OPENAI_API_KEY` is also set. Defaults to `True`.`

Some of these parameters are also available via parameters in the API call. If set, these will override the equivalent environment variables:

```python
class RetrievalRequest(BaseModel): ...


class RetrievalRequestArangoDB(RetrievalRequest):
    graph_name: str | None = None
    search_start: str | None = None  # "node", "edge", "chunk"
    num_centroids: int | None = None
    distance_strategy: str | None = None  #  # "COSINE", "EUCLIDEAN_DISTANCE"
    use_approx_search: bool | None = None
    enable_traversal: bool | None = None
    enable_summarizer: bool | None = None
    traversal_max_depth: int | None = None
    traversal_max_returned: int | None = None
    traversal_score_threshold: float | None = None
    traversal_query: str | None = None
```

See the `comps/cores/proto/api_protocol.py` file for more details on the API request and response models.

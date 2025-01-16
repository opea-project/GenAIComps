# Web Retriever Microservice

The Web Retriever Microservice is designed to efficiently search web pages relevant to the prompt, save them into the VectorDB, and retrieve the matched documents with the highest similarity. The retrieved documents will be used as context in the prompt to LLMs. Different from the normal RAG process, a web retriever can leverage advanced search engines for more diverse demands, such as real-time news, verifiable sources, and diverse sources.

## Start Microservice with Docker

### Build Docker Image

```bash
cd ../../../../
docker build -t opea/web-retriever:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/web_retrievers/src/Dockerfile .
```

### Start TEI Service

```bash
model=BAAI/bge-base-en-v1.5
volume=$PWD/data
docker run -d -p 6060:80 -v $volume:/data -e http_proxy=$http_proxy -e https_proxy=$https_proxy --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-1.5 --model-id $model --auto-truncate
```

### Start Web Retriever Service

```bash
# set TEI endpoint
export TEI_EMBEDDING_ENDPOINT="http://${your_ip}:6060"

# set search engine env variables
export GOOGLE_API_KEY=xxx
export GOOGLE_CSE_ID=xxx
```

```bash
docker run -d --name="web-retriever-server" -p 7077:7077 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy -e TEI_EMBEDDING_ENDPOINT=$TEI_EMBEDDING_ENDPOINT -e GOOGLE_API_KEY=$GOOGLE_API_KEY -e GOOGLE_CSE_ID=$GOOGLE_CSE_ID opea/web-retriever:latest
```

### Consume Web Retriever Service

To consume the Web Retriever Microservice, you can generate a mock embedding vector of length 768 with Python.

```bash
# Test
your_embedding=$(python -c "import random; embedding = [random.uniform(-1, 1) for _ in range(768)]; print(embedding)")

http_proxy= curl http://${your_ip}:7077/v1/web_retrieval \
  -X POST \
  -d "{\"text\":\"What is The Game of the Year 2024?\",\"embedding\":${your_embedding},\"k\":4}" \
  -H 'Content-Type: application/json'
```

## MCP Web Retriever Microservice (Experimental)

We also provide a simple web retriever server and a client based on the Model-Context-Protocol protocol. Please refer to [mcp](https://modelcontextprotocol.io/quickstart/server) for details.

### Build MCP Google Search web retriever image

```bash
cd ../../..
docker build --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -t opea/web-retrievers-mcp:latest -f comps/web_retrievers/src/Dockerfile.mcp .
```

### Start the services

```bash
export host_ip=$(hostname -I | awk '{print $1}')

export OLLAMA_MODEL=qwen2.5-coder
export GOOGLE_API_KEY=$GOOGLE_API_KEY
export GOOGLE_CSE_ID=$GOOGLE_CSE_ID
export OLLAMA_ENDPOINT=http://${host_ip}:11434

systemctl stop ollama.service # docker will start ollama instead, make sure there are no port conflicts
cd comps/web_retrievers/deployment/docker_compose
docker compose -f compose_web_retrievers_mcp.yaml up -d
```

### Run the client

```bash
docker exec -it web-retrievers-mcp bash

python mcp_google_search_client.py mcp_google_search_server.py

# Output log will be like following
# USER_AGENT environment variable not set, consider setting it to identify your requests.

# Connected to server with tools: ['get-google-search-answer']

# MCP Client Started!
# Type your queries or 'quit' to exit.

# Query: search some latest sports news
# ...
# ...
```

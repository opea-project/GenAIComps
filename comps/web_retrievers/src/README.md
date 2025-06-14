# Web Retriever Microservice

The Web Retriever Microservice is designed to efficiently search web pages relevant to the prompt, save them into the VectorDB, and retrieve the matched documents with the highest similarity. The retrieved documents will be used as context in the prompt to LLMs. Different from the normal RAG process, a web retriever can leverage advanced search engines for more diverse demands, such as real-time news, verifiable sources, and diverse sources.

## 🚀1. Start Microservice with Docker (Option 1)

### 1.1 Build Docker Image

```bash
cd ../../../../
docker build -t opea/web-retriever:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/web_retrievers/src/Dockerfile .
```

### 1.2 Start TEI Service

```bash
model=BAAI/bge-base-en-v1.5
volume=$PWD/data
docker run -d -p 6060:80 -v $volume:/data -e http_proxy=$http_proxy -e https_proxy=$https_proxy --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-1.6 --model-id $model --auto-truncate
```

### 1.3 Start Web Retriever Service

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

## 🚀2. Start Microservice with Docker Compose (Option 2)

Alternatively, you can start the web retriever microservice with Docker Compose.

```bash
export host_ip=$(hostname -I | awk '{print $1}')
export HF_TOKEN=${HF_TOKEN}
export GOOGLE_API_KEY=${GOOGLE_API_KEY}
export GOOGLE_CSE_ID=${GOOGLE_CSE_ID}
export TEI_PORT=6060
export no_proxy=$host_ip,$no_proxy
export EMBEDDING_MODEL_ID=BAAI/bge-base-en-v1.5
export TEI_EMBEDDING_ENDPOINT=http://${host_ip}:6060

docker compose -f ../deployment/docker_compose/compose.yaml up web-retriever-service tei-embedding-service -d
```

## 🚀3. Consume Web Retriever Service

To consume the Web Retriever Microservice, you can generate a mock embedding vector of length 768 with Python.

```bash
# Test
your_embedding=$(python -c "import random; embedding = [random.uniform(-1, 1) for _ in range(768)]; print(embedding)")

http_proxy= curl http://${your_ip}:7077/v1/web_retrieval \
  -X POST \
  -d "{\"text\":\"What is The Game of the Year 2024?\",\"embedding\":${your_embedding},\"k\":4}" \
  -H 'Content-Type: application/json'
```

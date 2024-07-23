export PATHWAY_HOST=0.0.0.0
export PATHWAY_PORT=8666

Running the Pathway Vector Store:
```bash
docker build -f Dockerfile.pathway -t vectorstore-pathway .

docker run -e PATHWAY_HOST=${PATHWAY_HOST} -e PATHWAY_PORT=${PATHWAY_PORT} -v ./data:/app/data -p ${PATHWAY_PORT}:${PATHWAY_PORT} vectorstore-pathway
```

Running the retriever:
```bash
###cd comps/retrievers/langchain/pathway/docker
docker build -t opea/retriever-pathway:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/retrievers/langchain/pathway/docker/Dockerfile .

docker run -p 7000:7000 -e PATHWAY_HOST=${PATHWAY_HOST} -e PATHWAY_PORT=${PATHWAY_PORT} --network="host" opea/retriever-pathway:latest
```

cd /comps/retrievers/langchain/pathway/docker
docker compose -f docker_compose_retriever.yaml down

Embedder:
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=${your_langchain_api_key}
export LANGCHAIN_PROJECT="opea/retriever"
model=BAAI/bge-base-en-v1.5
revision=refs/pr/4
volume=$PWD/data

# then run:
docker run -p 6060:80 -v $volume:/data -e http_proxy=$http_proxy -e https_proxy=$https_proxy --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-1.2 --model-id $model --revision $revision
```

If you are running the components individually and wishing to use the OPEA embedder service:
```bash
export TEI_EMBEDDING_ENDPOINT="http://127.0.0.1:6060"

docker run -e PATHWAY_HOST=${PATHWAY_HOST} -e PATHWAY_PORT=${PATHWAY_PORT} -e TEI_EMBEDDING_ENDPOINT=${TEI_EMBEDDING_ENDPOINT} -v ./data:/app/data -p ${PATHWAY_PORT}:${PATHWAY_PORT} --network="host" vectorstore-pathway
```


curl http://0.0.0.0:7000/v1/retrieval   -X POST   -d "{\"text\":\"What is the revenue of Nike in 2023?\",\"embedding\":${your_embedding}}"   -H 'Content-Type: application/json'


curl http://0.0.0.0:7000/v1/health_check   -X GET   -H 'Content-Type: application/json'


curl 127.0.0.1:6060/rerank     -X POST     -d '{"query":"What is Deep Learning?", "texts": ["Deep Learning is not...", "Deep learning is..."]}'     -H 'Content-Type: application/json'
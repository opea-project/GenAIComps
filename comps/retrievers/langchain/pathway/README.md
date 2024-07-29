# Retriever Microservice with Pathway

# 🚀Start Microservice with Python

## Install Requirements

```bash
pip install -r requirements.txt
```
### With the Docker compose

(Optionally) Start the embedder service separately.

> Note that Docker compose will start this service as well, this step is thus optional.

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=${your_langchain_api_key}
export LANGCHAIN_PROJECT="opea/retriever"
model=BAAI/bge-base-en-v1.5
revision=refs/pr/4
# TEI_EMBEDDING_ENDPOINT="http://${your_ip}:6060"  # if you want to use the hosted embedding service, example: "http://127.0.0.1:6060"

# then run:
docker run -p 6060:80 -e http_proxy=$http_proxy -e https_proxy=$https_proxy --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-1.2 --model-id $model --revision $revision
```

Health check the embedding service with:

```bash
curl 127.0.0.1:6060/rerank     -X POST     -d '{"query":"What is Deep Learning?", "texts": ["Deep Learning is not...", "Deep learning is..."]}'     -H 'Content-Type: application/json'
```

## Start the Pathway Vector DB Server

Make sure to set the Pathway environment variables.

> Note: If you are using `TEI_EMBEDDING_ENDPOINT`, make sure embedding service is already running.

```bash
export PATHWAY_HOST=0.0.0.0
export PATHWAY_PORT=8666
model=BAAI/bge-base-en-v1.5
revision=refs/pr/4
# TEI_EMBEDDING_ENDPOINT="http://${your_ip}:6060"  # uncomment if you want to use the hosted embedding service, example: "http://127.0.0.1:6060"
```

## Setting up the Pathway data sources

Pathway can listen to many sources simultaneously, such as local files, S3 folders, cloud storage, and any data stream. Whenever a new file is added or an existing file is modified, Pathway parses, chunks and indexes the documents in real-time.

See [pathway-io](https://pathway.com/developers/api-docs/pathway-io) for more information.

You can easily connect to the data inside the folder with the Pathway file system connector. The data will automatically be updated by Pathway whenever the content of the folder changes. In this example, we create a single data source that reads the files under the `./data` folder.

You can manage your data sources by configuring the `data_sources` in `pathway_vs.py`.

```python
import pathway as pw

data = pw.io.fs.read(
    "./data",
    format="binary",
    mode="streaming",
    with_metadata=True,
)  # This creates a Pathway connector that tracks
# all the files in the ./data directory

data_sources = [data]
```

Build the Docker and run the Pathway Vector Store:

```bash
cd comps/retrievers/langchain/pathway

docker build -t vectorstore-pathway .

# with locally loaded model, you may add `EMBED_MODEL` env variable to configure the model.
docker run -e PATHWAY_HOST=${PATHWAY_HOST} -e PATHWAY_PORT=${PATHWAY_PORT} -v ./data:/app/data -p ${PATHWAY_PORT}:${PATHWAY_PORT} vectorstore-pathway

# with the hosted embedder (network argument is needed for the vector server to reach to the embedding service)
docker run -e PATHWAY_HOST=${PATHWAY_HOST} -e PATHWAY_PORT=${PATHWAY_PORT} -e TEI_EMBEDDING_ENDPOINT=${TEI_EMBEDDING_ENDPOINT} -v ./data:/app/data -p ${PATHWAY_PORT}:${PATHWAY_PORT} --network="host" vectorstore-pathway
```

## Start Retriever Service

Note that the retriever service also expects the Pathway host and port variables to connect to the vector DB.

### With the Docker CLI

```bash
# make sure you are in the root folder of the repo
docker build -t opea/retriever-pathway:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/retrievers/langchain/pathway/docker/Dockerfile .

docker run -p 7000:7000 -e PATHWAY_HOST=${PATHWAY_HOST} -e PATHWAY_PORT=${PATHWAY_PORT} --network="host" opea/retriever-pathway:latest
```

### With the Docker compose

```bash
cd /comps/retrievers/langchain/pathway/docker

docker compose -f docker_compose_retriever.yaml build
docker compose -f docker_compose_retriever.yaml up

# shut down the containers
docker compose -f docker_compose_retriever.yaml down
```

Make sure the retriever service is working as expected:

```bash
curl http://0.0.0.0:7000/v1/health_check   -X GET   -H 'Content-Type: application/json'
```

send an example query:

```bash
exm_embeddings=$(python -c "import random; embedding = [random.uniform(-1, 1) for _ in range(768)]; print(embedding)")

curl http://0.0.0.0:7000/v1/retrieval   -X POST   -d "{\"text\":\"What is the revenue of Nike in 2023?\",\"embedding\":${exm_embeddings}}"   -H 'Content-Type: application/json'
```

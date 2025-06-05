# Dataprep microservice for financial domain data

## 1. Overview

We currently support ingestion of PDFs and URL links. The data should be financial domain, such as SEC filings and earnings call transcripts. If the data is not financial domain, you may encounter accuracy problems or errors.

The dataprep microservice saves financial documents into two databases:

1. One vector database of text chunks and tables
2. One KV store of full-length documents

Each unique company has its own index within the vector database and KV store, and metadata such as year, quarter, doc_title, doc_type and source are stored to enable metadata filtering to improve retrieval precision and recall. A company list is maintained so that when a new document comes, the document will either be mapped to an existing company or be added as a new company.

An LLM is used in processing the documents, including extracting metadata and generating summaries for text chunks and tables, and deciding if the document is about an existing company in the knowledge base or not. The default LLM to be used is `meta-llama/Llama-3.3-70B-Instruct`.

## 2. Deploy with docker

### 2.1 Start Redis vector database and Redis KV store

```bash
docker run --name redis-db -p 6379:6379 -p 8001:8001 -d redis/redis-stack:7.2.0-v9
docker run --name redis-kv -p 6380:6379 -p 8002:8001 -d redis/redis-stack:7.2.0-v9
```

### 2.2 Start Embedding Service

First, you need to start a TEI service.

```bash
your_port=6006
model="BAAI/bge-base-en-v1.5"
docker run -p $your_port:80 -v ./data:/data --name tei_server -e http_proxy=$http_proxy -e https_proxy=$https_proxy --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-1.6 --model-id $model
```

Then you need to test your TEI service using the following commands:

```bash
curl localhost:$your_port/embed \
    -X POST \
    -d '{"inputs":"What is Deep Learning?"}' \
    -H 'Content-Type: application/json'
```

### 2.3 Start vllm endpoint

First build vllm-gaudi docker image.

```bash
cd $WORKDIR
git clone https://github.com/HabanaAI/vllm-fork.git
# get the latest release tag of vllm gaudi
VLLM_VER=v0.6.6.post1+Gaudi-1.20.0
echo "Check out vLLM tag ${VLLM_VER}"
git checkout ${VLLM_VER}
docker build --no-cache -f Dockerfile.hpu -t opea/vllm-gaudi:latest --shm-size=128g . --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy
```

Then launch vllm on Gaudi.

```bash
export vllm_port=8086
export vllm_volume=$HF_CACHE_DIR
export max_length=16384
export model="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN=<your-hf-token>
docker run -d --runtime=habana --rm --name "vllm-gaudi-server" -e HABANA_VISIBLE_DEVICES=all -p $vllm_port:8000 -v $vllm_volume:/data -e HF_TOKEN=$HF_TOKEN -e HF_HOME=/data -e OMPI_MCA_btl_vader_single_copy_mechanism=none -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy -e VLLM_SKIP_WARMUP=true --cap-add=sys_nice --ipc=host opea/vllm-gaudi:comps --model ${model} --max-seq-len-to-capture $max_length --tensor-parallel-size 4
```

### 2.4 Build Docker Image for dataprep microservice

```bash
cd ../../ # go to GenAIComps
docker build -t opea/dataprep:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/src/Dockerfile .
```

### 2.5 Start dataprep microservice

```bash
export ip_address=$(hostname -I | awk '{print $1}')
export REDIS_URL_VECTOR="redis://${ip_address}:6379"
export REDIS_URL_KV="redis://${ip_address}:6380"
export LLM_MODEL="meta-llama/Llama-3.3-70B-Instruct"
export LLM_ENDPOINT="http://${ip_address}:8086"
export TEI_EMBEDDING_ENDPOINT="http://${your_ip}:6006"
export DATAPREP_COMPONENT_NAME="OPEA_DATAPREP_REDIS_FIANANCE"
export HF_TOKEN=<your-hf-token>
```

```bash
docker run -d --name="dataprep-redis-server-finance" -p 6007:5000 --runtime=runc --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e REDIS_URL_VECTOR=$REDIS_URL_VECTOR -e REDIS_URL_KV=$REDIS_URL_KV -e LLM_MODEL=$LLM_MODEL -e LLM_ENDPOINT=$LLM_ENDPOINT -e TEI_EMBEDDING_ENDPOINT=$TEI_EMBEDDING_ENDPOINT -e HF_TOKEN=$HF_TOKEN -e DATAPREP_COMPONENT_NAME=$DATAPREP_COMPONENT_NAME opea/dataprep:latest
```

### 2.6 Check the status of dataprep microservice

```bash
docker container logs -f dataprep-redis-server-finance
```

## 3. Consume Microservice

See example python script [here](../../../tests/dataprep/test_redis_finance.py).

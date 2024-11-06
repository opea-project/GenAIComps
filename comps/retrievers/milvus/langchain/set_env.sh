export host_ip=$(hostname -i)
export DOCKER_VOLUME_DIRECTORY="../../../vectorstores/milvus"
export no_proxy=${no_proxy},tei-embedding-service
export EMBEDDING_MODEL_ID="BAAI/bge-base-en-v1.5"
export MILVUS_HOST=${host_ip}
export MILVUS_PORT=19530
# Optional COLLECTION_NAME
export COLLECTION_NAME=${your_collection_name}


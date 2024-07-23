export PATHWAY_HOST=0.0.0.0
export PATHWAY_PORT=8666

docker build -f Dockerfile.pathway -t vectorstore-pathway .

docker run -e PATHWAY_HOST=${PATHWAY_HOST} -e PATHWAY_PORT=${PATHWAY_PORT} -v ./data:/app/data -p ${PATHWAY_PORT}:${PATHWAY_PORT} vectorstore-pathway


cd comps/retrievers/langchain/pathway/docker
docker build -t opea/retriever-pathway:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/retrievers/langchain/pathway/docker/Dockerfile .

docker run -p 7000:7000 -e PATHWAY_HOST=${PATHWAY_HOST} -e PATHWAY_PORT=${PATHWAY_PORT} --network="host" opea/retriever-pathway:latest


export TEI_EMBEDDING_ENDPOINT="http://127.0.0.1:6060"

docker run -e PATHWAY_HOST=${PATHWAY_HOST} -e PATHWAY_PORT=${PATHWAY_PORT} -e TEI_EMBEDDING_ENDPOINT=${TEI_EMBEDDING_ENDPOINT} -v ./data:/app/data -p ${PATHWAY_PORT}:${PATHWAY_PORT} --network="host" vectorstore-pathway
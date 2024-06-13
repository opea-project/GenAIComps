docker build -t opea/knowledge_graphs:latest \
    --build-arg https_proxy=$https_proxy \
    --build-arg http_proxy=$http_proxy \
    -f comps/knowledgegraphs/langchain/docker/Dockerfile .
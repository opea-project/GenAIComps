# Knowledge Graph Generation Microservice

This microservice, designed for generating knowledge graph with LLM, which take a document as input, then generate the corresponding graph with the help of LLM agents.

A prerequisite for using this microservice is that users must have a LLM text generation service and a knowledge gragh database already running. For LLM, you can refer to [llm microservices](https://github.com/opea-project/GenAIComps/tree/main/comps/llms/text-generation), such as TGI, vLLM, Ray Serve or vLLM on Ray. For knowledge graph database, currently we have support [Neo4J](https://neo4j.com/) for quick deployment. Users need to set the graph service's endpoint into an environment variable and microservie utilizes it for data injestion and retrieve.

# 🚀1. Start Microservice with Docker

## 1.1 Setup Environment Variables

```bash
export NEO4J_ENDPOINT="neo4j://${your_ip}:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD=${define_a_password}
export HUGGINGFACEHUB_API_TOKEN=${your_huggingface_api_token}
export LLM_ENDPOINT="http://${your_ip}:${your_port}"
export LLM_MODEL="meta-llama/Llama-2-7b-hf"
```

## 1.2 Start Neo4j Service

```bash
docker pull neo4j

docker run --rm \
    --publish=7474:7474 --publish=7687:7687 \
    --env NEO4J_AUTH=$NEO4J_USER/$NEO4J_PASSWORD \
    --volume=$PWD/neo4j_data:"/data" \
    --env='NEO4JLABS_PLUGINS=["apoc"]' \
    neo4j
```

## 1.3 Start LLM Service

You can start any LLM microserve, here we take TGI as an example.

```bash
docker run -p 8080:80 \
    -v $PWD/llm_data:/data --runtime=habana \
    -e HABANA_VISIBLE_DEVICES=all \
    -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
    -e HUGGING_FACE_HUB_TOKEN=$HUGGINGFACEHUB_API_TOKEN \
    --cap-add=sys_nice \
    --ipc=host \
    ghcr.io/huggingface/tgi-gaudi:2.0.0 \
    --model-id $LLM_MODEL \
    --max-input-tokens 1024 \
    --max-total-tokens 2048
```

Verify LLM service.

```bash
curl $LLM_ENDPOINT/generate \
  -X POST \
  -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":32}}' \
  -H 'Content-Type: application/json'
```

## 1.4 Start Microservice

```bash
cd ../..
docker build -t opea/graphs_generate:latest \
    --build-arg https_proxy=$https_proxy \
    --build-arg http_proxy=$http_proxy \
    -f comps/dataprep/graphs_generate/docker/Dockerfile .

docker run -it --rm \
    --name="graphs-generate-server" \
    -p 8070:8070 \
    --ipc=host \
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy \
    -e NEO4J_ENDPOINT=$NEO4J_ENDPOINT \
    -e NEO4J_USERNAME=$NEO4J_USERNAME \
    -e NEO4J_PASSWORD=$NEO4J_PASSWORD \
    -e HUGGINGFACEHUB_API_TOKEN=$HUGGINGFACEHUB_API_TOKEN \
    -e LLM_ENDPOINT=$LLM_ENDPOINT \
    opea/graphs_generate:latest
```

# 🚀2. Consume Knowledge Graph Service

curl http://172.17.0.1:8070/v1/graphs \
 -X POST \
 -d "{\"text\":\"data/wiki_documents.txt\",\"strtype\":\"doc\"}" \
 -H 'Content-Type: application/json'

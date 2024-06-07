# Table Extraction Microservice

# 🚀1. Start Microservice with Python（Option 1）

## 1.1 Install Requirements

```bash
apt-get install tesseract-ocr -y
apt-get install libtesseract-dev -y
apt-get install poppler-utils -y
pip install -r requirements.txt
```

## 1.2 Start TGI Service

```bash
export HUGGINGFACEHUB_API_TOKEN=${your_hf_api_token}
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=${your_langchain_api_key}
export LANGCHAIN_PROJECT="opea/gen-ai-comps:llms"
docker run -p 8008:80 -v ./data:/data --name tgi_service --shm-size 1g ghcr.io/huggingface/text-generation-inference:1.4 --model-id ${your_hf_llm_model}
```

## 1.3 Verify the TGI Service

```bash
curl http://${your_ip}:8008/generate \
  -X POST \
  -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":17, "do_sample": true}}' \
  -H 'Content-Type: application/json'
```

## 1.4 Setup Environment Variables

```bash
export HUGGINGFACEHUB_API_TOKEN=${your_hf_api_token}
export TGI_LLM_ENDPOINT="http://${your_ip}:8008"
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=${your_langchain_api_key}
export LANGCHAIN_PROJECT="opea/table:llms"
```

## 1.5 Start Table Extraction Microservice with Python Script

Start table extraction microservice with below command.

```bash
cd /your_project_path/GenAIComps/
cp comps/table_extraction/table_extract.py .
python table_extract.py
```

# 🚀2. Start Microservice with Docker (Option 2)

## 2.1 Start TGI Service

Please refer to 1.2.

## 2.2 Setup Environment Variables

```bash
export TGI_LLM_ENDPOINT="http://${your_ip}:8008"
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=${your_langchain_api_key}
export LANGCHAIN_PROJECT="opea/table:llms"
```

## 2.3 Build Docker Image

```bash
cd /your_project_path/GenAIComps
docker build --no-cache -t opea/table:latest -f comps/table_extraction/Dockerfile .
```

## 2.4 Run Docker with CLI (Option A)

```bash
docker run -it --name="table-server" --net=host --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e TGI_LLM_ENDPOINT=$TGI_LLM_ENDPOINT -e HUGGINGFACEHUB_API_TOKEN=$HUGGINGFACEHUB_API_TOKEN opea/table:latest
```

## 2.5 Run with Docker Compose (Option B)

```bash
cd /your_project_path/GenAIComps/comps/table_extraction
export LLM_MODEL_ID=${your_hf_llm_model}
export http_proxy=${your_http_proxy}
export https_proxy=${your_http_proxy}
export TGI_LLM_ENDPOINT="http://tgi-service:80"
export HUGGINGFACEHUB_API_TOKEN=${your_hf_api_token}
export LANGCHAIN_API_KEY=${your_langchain_api_key}
docker compose -f docker_compose_table.yaml up -d
```

# 🚀3. Consume Microservice

Once table extraction microservice is started, user can use below command to invoke the microservice.

`"table_strategy"` refers to the strategies to understand tables for table retrieval. As the setting progresses from "fast" to "hq" to "llm," the focus shifts towards deeper table understanding at the expense of processing speed. The default strategy is "fast"

Note: When start microservice with python, `"path"` should like `"/your_project_path/GenAIComps/comps/table_extraction/LLAMA2_page6.pdf" `. When start microservice with docker, `"path"` should like `"/home/user/comps/table_extraction/LLAMA2_page6.pdf" `.

```bash
curl http://${your_ip}:6008/v1/table/extract \
  -X POST \
  -d '{"path": "/path_to/LLAMA2_page6.pdf","table_strategy":"fast"}' \
  -H 'Content-Type: application/json'
```

```bash
curl http://${your_ip}:6008/v1/table/extract \
  -X POST \
  -d '{"path": "/path_to/LLAMA2_page6.pdf","table_strategy":"hq"}' \
  -H 'Content-Type: application/json'
```

```bash
curl http://${your_ip}:6008/v1/table/extract \
  -X POST \
  -d '{"path": "/path_to/LLAMA2_page6.pdf","table_strategy":"llm"}' \
  -H 'Content-Type: application/json'
```

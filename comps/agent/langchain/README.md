# langchain Agent Microservice

The langchain agent model refers to a framework that integrates the reasoning capabilities of large language models (LLMs) with the ability to take actionable steps, creating a more sophisticated system that can understand and process information, evaluate situations, take appropriate actions, communicate responses, and track ongoing situations.

![Architecture Overview](agent_arch.jpg)

# 🚀1. Start Microservice with Python（Option 1）

## 1.1 Install Requirements

```bash
cd comps/agent/langchain/
pip install -r requirements.txt
```

## 1.2 Start Microservice with Python Script

```bash
cd comps/agent/langchain/
python agent.py
```

# 🚀2. Start Microservice with Docker (Option 2)

## Build Microservices

```bash
cd GenAIComps/ # back to GenAIComps/ folder
docker build -t opea/comps-agent-langchain:latest -f comps/agent/langchain/docker/Dockerfile .
```

## start microservices

```bash
export model=meta-llama/Meta-Llama-3-8B-Instruct
export HUGGINGFACEHUB_API_TOKEN=${HUGGINGFACEHUB_API_TOKEN}
export HF_TOKEN=${HUGGINGFACEHUB_API_TOKEN}

# TGI serving
docker run -d --runtime=habana --name "comps-tgi-gaudi-service" -p 8080:80 -v ./data:/data -e HF_TOKEN=$HF_TOKEN -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host ghcr.io/huggingface/tgi-gaudi:latest --model-id $model

# check status
docker logs comps-tgi-gaudi-service

# Agent
docker run -d --runtime=runc --name="comps-langchain-agent-endpoint" -v $WORKPATH/comps/agent/langchain/tools:/home/user/comps/agent/langchain/tools -p 9090:9090 --ipc=host -e HUGGINGFACEHUB_API_TOKEN=${HUGGINGFACEHUB_API_TOKEN} -e model=${model} -e strategy=react -e llm_endpoint_url=http://${ip_address}:8080 -e llm_engine=tgi -e recursive_limit=5 -e require_human_feedback=false -e tools=/home/user/comps/agent/langchain/tools/custom_tools.yaml opea/comps-agent-langchain:latest

# check status
docker logs comps-langchain-agent-endpoint
```

> debug mode
>
> ```bash
> docker run --rm --runtime=runc --name="comps-langchain-agent-endpoint" -v ./comps/agent/langchain/:/home/user/comps/agent/langchain/ -p 9090:9090 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e HUGGINGFACEHUB_API_TOKEN=${HUGGINGFACEHUB_API_TOKEN} --env-file ${agent_env} opea/comps-agent-langchain:latest
> ```

# 🚀3. Get Status of Microservice

```bash
docker container logs -f comps-langchain-agent-endpoint
```

# 🚀4. Consume Microservice

Once microservice starts, user can use below script to invoke.

```bash
cd comps/agent/langchain/; python test.py --endpoint_test --ip_addr=${endpoint_ip_addr}

{"query": "What is the weather today in Austin?"}
data: 'The temperature in Austin today is 78°F.</s>'

data: [DONE]


```

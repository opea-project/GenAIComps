# langchain Agent Microservice

The langchain agent model refers to a framework that integrates the reasoning capabilities of large language models (LLMs) with the ability to take actionable steps, creating a more sophisticated system that can understand and process information, evaluate situations, take appropriate actions, communicate responses, and track ongoing situations.

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

## 2.1 Prepare env

```bash
export WORKDIR=<YOUR WORK DIRECTORY>
export POLYGON_API_KEY=<YOUR POLYGON API KEY>
export TAVILY_API_KEY=<YOUR TAVILY API KEY>
export HF_TOKEN=<YOUR HUGGINGFACE HUB TOKEN>
# optional, if you want to test OpenAI models
export OPENAI_API_KEY=<YOUR OPENAI_API_KEY>
export local_model_dir=<YOUR LOCAL DISK TO STORE MODEL>

export HF_TOKEN=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
export HUGGINGFACEHUB_API_TOKEN=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
```

## Use mistral as llm endpoint
``` bash
model=meta-llama/Llama-2-7b-hf

#single node
docker run --rm -p 8080:80 -v ${local_model_dir}:/data --runtime=habana --name "tgi-gaudi-mistral" -e HF_TOKEN=$HF_TOKEN -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host ghcr.io/huggingface/tgi-gaudi:2.0.0 --model-id $model --max-input-tokens 1024 --max-total-tokens 2048
```

## 2.2 Build Docker Image

```bash
cd ../../../ # back to GenAIComps/ folder
docker build -t opea/comps-agent-langchain:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/agent/langchain/docker/Dockerfile .
```

## 2.3 Run Docker with CLI

```bash
docker run -d --rm --runtime=runc --name="comps-langchain-agent-endpoint" -p 9000:9000 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e HUGGINGFACEHUB_API_TOKEN=${HUGGINGFACEHUB_API_TOKEN} opea/comps-agent-langchain:latest
```

> debug mode
> ```bash
> docker run --rm --runtime=runc --name="" -v ./comps/agent/langchain/:/home/user/comps/agent/langchain/ -p 9000:9000 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e HUGGINGFACEHUB_API_TOKEN=${HUGGINGFACEHUB_API_TOKEN} opea/comps-agent-langchain:latest
> ```

# 🚀3. Get Status of Microservice

```bash
docker container logs -f comps-langchain-agent-endpoint
```

# 🚀4. Consume Microservice

Once microservice starts, user can use below script to invoke.

```bash
cd comps/agent/langchain/; python test.py
```
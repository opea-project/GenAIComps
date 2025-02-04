# TGI LLM Microservice

[Text Generation Inference](https://github.com/huggingface/text-generation-inference) (TGI) is a toolkit for deploying and serving Large Language Models (LLMs). TGI enables high-performance text generation for the most popular open-source LLMs, including Llama, Falcon, StarCoder, BLOOM, GPT-NeoX, and more.

## Start TGI with docker compose

Set up environment.

```bash
export LLM_ENDPOINT_PORT=8008
export host_ip=${host_ip}
export HF_TOKEN=${HF_TOKEN}
export LLM_MODEL_ID="Intel/neural-chat-7b-v3-3"
export MAX_INPUT_TOKENS=1024
export MAX_TOTAL_TOKENS=2048
```

Run tgi on xeon.

```bash
cd deplopyment/docker_compose
docker compose -f compose.yaml up -d tgi-server
```

Run tgi on gaudi.

```bash
cd deplopyment/docker_compose
docker compose -f compose.yaml up -d tgi-gaudi-server
```

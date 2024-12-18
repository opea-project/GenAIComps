# Introduction

[Bedrock](https://aws.amazon.com/bedrock) Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies like AI21 Labs, Anthropic, Cohere, Meta, Mistral AI, Stability AI, and Amazon through a single API, along with a broad set of capabilities you need to build generative AI applications with security, privacy, and responsible AI.

## Get Started

## Setup Environment Variables

In order to start Bedrock service, you need to setup the following environment variables first.

```bash
export AWS_ACCESS_KEY_ID=${aws_access_key_id}
export AWS_SECRET_ACCESS_KEY=${aws_secret_access_key}
```

## Build Docker Image

```bash
cd GenAIComps/
docker build --no-cache -t opea/bedrock:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/llms/text-generation/bedrock/Dockerfile .
```

## Run the Bedrock Microservice

```bash
docker run -d --name bedrock -p  9009:9000 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY  opea/bedrock:latest
```

## Consume the Bedrock Microservice

```bash
curl http://${host_ip}:9009/v1/chat/completions \
  -X POST \
  -d '{"model": "us.anthropic.claude-3-5-haiku-20241022-v1:0", "messages": [{"role": "user", "content": "What is Deep Learning?"}], "max_tokens":17}' \
  -H 'Content-Type: application/json'

curl http://${host_ip}:9009/v1/chat/completions \
 -X POST \
 -d '{"model": "us.anthropic.claude-3-5-haiku-20241022-v1:0", "messages": [{"role": "user", "content": "What is Deep Learning?"}], "max_tokens":17, "stream": "true"}' \
 -H 'Content-Type: application/json'
```

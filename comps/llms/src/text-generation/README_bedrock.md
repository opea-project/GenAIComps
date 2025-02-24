# Introduction

[Bedrock](https://aws.amazon.com/bedrock) Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies like AI21 Labs, Anthropic, Cohere, Meta, Mistral AI, Stability AI, and Amazon through a single API, along with a broad set of capabilities you need to build generative AI applications with security, privacy, and responsible AI.

## Get Started

## Setup Environment Variables

In order to start Bedrock service, you need to setup the following environment variables first.

```bash
export AWS_REGION=${aws_region}
export AWS_ACCESS_KEY_ID=${aws_access_key_id}
export AWS_SECRET_ACCESS_KEY=${aws_secret_access_key}
```

If you're using an IAM Role, you also need to set the following environment variable.

```bash
export AWS_SESSION_TOKEN=${aws_session_token}
```

## Build Docker Image

```bash
cd GenAIComps/
docker build --no-cache -t opea/llm-textgen:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/llms/src/text-generation/Dockerfile .
```

## Run the Bedrock Microservice

```bash
docker run -d --name bedrock -p  9009:9000 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e LLM_COMPONENT_NAME="OpeaTextGenBedrock" -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY -e AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN -e BEDROCK_REGION=$AWS_REGION opea/llm-textgen:latest
```

(You can remove `-e AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN` if you are not using an IAM Role)

## Consume the Bedrock Microservice

```bash
curl http://${host_ip}:9009/v1/chat/completions \
  -X POST \
  -d '{"model": "us.anthropic.claude-3-5-haiku-20241022-v1:0", "messages": [{"role": "user", "content": "What is Deep Learning?"}], "max_tokens":17}' \
  -H 'Content-Type: application/json'

# stream mode
curl http://${host_ip}:9009/v1/chat/completions \
 -X POST \
 -d '{"model": "us.anthropic.claude-3-5-haiku-20241022-v1:0", "messages": [{"role": "user", "content": "What is Deep Learning?"}], "max_tokens":17, "stream": "true"}' \
 -H 'Content-Type: application/json'
```

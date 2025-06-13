# Introduction

This OPEA text generation service can connect to any OpenAI-compatible API endpoint, including local deployments (like vLLM or TGI) and remote services (like OpenRouter.ai).

## 1 Prepare TextGen docker image.

```bash
# Build the microservice docker

git clone https://github.com/opea-project/GenAIComps
cd GenAIComps

docker build \
  --no-cache \
  --build-arg https_proxy=$https_proxy \
  --build-arg http_proxy=$http_proxy \
  -t opea/llm-textgen:latest \
  -f comps/llms/src/text-generation/Dockerfile .
```

## 2 Setup Environment Variables

The key environment variable is `LLM_ENDPOINT`, which specifies the URL of the OpenAI-compatible API. This can be a local address (e.g., for vLLM or TGI) or a remote address.

```
export host_ip=$(hostname -I | awk '{print $1}')
export LLM_MODEL_ID="" # e.g. "google/gemma-3-1b-it:free"
export LLM_ENDPOINT=""  # e.g., "http://localhost:8000" (for local vLLM) or "https://openrouter.ai/api" (please make sure to omit /v1 suffix)
export OPENAI_API_KEY=""
```

## 3 Run the Textgen Service

```
export service_name="textgen-service-endpoint-openai"
docker compose -f comps/llms/deployment/docker_compose/compose_text-generation.yaml up ${service_name} -d
```

To observe logs:

```
docker logs textgen-service-endpoint-openai
```

## 4 Test the service

You can first test the remote/local endpoint with `curl`. If you're using a service like OpenRouter, you can test it directly first:

```
curl https://openrouter.ai/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
  "model": "'${LLM_MODEL_ID}'",
  "messages": [
    {
      "role": "user",
      "content": "Tell me a joke?"
    }
  ]
}'
```

Then you can test the OPEA text generation service that wrapped the endpoint, with the following:

```
curl http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"'${LLM_MODEL_ID}'","messages":[{"role":"user","content":"Tell me a joke?"}]}'
```

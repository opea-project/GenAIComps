# Introduction

This OPEA text generation service run any remote OpenAI style url endpoints, including OpenRouter.


## 1 Prepare TextGen docker image.

```bash
# Build the microservice docker
cd ${OPEA_GENAICOMPS_ROOT}

docker build \
  --no-cache \
  --build-arg https_proxy=$https_proxy \
  --build-arg http_proxy=$http_proxy \
  -t opea/llm-textgen:latest \
  -f comps/llms/src/text-generation/Dockerfile .
```

## 2 Setup Environment Variables

```
export host_ip=$(hostname -I | awk '{print $1}')
export LLM_MODEL_ID="" # e.g. "google/gemma-3-1b-it:free"
export REMOTE_ENDPOINT=""  # e.g. "https://openrouter.ai/api"  # Important to omit /v1, and no / at end 
export OPENAI_API_KEY="" 

```

## 3 Run the Remote Textgen Service

```
export service_name="textgen-service-endpoint-openai"
docker compose -f comps/llms/deployment/docker_compose/compose_text-generation.yaml up ${service_name} -d
```

To observe logs:
```
docker logs textgen-service-endpoint-openai
```

## 4 Test the remote service

For example, if you are using openrouter:

```
curl https://openrouter.ai/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
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

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer testkey" \
  -d '{
  "model": "'${LLM_MODEL_ID}'",
  "messages": [
    {
      "role": "user",
      "content": "Tell me a joke?"
    }
  ]
}'

## 5 Consume the Microservice

```
curl -X POST http://localhost:9000/v1/chat/completions \
    -H "Content-Type: application/json" \
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
        
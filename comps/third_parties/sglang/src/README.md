# SGLang Serving microservice

## 🚀1. Build the Docker Image

```bash
cd ../../../../
docker build -f comps/third_parties/sglang/src/Dockerfile --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -t opea/sglang-llm:latest .
```

## 🚀2. Start the microservice

```bash
export MODEL_ID="meta-llama/Llama-Guard-4-12B"

cd comps/third_parties/sglang/deployment/docker_compose
docker compose -f compose.yaml up -d
```

## 🚀3. Access the service

Then you need to test your service using the following commands:

```bash
http_proxy="" curl -X POST -H "Content-Type: application/json" -d '{"model": "meta-llama/Llama-Guard-4-12B", "messages": [{"role": "user", "content": "Hello! What is your name?"}], "max_tokens": 128}' http://localhost:8699/v1/chat/completions
```

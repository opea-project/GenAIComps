# ImageGen Microservice

The ImageGen microservice is a solution for generating images using text input. The ImageGen solution service takes a triton endpoint that serves the actual text-to-image model, and in turn this service provides a solution endpoint consumable by users.

# 1. Instructions to launch this solution

This solution requires 1 backing container to operate - a triton-based inference server for executing the diffusion model. We will walk through how to deploy both images below.

## 2.1 Build Model Server Docker Image

```cd triton && make build
```

## 2.2 Build Solution Server Docker Image

```docker build -t opea/image-gen:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f Dockerfile .
```

## 2.3 Run Docker with CLI

```bash
docker run -p 18000:8000 -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e HUGGINGFACE_API_TOKEN=${HUGGINGFACE_API_TOKEN} -e HABANA_VISIBLE_DEVICES=0 -v /opt/intel/huggingface/hub:/root/.cache/huggingface/hub ohio-image-triton:latest
docker run -p 9765:9765 -e IMAGE_GEN_TRITON_ENDPOINT=http://localhost:18000 opea/image-gen:latest
```

# 3. Consume Solution Service

You can use the following `curl` command to test whether the service is up. Notice that the first request can be slow because it needs to download the models.

```bash
curl http://localhost:9765/v1/images/generation \
    -H "Content-Type: application/json"   \
    -d '{"text":"A cat holding a fish skeleton"}'
```


# LVM Microservice with Video-LLaMA

This service provides specialized Visual Question and Answering (VQA) capabilities for video content using the Video-LLaMA model. It can analyze video clips and answer questions about them.

---

## Table of Contents

1. [Start Microservice](#start-microservice)
2. [Consume LVM Service](#consume-lvm-service)

---

## Start Microservice

### Build Docker Image

First, build the generic LVM microservice Docker image:

```bash
cd ../../../
docker build -t opea/lvm:latest \
  --build-arg https_proxy=$https_proxy \
  --build-arg http_proxy=$http_proxy \
  -f comps/lvms/src/Dockerfile .
```

### Run with Docker Compose

Deploy the Video-LLaMA service and the LVM microservice using Docker Compose.

1.  Export the required environment variables:

    ```bash
    export ip_address=$(hostname -I | awk '{print $1}')
    export LVM_PORT=9399
    export VIDEO_LLAMA_PORT=11506
    export LVM_ENDPOINT="http://$ip_address:$VIDEO_LLAMA_PORT"
    ```

2.  Navigate to the Docker Compose directory and start the services:
    ```bash
    cd comps/lvms/deployment/docker_compose/
    docker compose up video-llama-service lvm-video-llama -d
    ```

---

## Consume LVM Service

Once the service is running, you can send requests to the API.

### Use the LVM Service API

Send a POST request with a `video_url` and a prompt. You can specify which part of the video to analyze with `chunk_start` and `chunk_duration`.

```bash
curl http://localhost:9399/v1/lvm \
  -X POST \
  -d '{"video_url":"https://github.com/DAMO-NLP-SG/Video-LLaMA/raw/main/examples/silence_girl.mp4","chunk_start": 0,"chunk_duration": 9,"prompt":"What is the person doing?","max_new_tokens": 150}' \
  -H 'Content-Type: application/json'
```

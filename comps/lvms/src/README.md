# LVM Microservice

Visual Question and Answering is one of the multimodal tasks empowered by LVMs (Large Visual Models). This microservice supports visual Q&A by using LLaVA as the base large visual model. It accepts two inputs: a prompt and an image. It outputs the answer to the prompt about the image.

## 🚀1. Start Microservice with Docker (Option 1)

You have to build/start the [dependency](./integrations/dependency/) service based on your demands.

```bash
docker build --no-cache -t opea/lvm:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy  -f comps/lvms/src/Dockerfile .
# Change LVM_ENDPOINT to you dependency service endpoint
docker run -d --name="test-comps-lvm" -e LVM_ENDPOINT=http://localhost:8399 -e http_proxy=$http_proxy -e https_proxy=$https_proxy -p 9399:9399 --ipc=host opea/lvm:latest
```

## 🚀1. Start Microservice with Docker Compose (Option 2)

Alternatively, you can also start the TTS microservice with Docker Compose.

- LLaVA

```bash
export ip_address=$(hostname -I | awk '{print $1}')
export LVM_PORT=9399
export LLAVA_PORT=11500
export LVM_ENDPOINT=http://$ip_address:$LLAVA_PORT
docker compose -f comps/lvms/deployment/docker_compose/compose.yaml up llava-service lvm-llava -d
```

- LLaVA TGI on HPU Gaudi

```bash
export ip_address=$(hostname -I | awk '{print $1}')
export LVM_PORT=9399
export LLAVA_TGI_PORT=11502
export LVM_ENDPOINT=http://$ip_address:$LLAVA_TGI_PORT
docker compose -f comps/lvms/deployment/docker_compose/compose.yaml up llava-tgi-service lvm-llava-tgi -d
```

- LLaMA Vision

```bash
export ip_address=$(hostname -I | awk '{print $1}')
export LVM_PORT=9399
export LLAMA_VISION_PORT=11510
export LVM_ENDPOINT=http://$ip_address:$LLAMA_VISION_PORT

docker compose -f comps/lvms/deployment/docker_compose/compose.yaml up lvm-llama-vision llama-vision-service -d
```

- PredictionGuard

```bash
export ip_address=$(hostname -I | awk '{print $1}')
export PREDICTIONGUARD_PORT=9399

docker compose -f comps/lvms/deployment/docker_compose/compose.yaml up predictionguard-service -d
```

- Video LLaMA

```bash
export ip_address=$(hostname -I | awk '{print $1}')
export LVM_PORT=9399
export VIDEO_LLAMA_PORT=11506
export LVM_ENDPOINT=http://$ip_address:$VIDEO_LLAMA_PORT
docker compose -f comps/lvms/deployment/docker_compose/compose.yaml up video-llama-service lvm-video-llama -d
```

## Test

- LLaVA & llama-vision & PredictionGuard & TGI LLaVA

```bash
# curl with an image and a prompt
http_proxy="" curl http://localhost:9399/v1/lvm -XPOST -d '{"image": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8/5+hnoEIwDiqkL4KAcT9GO0U4BxoAAAAAElFTkSuQmCC", "prompt":"What is this?"}' -H 'Content-Type: application/json'

# curl with only the prompt
http_proxy="" curl http://localhost:9399/v1/lvm --silent --write-out "HTTPSTATUS:%{http_code}" -XPOST -d '{"image": "", "prompt":"What is deep learning?"}' -H 'Content-Type: application/json'
```

- video-llama

```bash
http_proxy="" curl -X POST http://localhost:9399/v1/lvm -d '{"video_url":"https://github.com/DAMO-NLP-SG/Video-LLaMA/raw/main/examples/silence_girl.mp4","chunk_start": 0,"chunk_duration": 9,"prompt":"What is the person doing?","max_new_tokens": 150}' -H 'Content-Type: application/json'
```

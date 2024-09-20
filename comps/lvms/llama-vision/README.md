# LVM Microservice

Visual Question and Answering is one of the multimodal tasks empowered by LVMs (Large Visual Models). This microservice supports visual Q&A by using Llama Vision as the base large visual model. It accepts two inputs: a prompt and an image. It outputs the answer to the prompt about the image.


## 🚀 Start Microservice with Docker

### Build Images

```bash
cd ../../../
docker build -t opea/lvm-llama-vision:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/lvms/llama-vision/Dockerfile .
```

### Start LLaVA and LVM Service

```bash
docker run -p 9399:9399 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e LLAMA_VISION_MODEL_ID="/mnt/models/Llama-3.2-11B-Vision-Instruct" opea/lvm-llama-vision:latest
```

### Test

```bash
# Use curl

# curl
http_proxy="" curl http://localhost:9399/v1/lvm -XPOST -d '{"image": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8/5+hnoEIwDiqkL4KAcT9GO0U4BxoAAAAAElFTkSuQmCC", "prompt":"What is this?"}' -H 'Content-Type: application/json'

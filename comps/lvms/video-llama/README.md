# LVM Microservice

This is a Docker-based microservice that runs Video-Llama as a Large Vision Model (LVM). It utilizes Llama-2-7b-chat-hf for conversations based on video dialogues. It support Intel Xeon CPU.


# 🚀1. Start Microservice with Docker

## 1.1 Build Images

```bash
cd video-llama
# Video-Llama Server Image
docker build --no-cache -t opea/video-llama-lvm-server:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f server/docker/Dockerfile .
# LVM Service Image
docker build --no-cache -t opea/lvm-video-llama:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f Dockerfile .
```

## 1.2 Start Video-Llama and LVM Services

```bash
# prepare environment variables
export ip_address=$(hostname -I | awk '{print $1}')
export no_proxy=$no_proxy,${ip_address}
export LVM_ENDPOINT=http://${ip_address}:9009
# Start service
docker compose -f docker_compose.yaml up -d
# wait for model download in video-llama server, takes about 1.5 hrs for at most 100Mb/s network
until docker logs video-llama-lvm-server 2>&1 | grep -q "Uvicorn running on"; do
    sleep 5m
done
```

# ✅ 2. Test

```bash
# use curl
export ip_address=$(hostname -I | awk '{print $1}')
## check video-llama
http_proxy="" curl -X POST 'http://${ip_address}:9009/generate' \
     --data-urlencode 'video_url=https://github.com/DAMO-NLP-SG/Video-LLaMA/raw/main/examples/silence_girl.mp4' \
     --data-urlencode 'start=0.0' \
     --data-urlencode 'duration=9' \
     --data-urlencode 'prompt=What is the person doing?' \
     --data-urlencode 'max_new_tokens=150'

## check lvm
http_proxy="" curl -X POST http://${ip_address}:9000/v1/lvm -d '{"video_url":"https://github.com/DAMO-NLP-SG/Video-LLaMA/raw/main/examples/silence_girl.mp4","chunck_start": 0,"chunck_duration": 9,"prompt":"What is the person doing?","max_new_tokens": 50}' -H 'Content-Type: application/json'

# or use python
export ip_address=$(hostname -I | awk '{print $1}')
python check-_lvm.py
```

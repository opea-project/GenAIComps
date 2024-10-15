
# Docsum

## Video2audio microservice
cd /GenAIComps/
#### Build video2audio microservice 
```bash
docker build -t opea/dataprep-docsum-video2audio-microservice:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy --build-arg no_proxy=${no_proxy} -f comps/dataprep/docsum/Dockerfile_microservice .  
```

#### Start video2audio microservice 
```bash
docker run --rm -d --name="dataprep-docsum-video2audio-microservice" -p 7078:7078 opea/dataprep-docsum-video2audio-microservice:latest
```


#### Build whisper and A2T Image
```bash
docker build -t opea/whisper:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/asr/whisper/dependency/Dockerfile .


docker build -t opea/a2t:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/docsum/audio2text/Dockerfile_microservice .

# docker build -t opea/asr:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/asr/whisper/Dockerfile .

```

```bash
host_ip=$(hostname -I | awk '{print $1}')
ip_address=$(hostname -I | awk '{print $1}')

docker run -d -p 7066:7066 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy opea/whisper:latest

docker run -d -p 9099:9099 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e A2T_ENDPOINT=http://$ip_address:7066 opea/a2t:latest

# docker run -d -p 9099:9099 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e ASR_ENDPOINT=http://$ip_address:7066 opea/asr:latest
```


#### Build Xeon-Backend-Server (Mega service) Image
cd GenAIExamples/DocSum
```bash
docker build -t opea/docsum-xeon-backend-server:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy --build-arg no_proxy=${no_proxy} -f Dockerfile .
```

#### Run Xeon-Backend-Server (Mega service) Image
```bash

# host_ip=$(hostname -I | awk '{print $1}')
# ip_address=$(hostname -I | awk '{print $1}')

# LLM_MODEL_ID="Intel/neural-chat-7b-v3-3"
# TGI_LLM_ENDPOINT="http://${host_ip}:8008"
# LLM_SERVICE_HOST_IP=${host_ip}

# - no_proxy=${no_proxy}
# - https_proxy=${https_proxy}
# - http_proxy=${http_proxy}
# - MEGA_SERVICE_HOST_IP=${MEGA_SERVICE_HOST_IP}
# - LLM_SERVICE_HOST_IP=${LLM_SERVICE_HOST_IP}

host_ip=$(hostname -I | awk '{print $1}')
ip_address=$(hostname -I | awk '{print $1}')

docker run --rm -it --name="docsum-xeon-backend-server" \
    -p 8888:8888 \
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy \
    -e ASR_ENDPOINT=http://$ip_address:7066\
    -e LLM_SERVICE_HOST_IP=$ip_address \
    -e MEGA_SERVICE_HOST_IP=$ip_address \
    opea/docsum-xeon-backend-server:latest

```

```bash
curl http://${host_ip}:7066/v1/asr -X POST d '{"byte_str": "UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA"}' -H 'Content-Type: application/json'


curl http://${host_ip}:3001/v1/audio/transcriptions -X POST -d '{"byte_str": "UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA"}' -H 'Content-Type: application/json'

```


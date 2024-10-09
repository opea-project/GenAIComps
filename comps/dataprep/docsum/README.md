
# Docsum

## Video2audio service
cd /GenAIComps/comps/dataprep/docsum/video2audio
#### Build video2audio service 
```bash
docker build -t opea/dataprep-docsum-video2audio:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy --build-arg no_proxy=${no_proxy} -f Dockerfile .  
```

#### Start video2audio service 
```bash
docker run --rm -d --name="dataprep-docsum-video2audio" -p 7077:7077 opea/dataprep-docsum-video2audio:latest
```

## Video2audio microservice
cd /GenAIComps/comps/dataprep/docsum
#### Build video2audio microservice 
```bash
docker build -t opea/dataprep-docsum-video2audio-microservice:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy --build-arg no_proxy=${no_proxy} -f comps/dataprep/docsum/Dockerfile_microservice .  
```

<!-- #### Start video2audio microservice 
```bash
docker run --rm -d --name="dataprep-docsum-video2audio-microservice" -p 7078:7078 --runtime=runc --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e http_proxy=$no_proxy opea/dataprep-docsum-video2audio-microservice:latest
```
 -->





```bash
ip_address=$(hostname -I | awk '{print $1}')

docker run --rm -it --name="dataprep-docsum-video2audio-microservice" -p 7078:7078 -e VIDEO2AUDIO_ENDPOINT=http://$ip_address:7077 opea/dataprep-docsum-video2audio-microservice:latest
```
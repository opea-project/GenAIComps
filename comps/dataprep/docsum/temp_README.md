
# Whisper Service 
## Build
```bash
docker build -t opea/whisper:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/asr/whisper/dependency/Dockerfile .
```
## Run 
```bash
docker run -d -p 7066:7066 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy opea/whisper:latest
```


# A2T Service 
## Build
```bash
docker build -t opea/a2t:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/docsum/audio2text/Dockerfile_microservice .


```
## Run 
```bash
ip_address=$(hostname -I | awk '{print $1}')

docker run -d -p 9099:9099 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e A2T_ENDPOINT=http://$ip_address:7066 opea/a2t:latest
```



# V2A Service 
## Build
```bash
docker build -t opea/v2a:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/docsum/video2audio/Dockerfile_microservice .
```
## Run 
```bash
ip_address=$(hostname -I | awk '{print $1}')

docker run -d -p 7078:7078 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy opea/v2a:latest
```


# Data Prep Service 
## Build
```bash
docker build -t opea/docsum_data:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/docsum/Dockerfile_data_prep_microservice .
```
## Run 
```bash
ip_address=$(hostname -I | awk '{print $1}')

docker run -it -p 7079:7079 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy \
    -e A2T_ENDPOINT=http://$ip_address:7066 \
    -e V2A_ENDPOINT=http://$ip_address:7078 \
    opea/docsum_data:latest 

```












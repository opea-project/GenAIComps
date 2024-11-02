# Multimedia to Text Services

This guide provides instructions on how to build and run various Docker services for converting multimedia content to text. The services include:

1. **Whisper Service**: Converts audio to text.
2. **A2T Service**: Another service for audio to text conversion.
3. **Video to Audio Service**: Extracts audio from video files.
4. **Multimedia2Text Service**: Transforms multimedia data to text data.

## Prerequisites

Ensure you have Docker installed and running on your system. Also, make sure you have the necessary proxy settings configured if you are behind a corporate firewall.

## Getting Started

First, navigate to the `GenAIComps` directory:

```bash
cd GenAIComps
```

### Whisper Service

The Whisper Service converts audio files to text. Follow these steps to build and run the service:

#### Build

```bash
docker build -t opea/whisper:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/asr/whisper/dependency/Dockerfile .
```

#### Run

```bash
docker run -d -p 7066:7066 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy opea/whisper:latest
```

### A2T Service

The A2T Service is another service for converting audio to text. Follow these steps to build and run the service:

#### Build

```bash
docker build -t opea/a2t:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/multimedia2text/audio2text/Dockerfile .
```

#### Run

```bash
ip_address=$(hostname -I | awk '{print $1}')

docker run -d -p 9099:9099 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e A2T_ENDPOINT=http://$ip_address:7066 opea/a2t:latest
```

### Video to Audio Service

The Video to Audio Service extracts audio from video files. Follow these steps to build and run the service:

#### Build

```bash
docker build -t opea/v2a:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/multimedia2text/video2audio/Dockerfile .
```

#### Run

```bash
docker run -d -p 7078:7078 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy opea/v2a:latest
```

### Multimedia2Text Service

The Multimedia2Text Service transforms multimedia data to text data. Follow these steps to build and run the service:

#### Build

```bash
docker build -t opea/multimedia2text:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/multimedia2text/Dockerfile .
```

#### Run

```bash
ip_address=$(hostname -I | awk '{print $1}')

docker run -it -p 7079:7079 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy \
    -e A2T_ENDPOINT=http://$ip_address:7066 \
    -e V2A_ENDPOINT=http://$ip_address:7078 \
    opea/multimedia2text:latest
```

## Validate Microservices

After building and running the services, you can validate them using the provided Python scripts. Below are the steps to validate each service:

### Whisper Service

Run the following command to validate the Whisper Service:

```bash
python comps/asr/whisper/dependency/check_whisper_server.py 
```

Expected output:

```
{'asr_result': 'who is pat gelsinger'}
```

### Audio2Text Service

Run the following command to validate the Audio2Text Service:

```bash
python comps/dataprep/multimedia2text/audio2text/check_a2t_server.py
```

Expected output:

```
{'downstream_black_list': [], 'id': '21b0459477abea6d85d20f4b5ddcb714', 'query': 'who is pat gelsinger'}
```

*Note: The `id` value will be different.*

### Video2Audio Service

Run the following command to validate the Video2Audio Service:

```bash
python comps/dataprep/multimedia2text/video2audio/check_v2a_microserver.py
```

Expected output:

```
========= Audio file saved as ======
comps/dataprep/multimedia2text/video2audio/converted_audio.wav
====================================
```

### Multimedia2Text Service

Run the following command to validate the Multimedia2Text Service:

```bash
python comps/dataprep/multimedia2text/check_multimedia2text.py 
```

Expected output:

```
Running test: Whisper service
>>> Whisper service Test Passed ... 

Running test: Audio2Text service
>>> Audio2Text service Test Passed ... 

Running test: Video2Text service
>>> Video2Text service Test Passed ... 

Running test: Multimedia2text service
>>> Multimedia2text service test for text data type passed ... 
>>> Multimedia2text service test for audio data type passed ... 
>>> Multimedia2text service test for video data type passed ... 
```







<!-- 

# Audio to Text Servide 

```
cd GenAIComps
```

## Whisper Service 
### Build
```bash
docker build -t opea/whisper:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/asr/whisper/dependency/Dockerfile .
```
### Run 
```bash
docker run -d -p 7066:7066 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy opea/whisper:latest
```


## A2T Service 
### Build
```bash
docker build -t opea/a2t:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/multimedia2text/audio2text/Dockerfile .


```
### Run 
```bash
ip_address=$(hostname -I | awk '{print $1}')

docker run -d -p 9099:9099 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e A2T_ENDPOINT=http://$ip_address:7066 opea/a2t:latest
```

# Video to Audio Service 
### Build
```bash
docker build -t opea/v2a:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/multimedia2text/video2audio/Dockerfile .
```
### Run 
```bash
docker run -d -p 7078:7078 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy opea/v2a:latest
```


# Data Preperation Service 
### Build
```bash
docker build -t opea/multimedia2text:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/multimedia2text/Dockerfile .
```
### Run 
```bash
ip_address=$(hostname -I | awk '{print $1}')

docker run -it -p 7079:7079 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy \
    -e A2T_ENDPOINT=http://$ip_address:7066 \
    -e V2A_ENDPOINT=http://$ip_address:7078 \
    opea/multimedia2text:latest 

```





# Macro Service 

cd GenAIExamples\Docsum

## Build
```bash 
docker build -t opea/docsum:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f Dockerfile .

```
## Run 
cd GenAIExamples/DocSum/docker_compose/intel/cpu/xeon
```bash
docker compose up -d
```

 -->



<!-- 
ip_address=$(hostname -I | awk '{print $1}')

docker run -it -p 8888:8888 --ipc=host \
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy  \
    -e no_proxy=$no_proxy  \
    -e DATA_SERVICE_HOST_IP=http://$ip_address \
    opea/docsum:latest 

``` -->
<!-- 
    -e A2T_ENDPOINT=http://$ip_address:7066 \
    -e V2A_ENDPOINT=http://$ip_address:7078 \  
    # -e A2T_ENDPOINT=http://$ip_address:7066 \
    # -e V2A_ENDPOINT=http://$ip_address:7078 \
    # -e DATA_ENDPOINT=http://$ip_address:7079 \   
    # -e DATA_SERVICE_HOST_IP=http://$ip_address \
-->












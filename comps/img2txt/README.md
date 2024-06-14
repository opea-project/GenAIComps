# Image To Text Microservice

Image-To-Text is one of the multimodal tasks that empowered by LLMs. This microservice uses LLaVA as the base model. It accepts two inputs: a prompt and an image. It outputs the answer to the prompt about the image. It is widely used for Visual Question and Answering tasks.

# 🚀1. Start Microservice with Python (Option 1)

## 1.1 Install Requirements

```bash
pip install -r requirements.txt
```

## 1.2 Start LLaVA Service/Test

* Xeon CPU

```bash
# Start LLaVA service
cd llava/
nohup python llava_server.py --device=cpu &
# Wait until the server is up
# Test
python check_llava_server.py
```

* Gaudi2 HPU

```bash
pip install optimum[habana]
```

```bash
cd llava/
# Start LLaVA service
nohup python llava_server.py &
# Test
python check_llava_server.py
```

## 1.3 Start Image To Text Service/Test

```bash
cd ..
# Start the OPEA Microservice
python img2txt.py
# Test
python check_img2txt.py
```


# 🚀1. Start Microservice with Docker (Option 2)

## 1.2 Build Images
### 1.2.1 LLaVA Server Image
* Xeon CPU

```bash
cd ../..
docker build -t opea/llava:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/img2txt/llava/Dockerfile .
```

* Gaudi2 HPU

```bash
cd ../..
docker build -t opea/llava:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/img2txt/llava/Dockerfile_hpu .
```

### 1.2.2 Img2txt Service Image

```bash
cd ../..
docker build -t opea/img2txt:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/img2txt/Dockerfile .
```


## 1.3 Start LLaVA and Img2txt Service

### 1.3.1 Start LLaVA server


* Xeon

```bash
docker run -p 8399:8399 -e http_proxy=$http_proxy --ipc=host -e https_proxy=$https_proxy opea/llava:latest
```

* Gaudi2 HPU
```bash
docker run -p 8399:8399 --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy opea/llava:latest
```


### 1.3.2 Start Img2txt service

```bash
ip_address=$(hostname -I | awk '{print $1}')

docker run -p 9399:9399 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e IMG2TXT_ENDPOINT=http://$ip_address:8399 opea/img2txt:latest
```


### 1.3.3 Test

```bash
# Use curl/python

# curl
http_proxy="" curl http://localhost:9399/v1/img2txt -XPOST -d '{"image": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8/5+hnoEIwDiqkL4KAcT9GO0U4BxoAAAAAElFTkSuQmCC", "prompt":"What is this?"}' -H 'Content-Type: application/json'

# python
python check_img2txt.py
```
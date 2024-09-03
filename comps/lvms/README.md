# Language Vision Model (LVM) Microservice

This microservice, designed for Large Vision Model (LVM) Inference, processes input consisting of a query string and/or image prompts. It constructs a prompt based on the input, which is then used to perform inference with a large vision model (e.g., LLaVA). The service delivers the inference results as output.

A prerequisite for using this microservice is that users must have an LVM service (transformers or Prediction Guard) already running. Overall, this microservice offers a streamlined way to integrate large vision model inference into applications, requiring minimal setup from the user. This allows for the seamless processing of text and image prompts to generate intelligent, context-aware responses.

## Getting started with transformers + fastAPI services

The [transformers](transformers) directory contains instructions for running services that serve predictions from a LLaVA LVM. Two services must be spun up to run the LVM:

1. A LLaVA model server under [transformers/llava](transformers/llava)
2. An OPEA LVM service under [transformers](transformers)

See [transformers/README.md](transformers/README.md) for more information.

## Getting started with Prediction Guard

The [predictionguard](predictionguard) directory contains instructions for running a single service that serves predictions from a LLaVA LVM via the Prediction Guard framework hosted on Intel Tiber Developer Cloud (ITDC). See [predictionguard](predictionguard) for more information.
# LVM Microservice

Visual Question and Answering is one of the multimodal tasks empowered by LVMs (Large Visual Models). This microservice supports visual Q&A by using LLaVA as the base large visual model. It accepts two inputs: a prompt and an image. It outputs the answer to the prompt about the image.

## 🚀1. Start Microservice with Python (Option 1)

### 1.1 Install Requirements

```bash
pip install -r requirements.txt
```

### 1.2 Start LLaVA Service/Test

- Xeon CPU

```bash
# Start LLaVA service
cd llava/
nohup python llava_server.py --device=cpu &
# Wait until the server is up
# Test
python check_llava_server.py
```

- Gaudi2 HPU

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

### 1.3 Start Image To Text Service/Test

```bash
cd ..
# Start the OPEA Microservice
python lvm.py
# Test
python check_lvm.py
```

## 🚀2. Start Microservice with Docker (Option 2)

### 2.1 Build Images

#### 2.1.1 LLaVA Server Image

- Xeon CPU

```bash
cd ../..
docker build -t opea/llava:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/lvms/llava/Dockerfile .
```

- Gaudi2 HPU

```bash
cd ../..
docker build -t opea/llava:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/lvms/llava/Dockerfile_hpu .
```

#### 2.1.2 LVM Service Image

```bash
cd ../..
docker build -t opea/lvm:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/lvms/Dockerfile .
```

### 2.2 Start LLaVA and LVM Service

#### 2.2.1 Start LLaVA server

- Xeon

```bash
docker run -p 8399:8399 -e http_proxy=$http_proxy --ipc=host -e https_proxy=$https_proxy opea/llava:latest
```

- Gaudi2 HPU

```bash
docker run -p 8399:8399 --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy opea/llava:latest
```

#### 2.2.2 Start LVM service

```bash
ip_address=$(hostname -I | awk '{print $1}')

docker run -p 9399:9399 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e LVM_ENDPOINT=http://$ip_address:8399 opea/lvm:latest
```

#### 2.2.3 Test

```bash
# Use curl/python

# curl
http_proxy="" curl http://localhost:9399/v1/lvm -XPOST -d '{"image": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8/5+hnoEIwDiqkL4KAcT9GO0U4BxoAAAAAElFTkSuQmCC", "prompt":"What is this?"}' -H 'Content-Type: application/json'

# python
python check_lvm.py
```

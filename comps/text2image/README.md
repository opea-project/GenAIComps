# Text-to-Image Microservice

Text-to-Image is a task that generate image conditioning on the provided text. This microservice supports text-to-image task by using Stable Diffusion (SD) model.

# 🚀1. Start Microservice with Python (Option 1)

## 1.1 Install Requirements

```bash
pip install -r requirements.txt
pip install -r dependency/requirements.txt
```

## 1.2 Start SD Service

```bash
# Start SD service
cd dependency/
python sd_server.py --token $HF_TOKEN
```

## 1.3 Start Text-to-Image Microservice

```bash
cd ..
# Start the OPEA Microservice
python text2image.py
```

# 🚀2. Start Microservice with Docker (Option 2)

## 2.1 Build Images

Select Stable Diffusion (SD) model and assign its name to a environment variable as below:

```bash
# SD3
export MODEL=stabilityai/stable-diffusion-3-medium-diffusers
# SDXL
export MODEL=stabilityai/stable-diffusion-xl-base-1.0
```

### 2.1.1 SD Server Image

Build SD server image on Xeon with below command:

```bash
cd ../..
docker build -t opea/sd:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy --build-arg MODEL=$MODEL -f comps/text2image/dependency/Dockerfile .
```

Build SD server image on Gaudi with below command:

```bash
cd ../..
docker build -t opea/sd-gaudi:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy --build-arg MODEL=$MODEL -f comps/text2image/dependency/Dockerfile.intel_hpu .
```

### 2.1.2 Text-to-Image Service Image

```bash
docker build -t opea/text2image:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/text2image/Dockerfile .
```

## 2.2 Start SD and Text-to-Image Service

### 2.2.1 Start SD server

Start SD server on Xeon with below command:

```bash
docker run --ipc=host -p 9378:9378 -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e HF_TOKEN=$HF_TOKEN opea/sd:latest
```

Start SD server on Gaudi with below command:

```bash
docker run -p 9378:9378 --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e HF_TOKEN=$HF_TOKEN opea/sd-gaudi:latest
```

### 2.2.2 Start Text-to-Image service

```bash
ip_address=$(hostname -I | awk '{print $1}')
docker run -p 9379:9379 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e SD_ENDPOINT=http://$ip_address:9378 opea/text2image:latest
```

### 2.2.3 Test

```bash
http_proxy="" curl http://localhost:9379/v1/text2image -XPOST -d '{"prompt":"An astronaut riding a green horse", "num_images_per_prompt":1}' -H 'Content-Type: application/json'
```

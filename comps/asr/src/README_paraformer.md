# Deploying ASR Service

This document provides a comprehensive guide to deploying the ASR microservice pipeline with the Paraformer model on Intel platforms.

**Note:** This is an alternative of the [Whisper ASR service](./README.md). The Paraformer model supports both English and Mandarin audio input, and empirically it shows better performance in Mandarin than English.

## Table of contents

- [🚀 1. Quick Start with Docker Compose](#-1-quick-start-with-docker-compose): The recommended method for a fast and easy setup.
- [🚀 2. Manual Step-by-Step Deployment (Advanced)](#-2-manual-step-by-step-deployment-advanced): For users who want to build and run each container individually.
- [🚀 3. Start Microservice with Python](#-3-start-microservice-with-python): For users who prefer to run the ASR microservice directly with Python scripts.

## 🚀 1. Quick Start with Docker Compose

This method uses Docker Compose to start all necessary services with a single command. It is the fastest and easiest way to get the service running.

### 1.1. Access the Code

Clone the repository and navigate to the deployment directory:

```bash
git clone https://github.com/opea-project/GenAIComps.git
cd GenAIComps/comps/asr/deployment/docker_compose
```

### 1.2. Deploy the Service

Choose the command corresponding to your target platform.

```bash
export ip_address=$(hostname -I | awk '{print $1}')
export ASR_ENDPOINT=http://$ip_address:7066
export no_proxy=localhost,$no_proxy
```

- **For Intel® Core® CPU:**
  ```bash
  docker compose -f ../docker_compose/compose.yaml up funasr-paraformer-service asr-funasr-paraformer -d
  ```
  **Note:** it might take some time for `funasr-paraformer-service` to get ready, depending on the model download time in your network environment. If it fails to start with error message `dependency failed to start: container funasr-paraformer-service is unhealthy`, try increasing healthcheck retries in `GenAIComps/comps/third_parties/funasr/deployment/docker_compose/compose.yaml`

### 1.3. Validate the Service

Once the containers are running, you can validate the service. **Note:** Run these commands from the root of the `GenAIComps` repository.

```bash
# Test
wget https://github.com/intel/intel-extension-for-transformers/raw/main/intel_extension_for_transformers/neural_chat/assets/audio/sample.wav
curl http://localhost:9099/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F file="@./sample.wav" \
  -F model="paraformer-zh"
```

### 1.4. Clean Up the Deployment

To stop and remove the containers, run the following command from the `comps/asr/deployment/docker_compose` directory:

```bash
docker compose down
```

---

## 🚀 2. Manual Step-by-Step Deployment (Advanced)

This section provides detailed instructions for building the Docker images and running each microservice container individually.

### 2.1. Clone the Repository

If you haven't already, clone the repository and navigate to the root directory:

```bash
git clone https://github.com/opea-project/GenAIComps.git
cd GenAIComps
```

### 2.2. Build the Docker Images

#### 2.2.1. Build FunASR Paraformer Server Image

- **For Intel® Core® CPU:**
  ```bash
  docker build -t opea/funasr-paraformer:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/third_parties/funasr/src/Dockerfile .
  ```

#### 2.2.2. Build ASR Service Image

```bash
docker build -t opea/asr:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/asr/src/Dockerfile .
```

### 2.3 Start FunASR Paraformer and ASR Service

#### 2.3.1 Start FunASR Paraformer Server

- Core CPU

```bash
docker run -p 7066:7066 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy opea/funasr-paraformer:latest
```

#### 2.3.2 Start ASR service

```bash
ip_address=$(hostname -I | awk '{print $1}')

docker run -d -p 9099:9099 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy -e ASR_ENDPOINT=http://$ip_address:7066 opea/asr:latest
```

### 2.4 Validate the Service

After starting both containers, test the asr service endpoint. Make sure you are in the root directory of the `GenAIComps` repository.

```bash
# Use curl or python

# curl
wget https://github.com/intel/intel-extension-for-transformers/raw/main/intel_extension_for_transformers/neural_chat/assets/audio/sample.wav
curl http://localhost:9099/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F file="@./sample.wav" \
  -F model="paraformer-zh"

# python
python check_asr_server.py
```

### 2.6. Clean Up the Deployment

To stop and remove the containers you started manually, use the `docker stop` and `docker rm` commands.

- **For Intel® Core® CPU:**
  ```bash
  docker stop funasr-paraformer-service asr-funasr-paraformer-service
  docker rm funasr-paraformer-service asr-funasr-paraformer-service
  ```

## 🚀 3. Start Microservice with Python

To start the ASR microservice with Python, you need to first install python packages.

### 3.1 Install Requirements

```bash
pip install -r requirements-cpu.txt
```

### 3.2 Start FunASR Paraformer Service/Test

- Core CPU

```bash
cd comps/third_parties/funasr/src
nohup python funasr_server.py --device=cpu &
python check_funasr_server.py
```

Note: please make sure that port 7066 is not occupied by other services. Otherwise, use the command `npx kill-port 7066` to free the port.

### 3.3 Start ASR Service/Test

```bash
cd ../../..
python opea_asr_microservice.py
python check_asr_server.py
```

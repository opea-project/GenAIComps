# Deploying ASR Service

This document provides a comprehensive guide to deploying the ASR microservice pipeline on Intel platforms.

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

- **For Intel® Xeon® CPU:**

  ```bash
  docker compose -f ../deployment/docker_compose/compose.yaml up whisper-service asr-whisper -d
  ```

- **For Intel® Gaudi® 2 HPU:**
  ```bash
  docker compose -f ../deployment/docker_compose/compose.yaml up whisper-gaudi-service asr-whisper-gaudi -d
  ```

### 1.3. Validate the Service

Once the containers are running, you can validate the service. **Note:** Run these commands from the root of the `GenAIComps` repository.

```bash
# Test
wget https://github.com/intel/intel-extension-for-transformers/raw/main/intel_extension_for_transformers/neural_chat/assets/audio/sample.wav
curl http://localhost:9099/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F file="@./sample.wav" \
  -F model="openai/whisper-small"
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

#### 2.2.1. Build Whisper Server Image

- **For Intel® Xeon® CPU:**
  ```bash
  docker build -t opea/whisper:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/third_parties/whisper/src/Dockerfile .
  ```
- **For Intel® Gaudi® 2 HPU:**
  ```bash
  docker build -t opea/whisper-gaudi:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/third_parties/whisper/src/Dockerfile.intel_hpu .
  ```

#### 2.2.2. Build ASR Service Image

```bash
docker build -t opea/asr:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/asr/src/Dockerfile .
```

### 2.3 Start Whisper and ASR Service

#### 2.3.1 Start Whisper Server

- Xeon

```bash
docker run -p 7066:7066 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy opea/whisper:latest
```

- Gaudi2 HPU

```bash
docker run -p 7066:7066 --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy opea/whisper-gaudi:latest
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
  -F model="openai/whisper-small"

# python
python check_asr_server.py
```

### 2.6. Clean Up the Deployment

To stop and remove the containers you started manually, use the `docker stop` and `docker rm` commands.

- **For Intel® Xeon® CPU:**

  ```bash
  docker stop whisper-service asr
  docker rm whisper-service asr
  ```

- **For Intel® Gaudi® 2 HPU:**
  ```bash
  docker stop whisper-gaudi-service asr-whisper-gaudi
  docker rm whisper-gaudi-service asr-whisper-gaudi
  ```

## 🚀 3. Start Microservice with Python

To start the ASR microservice with Python, you need to first install python packages.

### 3.1 Install Requirements

```bash
pip install -r requirements.txt
```

### 3.2 Start Whisper Service/Test

- Xeon CPU

```bash
cd comps/third_parties/whisper/src
nohup python whisper_server.py --device=cpu &
python check_whisper_server.py
```

Note: please make sure that port 7066 is not occupied by other services. Otherwise, use the command `npx kill-port 7066` to free the port.

If the Whisper server is running properly, you should see the following output:

```bash
{'asr_result': 'Who is pat gelsinger'}
```

- Gaudi2 HPU

```bash
pip install optimum[habana]

cd comps/third_parties/whisper/src
nohup python whisper_server.py --device=hpu &
python check_whisper_server.py

# Or use openai protocol compatible curl command
# Please refer to https://platform.openai.com/docs/api-reference/audio/createTranscription
wget https://github.com/intel/intel-extension-for-transformers/raw/main/intel_extension_for_transformers/neural_chat/assets/audio/sample.wav
curl http://localhost:7066/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F file="@./sample.wav" \
  -F model="openai/whisper-small"
```

### 3.3 Start ASR Service/Test

```bash
cd ../../..
python opea_asr_microservice.py
python check_asr_server.py
```

While the Whisper service is running, you can start the ASR service. If the ASR service is running properly, you should see the output similar to the following:

```bash
{'text': 'who is pat gelsinger'}
```

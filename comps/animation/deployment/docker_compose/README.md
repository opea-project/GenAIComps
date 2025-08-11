# Deploying Avatar Animation Service

This document provides a comprehensive guide to deploying the Avatar Animation microservice pipeline on Intel platforms.

This guide covers two deployment methods:

- [🚀 1. Quick Start with Docker Compose](#-1-quick-start-with-docker-compose): The recommended method for a fast and easy setup.
- [🚀 2. Manual Step-by-Step Deployment (Advanced)](#-2-manual-step-by-step-deployment-advanced): For users who want to build and run each container individually.

## 🚀 1. Quick Start with Docker Compose

This method uses Docker Compose to start all necessary services with a single command. It is the fastest and easiest way to get the service running.

### 1.1. Access the Code

Clone the repository and navigate to the deployment directory:

```bash
git clone https://github.com/opea-project/GenAIComps.git
cd GenAIComps/comps/animation/deployment/docker_compose
```

### 1.2. Deploy the Service

Choose the command corresponding to your target platform.

- **For Intel® Xeon® CPU:**

  ```bash
  docker compose -f compose.yaml up animation -d
  ```

- **For Intel® Gaudi® 2 HPU:**
  ```bash
  docker compose -f compose.yaml up animation-gaudi -d
  ```

### 1.3. Validate the Service

Once the containers are running, you can validate the service. **Note:** Run these commands from the root of the `GenAIComps` repository.

```bash
# Navigate back to the root directory if you are in the docker_compose folder
cd ../../..

# Validate the Animation service endpoint
export ip_address=$(hostname -I | awk '{print $1}')
curl http://${ip_address}:9066/v1/animation -X POST \
-H "Content-Type: application/json" \
-d @comps/animation/src/assets/audio/sample_question.json
```

The expected output will be a JSON object containing the path to the generated video file:

```json
{ "wav2lip_result": ".../GenAIComps/comps/animation/src/assets/outputs/result.mp4" }
```

The generated video `result.mp4` will be available in the `comps/animation/src/assets/outputs/` directory.

### 1.4. Clean Up the Deployment

To stop and remove the containers, run the following command from the `comps/animation/deployment/docker_compose` directory:

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

#### 2.2.1. Build Wav2Lip Server Image

- **For Intel® Xeon® CPU:**
  ```bash
  docker build -t opea/wav2lip:latest -f comps/third_parties/wav2lip/src/Dockerfile .
  ```
- **For Intel® Gaudi® 2 HPU:**
  ```bash
  docker build -t opea/wav2lip-gaudi:latest -f comps/third_parties/wav2lip/src/Dockerfile.intel_hpu .
  ```

#### 2.2.2. Build Animation Server Image

```bash
docker build -t opea/animation:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/animation/src/Dockerfile .
```

### 2.3. Configure Environment Variables

Set the necessary environment variables for the containers.

- **For Intel® Xeon® CPU:**

  ```bash
  export ip_address=$(hostname -I | awk '{print $1}')
  export DEVICE="cpu"
  export WAV2LIP_PORT=7860
  export CHECKPOINT_PATH='/usr/local/lib/python3.11/site-packages/Wav2Lip/checkpoints/wav2lip_gan.pth'
  export PYTHON_PATH='/usr/bin/python3.11'
  ```

- **For Intel® Gaudi® 2 HPU:**
  ```bash
  export ip_address=$(hostname -I | awk '{print $1}')
  export DEVICE="hpu"
  export WAV2LIP_PORT=7860
  export CHECKPOINT_PATH='/usr/local/lib/python3.10/dist-packages/Wav2Lip/checkpoints/wav2lip_gan.pth'
  export PYTHON_PATH='/usr/bin/python3.10'
  ```

### 2.4. Run the Microservice Containers

#### 2.4.1. Run Wav2Lip Microservice

- **For Intel® Xeon® CPU:**

  ```bash
  docker run --privileged -d --name "wav2lip-service" -p $WAV2LIP_PORT:$WAV2LIP_PORT --ipc=host \
  -w /home/user/comps/animation/src \
  -v $(pwd)/comps/animation/src/assets:/home/user/comps/animation/src/assets \
  -e PYTHON=$PYTHON_PATH \
  -e DEVICE=$DEVICE \
  -e CHECKPOINT_PATH=$CHECKPOINT_PATH \
  -e WAV2LIP_PORT=$WAV2LIP_PORT \
  opea/wav2lip:latest
  ```

- **For Intel® Gaudi® 2 HPU:**
  ```bash
  docker run --privileged -d --name "wav2lip-gaudi-service" -p $WAV2LIP_PORT:$WAV2LIP_PORT --runtime=habana --cap-add=sys_nice --ipc=host \
  -w /home/user/comps/animation/src \
  -v $(pwd)/comps/animation/src/assets:/home/user/comps/animation/src/assets \
  -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
  -e PYTHON=$PYTHON_PATH \
  -e DEVICE=$DEVICE \
  -e CHECKPOINT_PATH=$CHECKPOINT_PATH \
  -e WAV2LIP_PORT=$WAV2LIP_PORT \
  opea/wav2lip-gaudi:latest
  ```

#### 2.4.2. Run Animation Microservice

```bash
docker run -d --name "animation-service" -p 9066:9066 --ipc=host \
  -e http_proxy=$http_proxy \
  -e https_proxy=$https_proxy \
  -e WAV2LIP_ENDPOINT=http://$ip_address:$WAV2LIP_PORT \
  opea/animation:latest
```

### 2.5. Validate the Service

After starting both containers, test the animation service endpoint. Make sure you are in the root directory of the `GenAIComps` repository.

```bash
# The ip_address variable should be set from step 2.3
curl http://${ip_address}:9066/v1/animation -X POST \
-H "Content-Type: application/json" \
-d @comps/animation/src/assets/audio/sample_question.json
```

You should see a successful response with the path to the output video.

### 2.6. Clean Up the Deployment

To stop and remove the containers you started manually, use the `docker stop` and `docker rm` commands.

- **For Intel® Xeon® CPU:**

  ```bash
  docker stop wav2lip-service animation-service
  docker rm wav2lip-service animation-service
  ```

- **For Intel® Gaudi® 2 HPU:**
  ```bash
  docker stop wav2lip-gaudi-service animation-service
  docker rm wav2lip-gaudi-service animation-service
  ```

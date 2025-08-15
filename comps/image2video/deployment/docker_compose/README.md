# Deploying Image-to-Video Service

This document provides a comprehensive guide to deploying the Image-to-Video microservice pipeline on Intel platforms.

This guide covers two deployment methods:

- [🚀 1. Quick Start with Docker Compose](#-1-quick-start-with-docker-compose): The recommended method for a fast and easy setup.
- [🚀 2. Manual Step-by-Step Deployment (Advanced)](#-2-manual-step-by-step-deployment-advanced): For users who want to build and run the container individually.

## 🚀 1. Quick Start with Docker Compose

This method uses Docker Compose to start the service with a single command. It is the fastest and easiest way to get the service running.

### 1.1. Access the Code

Clone the repository and navigate to the deployment directory:

```bash
git clone https://github.com/opea-project/GenAIComps.git
cd GenAIComps/comps/image2video/deployment/docker_compose
```

### 1.2. Deploy the Service

Choose the command corresponding to your target platform.

- **For Intel® Xeon® CPU:**

  ```bash
  docker compose -f compose.yaml up image2video -d
  ```

- **For Intel® Gaudi® 2 HPU:**
  ```bash
  docker compose -f compose.yaml up image2video-gaudi -d
  ```

### 1.3. Validate the Service

Once the container is running, you can validate the service. **Note:** Run this command from the root of the `GenAIComps` repository.

```bash
export ip_address=$(hostname -I | awk '{print $1}')
curl http://${ip_address}:9369/v1/image2video -XPOST \
-d '{"images_path":[{"image_path":"https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png"}]}' \
-H 'Content-Type: application/json'
```

The expected output will be a JSON object containing the path to the generated video file.

### 1.4. Clean Up the Deployment

To stop and remove the containers, run the following command from the `comps/image2video/deployment/docker_compose` directory:

```bash
docker compose down
```

---

## 🚀 2. Manual Step-by-Step Deployment (Advanced)

This section provides detailed instructions for building the Docker image and running the microservice container individually.

### 2.1. Clone the Repository

If you haven't already, clone the repository and navigate to the root directory:

```bash
git clone https://github.com/opea-project/GenAIComps.git
cd GenAIComps
```

### 2.2. Build the Docker Image

- **For Intel® Xeon® CPU:**
  ```bash
  docker build -t opea/image2video:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/image2video/src/Dockerfile .
  ```
- **For Intel® Gaudi® 2 HPU:**
  ```bash
  docker build -t opea/image2video-gaudi:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/image2video/src/Dockerfile.intel_hpu .
  ```

### 2.3. Configure Environment Variables

Set the necessary environment variables for validation.

```bash
export ip_address=$(hostname -I | awk '{print $1}')
```

### 2.4. Run the Microservice Container

#### 2.4.1. Run Image-to-Video Microservice on Xeon

```bash
docker run -d --name "image2video-service" --ipc=host -p 9369:9369 -e http_proxy=$http_proxy -e https_proxy=$https_proxy opea/image2video:latest
```

#### 2.4.2. Run Image-to-Video Microservice on Gaudi

```bash
docker run -d --name "image2video-gaudi-service" -p 9369:9369 --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy opea/image2video-gaudi:latest
```

### 2.5. Validate the Service

After starting the container, test the service endpoint. Make sure you are in the root directory of the `GenAIComps` repository.

```bash
# The ip_address variable should be set from step 2.3
curl http://${ip_address}:9369/v1/image2video -XPOST \
-d '{"images_path":[{"image_path":"https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png"}]}' \
-H 'Content-Type: application/json'
```

You should see a successful response with the path to the output video.

### 2.6. Clean Up the Deployment

To stop and remove the container you started manually, use the `docker stop` and `docker rm` commands.

- **For Intel® Xeon® CPU:**

  ```bash
  docker stop image2video-service
  docker rm image2video-service
  ```

- **For Intel® Gaudi® 2 HPU:**
  ```bash
  docker stop image2video-gaudi-service
  docker rm image2video-gaudi-service
  ```

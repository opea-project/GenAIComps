# Dataprep Microservice for Multimodal Data with Redis

This `dataprep` microservice accepts the following from the user and ingests them into a Redis vector store:

- Videos (mp4 files) and their transcripts (optional)
- Images (gif, jpg, jpeg, and png files) and their captions (optional)
- Audio (wav files)
- PDFs (with text and images)

## 🚀1. Start Microservice with Docker

### 1.1 Start Redis Stack Server

Please refer to this [readme](../../third_parties/redis/src/README.md).

### 1.2 Start LVM Microservice (Optional)

This is required only if you are going to consume the _generate_captions_ API of this microservice as described [here](#43-consume-generate_captions-api).

Please refer to this [readme](../../lvms/src/README.md) to start the LVM microservice.
After LVM is up, set up environment variables.

```bash
export your_ip=$(hostname -I | awk '{print $1}')
export LVM_ENDPOINT="http://${your_ip}:9399/v1/lvm"
```

### 1.3 Setup Environment Variables

```bash
export your_ip=$(hostname -I | awk '{print $1}')
export EMBEDDING_MODEL_ID="BridgeTower/bridgetower-large-itm-mlm-itc"
export REDIS_URL="redis://${your_ip}:6379"
export WHISPER_MODEL="base"
export INDEX_NAME=${your_redis_index_name}
export HUGGINGFACEHUB_API_TOKEN=${your_hf_api_token}
```

### 1.4 Build Docker Image

```bash
cd ../../../../
docker build -t opea/dataprep:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/src/Dockerfile .
```

### 1.5 Run Docker with CLI (Option A)

```bash
docker run -d --name="dataprep-multimodal-redis" -p 6007:5000 --runtime=runc --ipc=host -e no_proxy=$no_proxy -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e REDIS_HOST=$your_ip -e REDIS_URL=$REDIS_URL -e INDEX_NAME=$INDEX_NAME -e LVM_ENDPOINT=$LVM_ENDPOINT -e HUGGINGFACEHUB_API_TOKEN=$HUGGINGFACEHUB_API_TOKEN -e MULTIMODAL_DATAPREP=true -e DATAPREP_COMPONENT_NAME="OPEA_DATAPREP_MULTIMODALREDIS" opea/dataprep-multimodal-redis:latest
```

### 1.6 Run with Docker Compose (Option B - deprecated, will move to genAIExample in future)

```bash
cd comps/dataprep/multimodal/redis/langchain
docker compose -f compose_redis_multimodal.yaml up -d
```

## 🚀2. Status Microservice

```bash
docker container logs -f dataprep-multimodal-redis
```

## 🚀3. Consume Microservice

Once this dataprep microservice is started, user can use the below commands to invoke the microservice to convert images, videos, text, and PDF files to embeddings and save to the Redis vector store.

This microservice provides 3 different ways for users to ingest files into Redis vector store corresponding to the 3 use cases.

### 3.1 Consume _ingest_ API

**Use case:** This API is used for videos accompanied by transcript files (`.vtt` format), images accompanied by text caption files (`.txt` format), and PDF files containing a mix of text and images.

**Important notes:**

- Make sure the file paths after `files=@` are correct.
- Every transcript or caption file's name must be identical to its corresponding video or image file's name (except their extension - .vtt goes with .mp4 and .txt goes with .jpg, .jpeg, .png, or .gif). For example, `video1.mp4` and `video1.vtt`. Otherwise, if `video1.vtt` is not included correctly in the API call, the microservice will return an error `No captions file video1.vtt found for video1.mp4`.
- It is assumed that PDFs will contain at least one image. Each image in the file will be embedded along with the text that appears on the same page as the image.

#### Single video-transcript pair upload

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./video1.mp4" \
    -F "files=@./video1.vtt" \
    http://localhost:6007/v1/dataprep/ingest
```

#### Single image-caption pair upload

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./image.jpg" \
    -F "files=@./image.txt" \
    http://localhost:6007/v1/dataprep/ingest
```

#### Multiple file pair upload

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./video1.mp4" \
    -F "files=@./video1.vtt" \
    -F "files=@./video2.mp4" \
    -F "files=@./video2.vtt" \
    -F "files=@./image1.png" \
    -F "files=@./image1.txt" \
    -F "files=@./image2.jpg" \
    -F "files=@./image2.txt" \
    -F "files=@./example.pdf" \
    http://localhost:6007/v1/dataprep/ingest
```

### 3.2 Consume _generate_transcripts_ API

**Use case:** This API should be used when a video has meaningful audio or recognizable speech but its transcript file is not available, or for audio files with speech.

In this use case, this microservice will use [`whisper`](https://openai.com/index/whisper/) model to generate the `.vtt` transcript for the video or audio files.

#### Single file upload

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./video1.mp4" \
    http://localhost:6007/v1/dataprep/generate_transcripts
```

#### Multiple file upload

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./video1.mp4" \
    -F "files=@./video2.mp4" \
    -F "files=@./audio1.wav" \
    http://localhost:6007/v1/dataprep/generate_transcripts
```

### 3.3 Consume _generate_captions_ API

**Use case:** This API should be used when uploading an image, or when uploading a video that does not have meaningful audio or does not have audio.

In this use case, there is no meaningful language transcription. Thus, it is preferred to leverage a LVM microservice to summarize the frames.

- Single video upload

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./video1.mp4" \
    http://localhost:6007/v1/dataprep/generate_captions
```

- Multiple video upload

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./video1.mp4" \
    -F "files=@./video2.mp4" \
    http://localhost:6007/v1/dataprep/generate_captions
```

- Single image upload

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./image.jpg" \
    http://localhost:6007/v1/dataprep/generate_captions
```

### 3.4 Consume get API

To get names of uploaded files, use the following command.

```bash
curl -X POST \
    -H "Content-Type: application/json" \
    http://localhost:6007/v1/dataprep/get
```

### 3.5 Consume delete API

To delete uploaded files and clear the database, use the following command.

```bash
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"file_path": "all"}' \
    http://localhost:6007/v1/dataprep/delete
```

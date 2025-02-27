# LLM Native Microservice

LLM Native microservice uses [optimum-habana](https://github.com/huggingface/optimum-habana) for model initialization and warm-up, focusing solely on large language models (LLMs). It operates without frameworks like TGI/VLLM, using PyTorch directly for inference, and supports only non-stream formats. This streamlined approach optimizes performance on Habana hardware.

## 🚀1. Start Microservice

### 1.1 Setup Environment Variables

In order to start Native LLM service, you need to setup the following environment variables first.

For LLM model, both `Qwen`, `Falcon3` and `Phi4` models are supported. Users can set different models by changing the `LLM_MODEL_ID` below.

```bash
export LLM_MODEL_ID="Qwen/Qwen2-7B-Instruct"
export HF_TOKEN="your_huggingface_token"
export TEXTGEN_PORT=10512
export LLM_COMPONENT_NAME="OpeaTextGenNative"
export host_ip=${host_ip}
```

Note. If you want to run "microsoft/Phi-4-multimodal-instruct", please download the [model weights](https://huggingface.co/microsoft/Phi-4-multimodal-instruct/tree/main) manually and put at `/path/to/Phi-4-multimodal-instruct` locally, then setup following environment.

```bash
export LLM_MODEL_ID="/path/to/Phi-4-multimodal-instruct"
export LLM_COMPONENT_NAME="OpeaTextGenNativePhi4Multimodal"
```

### 1.2 Build Docker Image

```bash
## For `Qwen` and `Falcon`
dockerfile_path="comps/llms/src/text-generation/Dockerfile.intel_hpu"
export image_name="opea/llm-textgen-gaudi:latest"

## For `Phi4`
# dockerfile_path="comps/llms/src/text-generation/Dockerfile.intel_hpu_phi4"
# export image_name="opea/llm-textgen-phi4-gaudi:latest"

cd ../../../../../
docker build -t $image_name --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f $dockerfile_path .
```

To start a docker container, you have two options:

- A. Run Docker with CLI
- B. Run Docker with Docker Compose

You can choose one as needed.

### 1.3 Run Docker with CLI (Option A)

```bash
docker run -d --runtime=habana --name="llm-native-server" -p $TEXTGEN_PORT:9000 -e https_proxy=$https_proxy -e http_proxy=$http_proxy -e TOKENIZERS_PARALLELISM=false -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host -e LLM_MODEL_ID=${LLM_MODEL_ID} -e LLM_COMPONENT_NAME=$LLM_COMPONENT_NAME $image_name
```

### 1.4 Run Docker with Docker Compose (Option B)

```bash
export service_name="textgen-native-gaudi"
# export service_name="textgen-native-phi4-gaudi" # For Phi-4-mini-instruct
# export service_name="textgen-native-phi4-multimodal-gaudi" #Phi-4-multimodal-instruct
cd comps/llms/deployment/docker_compose
docker compose -f compose_text-generation.yaml up ${service_name} -d
```

## 🚀2. Consume LLM Service

### 2.1 Check Service Status

```bash
curl http://${your_ip}:9000/v1/health_check\
  -X GET \
  -H 'Content-Type: application/json'
```

### 2.2 Consume LLM Service

```bash
curl http://${your_ip}:9000/v1/chat/completions\
  -X POST \
  -d '{"messages":"What is Deep Learning?", "max_tokens":17}' \
  -H 'Content-Type: application/json'
```

If you run a multimodal model such as `Phi-4-multimodal-instruct`, you can try with image or audio input.

```bash
#image
curl http://${your_ip}:9000/v1/chat/completions\
  -X POST \
  -d '{"messages":"What is shown in this image?", "image_path":"/path/to/image", "max_tokens":17}' \
  -H 'Content-Type: application/json'

#audio
curl http://${your_ip}:9000/v1/chat/completions\
  -X POST \
  -d '{"messages":"Based on the attached audio, generate a comprehensive text transcription of the spoken content.", "audio_path":"/path/to/audio", "max_tokens":17}' \
  -H 'Content-Type: application/json'
```

# IPEX Serving microservice

[Intel® Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch) delivers advanced optimizations to accelerate Large Language Model (LLM) inference on Intel hardware. It enhances performance through techniques such as paged attention and ROPE fusion, while also supporting a range of precision formats, including FP32, BF16, Smooth Quantization INT8, and prototype weight-only quantization in INT8/INT4.

For more details, refer to the [README](https://github.com/intel/intel-extension-for-pytorch/blob/main/examples/cpu/llm/README.md)

## 🚀1. Build the Docker Image

The Dockerfile used here is primarily sourced from the IPEX project, with additions to incorporate serving capabilities for LLM inference. This Dockerfile enables SSH passwordless login, primarily for implementing distributed inference, although distributed inference is not currently applied but will be added soon.

```bash
cd ../../../../
docker build -f comps/third_parties/ipex/src/Dockerfile --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy --build-arg COMPILE=ON --build-arg PORT_SSH=2345 -t opea/ipex-llm:latest .
```

## 🚀2. Start the microservice

```bash
export MODEL_ID="microsoft/phi-4"

cd comps/third_parties/ipex/deployment/docker_compose
docker compose -f compose.yaml up -d
```

## 🚀3. Access the service

Then you need to test your service using the following commands:

```bash
http_proxy="" curl -X POST -H "Content-Type: application/json" -d '{"model": "microsoft/phi-4", "messages": [{"role": "user", "content": "Hello! What is your name?"}], "max_tokens": 128}' http://localhost:8688/v1/chat/completions
```

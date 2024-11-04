# Introduction

This Ollama server was compiled from the [official Ollama repository](https://github.com/ollama/ollama) with additional flags suitable for Intel Xeon CPU. Below are the compilation flags:
- `DGGML_AVX=on` 
- `DGGML_AVX2=on` 
- `DGGML_F16C=on` 
- `DGGML_FMA=on` 
- `DGGML_AVX512=on` 
- `DGGML_AVX512_VNNI=on` 
- `DGGML_AVX512_VBMI=on`

## Usage

1. Start the microservice

```bash
docker run --network host opea/llm-ollama-cpp-xeon:latest
```

2. Send an application/json request to the API endpoint of Ollama to interact.

```bash
curl --noproxy "*" http://localhost:11434/api/generate -d '{
  "model": "phi3",
  "prompt":"Why is the sky blue?"
}'
```

## Build Docker Image

```bash
cd comps/llms/text-generation/ollama/cpp/xeon
docker build -t opea/llm-ollama-cpp-xeon:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f Dockerfile .
```
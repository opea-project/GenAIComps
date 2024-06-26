# VLLM-Ray-Serve Endpoint Service

[Ray](https://docs.ray.io/en/latest/serve/index.html) is an LLM serving solution that makes it easy to deploy and manage a variety of open source LLMs, built on [Ray Serve](https://docs.ray.io/en/latest/serve/index.html), has native support for autoscaling and multi-node deployments, which is easy to use for LLM inference serving on Intel Gaudi2 accelerators. The Intel Gaudi2 accelerator supports both training and inference for deep learning models in particular for LLMs. Please visit [Habana AI products](<(https://habana.ai/products)>) for more details.

[vLLM](https://github.com/vllm-project/vllm) is a fast and easy-to-use library for LLM inference and serving, it delivers state-of-the-art serving throughput with a set of advanced features such as PagedAttention, Continuous batching and etc.. Besides GPUs, vLLM already supported [Intel CPUs](https://www.intel.com/content/www/us/en/products/overview.html) and [Gaudi accelerators](https://habana.ai/products).

This guide provides an example on how to launch vLLM with Ray serve endpoint on Gaudi accelerators.

## Getting Started

### Set up environment

```bash
export HUGGINGFACEHUB_API_TOKEN=${your_hf_api_token}
export vLLM_RAY_Serve_ENDPOINT="http://${your_ip}:8008"
export LLM_MODEL=${your_hf_llm_model}
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT="opea/llms"
export CHAT_PROCESSOR="ChatModelLlama"
```

### Launch VLLM Ray Gaudi Service

```bash
bash ./launch_vllm_ray_service.sh
```

For gated models such as `LLAMA-2`, you need set the environment variable `HF_TOKEN=<token>` to access the Hugging Face Hub.

Please follow this link [huggingface token](https://huggingface.co/docs/hub/security-tokens) to get the access token and export `HF_TOKEN` environment with the token.

```bash
export HF_TOKEN=<token>
```

And then you can make requests with the OpenAI-compatible APIs like below to check the service status:

```bash
curl http://127.0.0.1:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": <model_name>,
  "messages": [{"role": "user", "content": "What is deep learning?"}],
  "max_tokens": 32,
  }'
```

For more information about the OpenAI APIs, you can checkeck the [OpenAI official document](https://platform.openai.com/docs/api-reference/).

#### Customize Ray Gaudi Service

The ./serving/ray/launch_ray_service.sh script accepts five parameters:

- **port_number**: The port number assigned to the Ray Gaudi endpoint, with the default being 8080.
- model_name: The model name utilized for LLM, with the default set to "meta-llama/Llama-2-7b-chat-hf".
- chat_processor: The chat processor for handling the prompts, with the default set to "ChatModelNoFormat", and the optional selection can be "ChatModelLlama", "ChatModelGptJ" and "ChatModelGemma".
- num_cpus_per_worker: The number of CPUs specifies the number of CPUs per worker process.
- num_hpus_per_worker: The number of HPUs specifies the number of HPUs per worker process.

You have the flexibility to customize five parameters according to your specific needs. Additionally, you can set the Ray Gaudi endpoint by exporting the environment variable `VLLM_RAY_Serve_ENDPOINT`:

```bash
export VLLM_RAY_Serve_ENDPOINT="http://xxx.xxx.xxx.xxx:8008"
export LLM_MODEL=<model_name> # example: export LLM_MODEL="meta-llama/Llama-2-7b-chat-hf"
```

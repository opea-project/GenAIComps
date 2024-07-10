# VLLM-Ray-Serve Endpoint Service

[Ray](https://docs.ray.io/en/latest/serve/index.html) is an LLM serving solution that makes it easy to deploy and manage a variety of open source LLMs, built on [Ray Serve](https://docs.ray.io/en/latest/serve/index.html), has native support for autoscaling and multi-node deployments, which is easy to use for LLM inference serving on Intel Gaudi2 accelerators. The Intel Gaudi2 accelerator supports both training and inference for deep learning models in particular for LLMs. Please visit [Habana AI products](<(https://habana.ai/products)>) for more details.

[vLLM](https://github.com/vllm-project/vllm) is a fast and easy-to-use library for LLM inference and serving, it delivers state-of-the-art serving throughput with a set of advanced features such as PagedAttention, Continuous batching and etc.. Besides GPUs, vLLM already supported [Intel CPUs](https://www.intel.com/content/www/us/en/products/overview.html) and [Gaudi accelerators](https://habana.ai/products).

This guide provides an example on how to launch vLLM with Ray serve endpoint on Gaudi accelerators.

## Getting Started

### Set up environment

```bash
export HUGGINGFACEHUB_API_TOKEN=${your_hf_api_token}
export vLLM_RAY_ENDPOINT="http://${your_ip}:8006"
export LLM_MODEL=${your_hf_llm_model}
```

### Launch VLLM Ray Gaudi Service

```bash
bash ./launch_vllm_ray.sh
```

For gated models such as `LLAMA-2`, you need set the environment variable `HUGGINGFACEHUB_API_TOKEN=<token>` to access the Hugging Face Hub.

Please follow this link [huggingface token](https://huggingface.co/docs/hub/security-tokens) to get the access token and export `HUGGINGFACEHUB_API_TOKEN` environment with the token.

```bash
export HUGGINGFACEHUB_API_TOKEN=<token>
```

And then you can make requests with the OpenAI-compatible APIs like below to check the service status:

```bash
curl http://127.0.0.1:8006/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": <model_name>,
  "messages": [{"role": "user", "content": "What is deep learning?"}],
  "max_tokens": 32,
  }'
```

For more information about the OpenAI APIs, you can checkeck the [OpenAI official document](https://platform.openai.com/docs/api-reference/).

#### Customize Ray Gaudi Service

The launch_vllm_ray.sh script accepts three parameters:

- **port_number**: The port number assigned to the vLLm Ray Gaudi endpoint, with the default being 8006.
- model_name: The model name utilized for LLM, with the default set to "facebook/opt-125m".
- num_hpus_per_worker: The number of HPUs specifies the number of HPUs per worker process.

You have the flexibility to customize three parameters according to your specific needs. Additionally, you can set the Ray Gaudi endpoint by exporting the environment variable `vLLM_RAY_ENDPOINT`:

```bash
export vLLM_RAY_ENDPOINT="http://xxx.xxx.xxx.xxx:8006"
export LLM_MODEL=<model_name> # example: export LLM_MODEL="facebook/opt-125m"
```

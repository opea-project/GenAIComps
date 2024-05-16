
# Use vLLM with OpenVINO

## Build Docker Image

```bash
git clone --branch openvino-model-executor https://github.com/ilya-lavrenov/vllm.git
cd vllm
docker build -t vllm:openvino -f Dockerfile.openvino .
```

Once it successfully finishes you will have a `vllm:openvino` image. It can directly spawn a serving container with OpenAI API endpoint or you can work with it interactively via bash shell.

## Use vLLM serving with OpenAI API

_All below steps assume you are in `vllm` root directory._

### Start The Server:

```bash
# It's advised to mount host HuggingFace cache to reuse downloaded models between the runs.
docker run --rm -p 8000:8000 -v $HOME/.cache/huggingface:/root/.cache/huggingface vllm:openvino --model meta-llama/Llama-2-7b-hf --port 8000 --disable-log-requests --swap-space 50

### Additional server start up parameters that could be useful:
# --max-num-seqs <max number of sequences per iteration> (default: 256)
# --swap-space <GiB for KV cache> (default: 4)
```

### Request Completion With Curl:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "meta-llama/Llama-2-7b-hf",
  "prompt": "What is the key advantage of Openvino framework?",
  "max_tokens": 300,
  "temperature": 0.7
  }'
```

## Use Int-8 Weights Compression

Weights int-8 compression is disabled by default. For better performance and lesser memory consumption, the weights compression can be enabled by setting the environment variable `VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=1`.
To pass the variable in docker, use `-e VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=1` as an additional argument to `docker run` command in the examples above.

The variable enables weights compression logic described in [optimum-intel 8-bit weights quantization](https://huggingface.co/docs/optimum/intel/optimization_ov#8-bit).
Hence, even if the variable is enabled, the compression is applied only for models starting with a certain size and avoids compression of too small models due to a significant accuracy drop.

## Use UInt-8 KV cache Compression

KV cache uint-8 compression is disabled by default. For better performance and lesser memory consumption, the KV cache compression can be enabled by setting the environment variable `VLLM_OPENVINO_CPU_KV_CACHE_PRECISION=u8`.
To pass the variable in docker, use `-e VLLM_OPENVINO_CPU_KV_CACHE_PRECISION=u8` as an additional argument to `docker run` command in the examples above.

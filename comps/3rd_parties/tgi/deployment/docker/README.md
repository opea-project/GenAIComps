## Launch TGI endpoint

```bash
export HF_TOKEN=${your_hf_api_token}
docker run -p 8008:80 -v ./data:/data --name tgi_service --shm-size 1g ghcr.io/huggingface/text-generation-inference:2.1.0 --model-id ${your_hf_llm_model}
```

## Verify the TGI Service

```bash
curl http://${your_ip}:8008/v1/chat/completions \
     -X POST \
     -d '{"model": ${your_hf_llm_model}, "messages": [{"role": "user", "content": "What is Deep Learning?"}], "max_tokens":17}' \
     -H 'Content-Type: application/json'
```

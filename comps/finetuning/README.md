# LLM Fine-tuning Microservice

LLM Fine-tuning microservice involves adapting a base model to a specific task or dataset to improve its performance on that task.

# 🚀1. Start Microservice with Python

## 1.1 Install Requirements

```bash
pip install -r requirements.txt
```

## 1.2 Start Finetuning Service with Python Script

### 1.2.1 Start Ray Cluster

TBD

### 1.2.2 Start Finetuning Service

```bash
export RAY_ADDRESS="ray://${ray_head_ip}:10001"
python finetuning/finetuning.py
```

# 🚀2. Consume Finetuning Service

## 2.1 Check Service Status

```bash
curl http://${your_ip}:8000/v1/health_check\
  -X GET \
  -H 'Content-Type: application/json'
```

## 2.2 Create fine-tuning job

Assuming a training file `file-vGxE9KywnSUkEL6dv9qZxKAF.jsonl` is uploaded, the following script launches a finetuning job using `meta-llama/Llama-2-7b-chat-hf` as base model:

```bash
curl http://${your_ip}:8000/v1/fine_tuning/jobs \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "training_file": "file-vGxE9KywnSUkEL6dv9qZxKAF.jsonl",
    "model": "meta-llama/Llama-2-7b-chat-hf"
  }'
```

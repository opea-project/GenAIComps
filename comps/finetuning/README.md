# LLM Fine-tuning Microservice

LLM Fine-tuning microservice involves adapting a base model to a specific task or dataset to improve its performance on that task.

# 🚀1. Start Microservice with Python

## 1.1 Install Requirements

```bash
pip install -r requirements.txt
```

## 1.2 Start Finetuning Service with Python Script

### 1.2.1 Start Ray Cluster

OneCCL and Intel MPI libraries should be dynamically linked in every node before Ray starts:
```bash
source $(python -c "import oneccl_bindings_for_pytorch as torch_ccl; print(torch_ccl.cwd)")/env/setvars.sh
```

Start Ray locally using the following command.
```bash
ray start --head
```

For a multi-node cluster, start additional Ray worker nodes with below command.
```bash
ray start --address='${head_node_ip}:6379'
```

### 1.2.2 Start Finetuning Service

```bash
export RAY_ADDRESS="ray://${ray_head_ip}:10001"
python finetuning/finetuning_service.py
```

# 🚀2. Consume Finetuning Service

## 2.1 Create fine-tuning job

Assuming a training file `alpaca_data.json` is uploaded, the following script launches a finetuning job using `meta-llama/Llama-2-7b-chat-hf` as base model:

```bash
curl http://${your_ip}:8000/v1/fine_tuning/jobs \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "training_file": "alpaca_data.json",
    "model": "meta-llama/Llama-2-7b-chat-hf"
  }'
```

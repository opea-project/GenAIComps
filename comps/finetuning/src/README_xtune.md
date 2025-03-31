## 🚀1. Start Microservice with Python (Option 1)

### 1.1 Install Requirements

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install intel-extension-for-pytorch
python -m pip install oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
pip install -r requirements.txt
```

### 1.2 Install Requirements for Xtune

follow [doc](./integrations/xtune/README.md) to install requirements and download dataset.

### 1.2 Start Finetuning Service with Python Script

#### 1.2.1 Start Ray Cluster

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

#### 1.2.2 Start Finetuning Service

```bash
export HF_TOKEN=${your_huggingface_token}
export FINETUNING_COMPONENT_NAME="XTUNE_FINETUNING"
python opea_finetuning_microservice.py
```

## 🚀3. Consume Finetuning Service

#### 3.1 How to use xtune

Use the following command to launch a job for xtune:

```bash
# create a finetuning job
curl http://0.0.0.0:8015/v1/fine_tuning/jobs -X POST -H "Content-Type: application/json"   -d '{
    "training_file": "",
    "model": "vit_b16",
    "General":{
      "xtune_config":{"tool":"clip","device":"XPU", "dataset_root":"/home/data/", "trainer": "clip_adapter_hf", "dataset":"caltech101", "model":"vit_b16"}
    }
  }'
```

### 3.2 Manage fine-tuning job

Below commands show how to list finetuning jobs, retrieve a finetuning job, cancel a finetuning job and list checkpoints of a finetuning job.

```bash
# list finetuning jobs
curl http://${your_ip}:8015/v1/fine_tuning/jobs -X GET

# retrieve one finetuning job
curl http://localhost:8015/v1/fine_tuning/jobs/retrieve -X POST -H "Content-Type: application/json" -d '{"fine_tuning_job_id": ${fine_tuning_job_id}}'

# cancel one finetuning job
curl http://localhost:8015/v1/fine_tuning/jobs/cancel -X POST -H "Content-Type: application/json" -d '{"fine_tuning_job_id": ${fine_tuning_job_id}}'


```

### 3.3 Check fine-tuning job full log

```bash
cat /tmp/test.log
```

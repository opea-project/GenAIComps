# SQFT Fine-tuning Microservice

Fine-tuning microservice with SQFT involves adapting a model to a specific task or dataset to improve its performance on that task, we currently support instruction tuning for LLMs.

## 🚀1. Start Microservice with Python (Option 1)

### 1.1 Install Requirements

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install intel-extension-for-pytorch
python -m pip install oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
pip install -r requirements.txt
```
To enable elastic adapter fine-tuning (Neural Low-Rank Adapter Search) or SparsePEFT from [SQFT](https://arxiv.org/abs/2410.03750), please perform this additional installation:

```bash
PATH_TO_FINETUNE=$PWD
mkdir third_party && cd third_party

# transformers (for Neural Lora Search)
git clone https://github.com/huggingface/transformers.git
cd transformers && git checkout v4.44.2 && git apply --ignore-space-change --ignore-whitespace ${PATH_TO_FINETUNE}/patches/transformers-v4.44.2.patch && pip install -e . && cd ..

# peft (for Neural Low-Rank Adapter Search and SparsePEFT)
git clone https://github.com/huggingface/peft.git
cd peft && git checkout v0.10.0 && git apply --ignore-space-change --ignore-whitespace ${PATH_TO_FINETUNE}/patches/peft-v0.10.0.patch && pip install -e . && cd ..

# nncf (for Neural Lora Search)
git clone https://github.com/openvinotoolkit/nncf.git
cd nncf && git checkout v2.12.0 && git apply --ignore-space-change --ignore-whitespace ${PATH_TO_FINETUNE}/patches/nncf-v2.12.0.patch && pip install -e . && cd ..
```

### 1.2 Start Fine-tuning Service with Python Script

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
export HF_TOKEN=<your huggingface token>
export PYTHONPATH=<path to GenAIComps>
python finetuning_sqft_service.py
```

## 🚀2. Start Microservice with Docker (Option 2)

### 2.1 Setup on CPU

#### 2.1.1 Build Docker Image

Build docker image with below command:

```bash
export HF_TOKEN=${your_huggingface_token}
cd ../../
docker build -t opea/finetuning:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy --build-arg HF_TOKEN=$HF_TOKEN -f comps/finetuning_sqft/Dockerfile .
```

#### 2.1.2 Run Docker with CLI

Start docker container with below command:

```bash
docker run -d --name="finetuning-server" -p 8015:8015 --runtime=runc --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy opea/finetuning:latest
```

## 🚀3. Consume Fine-tuning Service

### 3.1 Upload a training file

Download a training file, such as `alpaca_data.json` for instruction tuning and upload it to the server with below command, this file can be downloaded in [here](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json):

```bash
# upload a training file
curl http://${your_ip}:8015/v1/files -X POST -H "Content-Type: multipart/form-data" -F "file=@./alpaca_data.json" -F purpose="fine-tune"
```


### 3.2 Create fine-tuning job

#### 3.2.1 Instruction Tuning

After a training file like `alpaca_data.json` is uploaded, use the following command to launch a fine-tuning job using `meta-llama/Llama-2-7b-chat-hf` as base model:

```bash
# create a finetuning job
curl http://${your_ip}:8015/v1/fine_tuning/jobs \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "training_file": "alpaca_data.json",
    "model": "meta-llama/Llama-2-7b-chat-hf"
  }'
 
# create a finetuning job (with SparsePEFT)
curl http://${your_ip}:8015/v1/fine_tuning/jobs \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "training_file": "alpaca_data.json",
    "model": <path to sparse model>,
    "General": {
      "lora_config": {
        "sparse_adapter": true
      }
    }
  }'
  
# create a fine-tuning job (with Neural Low-rank adapter Search)
# Max LoRA rank: 16
#   LoRA target modules            -> Low-rank search space
#   ["q_proj", "k_proj", "v_proj"] -> [16,12,8]
#   ["up_proj"]                    -> [16,12,8]
#   ["down_proj"]                  -> [16,12,8]
curl http://${your_ip}:8015/v1/fine_tuning/jobs \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "training_file": "alpaca_data.json",
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "General": {
      "lora_config": {
        "r": 16,
        "neural_lora_search": true,
        "target_module_groups": [["q_proj", "k_proj", "v_proj"], ["up_proj"], ["down_proj"]],
        "search_space": ["16,12,8", "16,12,8", "16,12,8"]
      }
    }
  }'
```

Below are some explanations for the parameters related to the Neural Low-rank adapter Search algorithm:

- `target_module_groups` specifies the target module groups, which means that the adapters within the same group will share the same activated low-rank value.
- `search_space` specifies the search space for each target module (adapter) group. 
Here, it is `["16,12,8", "16,12,8", "16,12,8"]`, meaning that the search space for each group is [16, 12, 8].

Note that the number of groups should be equal to the number of search spaces (one-to-one correspondence).
Feel free to try your favorite group design and search spaces.

### 3.3 Manage fine-tuning job

Below commands show how to list fine-tuning jobs, retrieve a fine-tuning job, cancel a fine-tuning job and list checkpoints of a fine-tuning job.

```bash
# list fine-tuning jobs
curl http://${your_ip}:8015/v1/fine_tuning/jobs -X GET

# retrieve one fine-tuning job
curl http://localhost:8015/v1/fine_tuning/jobs/retrieve -X POST -H "Content-Type: application/json" -d '{"fine_tuning_job_id": ${fine_tuning_job_id}}'

# cancel one fine-tuning job
curl http://localhost:8015/v1/fine_tuning/jobs/cancel -X POST -H "Content-Type: application/json" -d '{"fine_tuning_job_id": ${fine_tuning_job_id}}'

# list checkpoints of a fine-tuning job
curl http://${your_ip}:8015/v1/finetune/list_checkpoints -X POST -H "Content-Type: application/json" -d '{"fine_tuning_job_id": ${fine_tuning_job_id}}'
```

### 3.4 Leverage fine-tuned model

#### 3.4.1 Extract the sub-adapter

After completing the super-adapter fine-tuning (the checkpoints of the fine-tuning job), 
the following command demonstrates how to extract the heuristic sub-adapter.
Additionally, more powerful sub-adapters can be obtained through other advanced search algorithms.

```bash
curl http://${your_ip}:8015/v1/finetune/extract_adapter \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "fine_tuning_job_id": ${fine_tuning_job_id},
    "sub_adapter_version": "heuristic"
  }'
```

`sub_adapter_version` can be heuristic, minimal, or a custom name.
When `sub_adapter_version` is set to a custom name, we need to provide a specific configuration in `custom_config`.
The extracted adapter will be saved in `<path to output directory> / <sub_adapter_version>`.

<details>
<summary>An example of a custom configuration </summary>

```bash
curl http://${your_ip}:8015/v1/finetune/extract_adapter \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "fine_tuning_job_id": ${fine_tuning_job_id},
    "sub_adapter_version": "custom",
    "custom_config": [8, 8, 16, 8, 8, 12, 8, 12, 12, 12, 8, 16, 12, 16, 16, 12, 12, 8, 8, 16, 8, 8, 12, 8, 16, 12, 8, 16, 8, 16, 12, 8, 8, 16, 16, 16, 16, 16, 8, 12, 12, 16, 12, 16, 12, 16, 16, 12, 8, 12, 12, 8, 8, 12, 8, 12, 12, 8, 16, 8, 8, 8, 8, 12, 16, 16],
  }'
```

In the fine-tuning job with Neural Low-rank adapter Search algorithm,  the `nncf_config.json` file (which includes the elastic adapter information) will be saved in the output directory.
The `custom_config` must correspond with the `overwrite_groups` (adapter modules) or `overwrite_groups_widths`
(search space for the rank of adapter modules) in `nncf_config.json`. 
The above command corresponds to the example in [example_nncf_config/nncf_config.json](./example_nncf_config/nncf_config.json), 
and it will save the sub-adapter to `<path to output directory> / custom`.

</details>

#### 3.4.2 Merge

The following command demonstrates how to merge the sub-adapter to the base pretrained model:

```bash
curl http://${your_ip}:8015/v1/ffinetune/merge_adapter \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "fine_tuning_job_id": ${fine_tuning_job_id},
    "sub_adapter_version": "heuristic"
  }'
```

The merged model will be saved in `<path to output directory> / <sub_adapter_version> / merged_model`.

## 🚀4. Descriptions for Finetuning parameters

We utilize [OpenAI finetuning parameters](https://platform.openai.com/docs/api-reference/fine-tuning) and extend it with more customizable parameters, see the definitions at [finetune_sqft_config](./finetune_sqft_config.py).

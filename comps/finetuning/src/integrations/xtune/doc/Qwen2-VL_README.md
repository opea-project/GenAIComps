# Qwen2-VL Training and Hyperparameter Optimization

This is a flexible Qwen2-VL model training and Optuna hyperparameter optimization script that supports multiple running modes.

## 🚀 Features

- **Optuna Hyperparameter Optimization**: Automatically search for optimal `learning rate` and `warmup steps`of best `bleu-4` result
- **Train+Eval Mode**: Finetune and evaluate with fixed parameters
- **Train-Only Mode**: Only perform model finetune without evaluation
- **Flexible Configuration**: Support nested JSON configuration files and command-line parameter overrides
- **Fintune by command lines**: You can also finetune qwen2-vl by command lines.

## 📋 Running Modes

The script automatically selects running mode based on parameters present in the configuration file:

1. **Optuna Optimization Mode**: When configuration file contains `optuna` parameters
2. **Train+Eval Mode**: When configuration file contains `training` and `evaluation` parameters (without `optuna`)
3. **Train-Only Mode**: When configuration file only contains `training` parameters

## 🛠️ Usage

### 1. Optuna Hyperparameter Optimization Mode

Create configuration file `config_optuna.json`:

```json
{
  "base": {
    "working_directory": "your work dir",
    "base_output_dir": "./saves/optuna_tuning",
    "model_base_path": "./models",
    "model_name": "Qwen2-VL-7B-Instruct-GPTQ-Int8",
    "print_detail_log": true
  },

  "optuna": {
    "n_trials": 10,
    "sampler_name": "TPESampler",
    "study_name": "qwen2vl_hyperparameter_tuning",
    "learning_rate_min": 1e-6,
    "learning_rate_max": 1e-3,
    "warmup_steps_min": 50,
    "warmup_steps_max": 500
  },

  "training": {
    "train_dataset": "activitynet_train_qa_5",
    "n_epochs": 5,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "train_cutoff_len": 2048,
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 1.0,
    "logging_steps": 10,
    "save_steps": 500,
    "optim": "adamw_torch",
    "preprocessing_num_workers": 16,
    "bf16": true,
    "finetuning_type": "lora",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "lora_target": "all",
    "max_samples": 100000,
    "video_fps": 0.1
  },

  "evaluation": {
    "eval_dataset": "activitynet_qa_val_500_limit_20s",
    "per_device_eval_batch_size": 1,
    "predict_with_generate": true,
    "eval_cutoff_len": 1024,
    "max_new_tokens": 128,
    "top_p": 0.7,
    "temperature": 0.95,
    "quantization_method": "bitsandbytes",
    "max_samples": 100000,
    "video_fps": 0.1
  }
}
```

Run:

```bash
python optuna_tuning.py --config_file config_optuna.json
```

#### Visualization

If you want to see visual results of Optuna:
Run:

```sh
sudo ufw allow 8084
optuna-dashboard --host 0.0.0.0 --port 8084 sqlite:///./saves/optuna_tuning/optuna_study.db
```

Open in the website:

```
http://<serverIP>:8084/dashboard
```

### 2. Train+Eval Mode

Create configuration file `config_finetune_eval.json` (without `optuna` parameters):

```json
{
  "base": {
    "working_directory": "your work dir",
    "base_output_dir": "./saves/training_run",
    "model_base_path": "./models",
    "model_name": "Qwen2-VL-7B-Instruct-GPTQ-Int8",
    "print_detail_log": false
  },

  "training": {
    "train_dataset": "activitynet_qa_2000_limit_20s",
    "n_epochs": 5,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "train_cutoff_len": 2048,
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 1.0,
    "logging_steps": 10,
    "save_steps": 500,
    "optim": "adamw_torch",
    "preprocessing_num_workers": 16,
    "bf16": true,
    "finetuning_type": "lora",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "lora_target": "all",
    "max_samples": 100000,
    "video_fps": 0.1,
    "learning_rate": 5e-5,
    "warmup_steps": 100
  },

  "evaluation": {
    "eval_dataset": "activitynet_qa_val_500_limit_20s",
    "per_device_eval_batch_size": 1,
    "predict_with_generate": true,
    "eval_cutoff_len": 1024,
    "max_new_tokens": 128,
    "top_p": 0.7,
    "temperature": 0.95,
    "quantization_method": "bitsandbytes",
    "max_samples": 100000,
    "video_fps": 0.1
  }
}
```

Run:

```bash
python optuna_tuning.py --config_file config_finetune_eval.json
```

### 3. Train-Only Mode

Create configuration file `config_finetune_only.json` (without `optuna` and `evaluation` parameters):

```json
{
  "base": {
    "working_directory": "your work dir",
    "base_output_dir": "./saves/training_run",
    "model_base_path": "./models",
    "model_name": "Qwen2-VL-7B-Instruct-GPTQ-Int8",
    "print_detail_log": false
  },

  "training": {
    "train_dataset": "activitynet_qa_2000_limit_20s",
    "n_epochs": 5,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "train_cutoff_len": 2048,
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 1.0,
    "logging_steps": 10,
    "save_steps": 500,
    "optim": "adamw_torch",
    "preprocessing_num_workers": 16,
    "bf16": true,
    "finetuning_type": "lora",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "lora_target": "all",
    "max_samples": 100000,
    "video_fps": 0.1,
    "learning_rate": 5e-5,
    "warmup_steps": 100
  }
}
```

Run:

```bash
python optuna_tuning.py --config_file config_finetune_only.json
```

### 4. Command line finetune

##

```
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /model/Qwen2-VL-7B-Instruct-GPTQ-Int8 \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen2_vl \
    --flash_attn auto \
    --dataset_dir data \
    --dataset activitynet_qa_2000_limit_20s \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 20.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir saves/Qwen2-VL-7B-Instruct-GPTQ-Int8/lora/finetune_test_valmetrics_evalstep8 \
    --bf16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --optim adamw_torch \
    --video_fps 0.1 \
    --per_device_eval_batch_size 1 \
    --eval_strategy steps \
    --eval_steps 100 \
    --eval_dataset activitynet_qa_val_500_limit_20s \
    --predict_with_generate true \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all
```

## ⚙️ Configuration Parameters

### Base Configuration (base)

| Parameter           | Description                                                           | Default Value                    | Type    |
| ------------------- | --------------------------------------------------------------------- | -------------------------------- | ------- |
| `working_directory` | Working directory for running commands                                | "./"                             | string  |
| `base_output_dir`   | Base directory for saving results                                     | "./saves/optuna_tuning"          | string  |
| `model_base_path`   | Base path for model files                                             | "./models"                       | string  |
| `model_name`        | Model name                                                            | "Qwen2-VL-7B-Instruct-GPTQ-Int8" | string  |
| `print_detail_log`  | Whether to print detailed finetune or evaluation logs of LLamaFactory | true                             | boolean |

### Optuna Configuration (optuna) - Required only for optimization mode

| Parameter           | Description                                  | Default Value                   | Type    |
| ------------------- | -------------------------------------------- | ------------------------------- | ------- |
| `n_trials`          | Number of Optuna trials                      | 10                              | integer |
| `sampler_name`      | Optuna sampler name                          | "TPESampler"                    | string  |
| `study_name`        | Optuna study name                            | "qwen2vl_hyperparameter_tuning" | string  |
| `learning_rate_min` | Minimum value for learning rate search range | 1e-6                            | float   |
| `learning_rate_max` | Maximum value for learning rate search range | 1e-3                            | float   |
| `warmup_steps_min`  | Minimum value for warmup steps search range  | 50                              | integer |
| `warmup_steps_max`  | Maximum value for warmup steps search range  | 500                             | integer |

### Training Configuration (training)

| Parameter                     | Description                                  | Default Value                   | Type    |
| ----------------------------- | -------------------------------------------- | ------------------------------- | ------- |
| `n_epochs`                    | Number of training epochs                    | 8                               | integer |
| `train_dataset`               | Training dataset name                        | "activitynet_qa_2000_limit_20s" | string  |
| `per_device_train_batch_size` | Training batch size per device               | 2                               | integer |
| `gradient_accumulation_steps` | Gradient accumulation steps                  | 8                               | integer |
| `lr_scheduler_type`           | Learning rate scheduler type                 | "cosine"                        | string  |
| `train_cutoff_len`            | Training sequence cutoff length              | 2048                            | integer |
| `max_grad_norm`               | Maximum gradient clipping                    | 1.0                             | float   |
| `logging_steps`               | Logging interval steps                       | 10                              | integer |
| `save_steps`                  | Checkpoint saving interval steps             | 500                             | integer |
| `optim`                       | Optimizer                                    | "adamw_torch"                   | string  |
| `preprocessing_num_workers`   | Number of preprocessing workers              | 16                              | integer |
| `bf16`                        | Whether to use bf16 precision                | true                            | boolean |
| `finetuning_type`             | Fine-tuning type                             | "lora"                          | string  |
| `lora_rank`                   | LoRA rank                                    | 8                               | integer |
| `lora_alpha`                  | LoRA alpha                                   | 16                              | integer |
| `lora_dropout`                | LoRA dropout                                 | 0.0                             | float   |
| `lora_target`                 | LoRA target modules                          | "all"                           | string  |
| `max_samples`                 | Maximum number of samples                    | 100000                          | integer |
| `video_fps`                   | Video frame rate                             | 0.1                             | float   |
| `learning_rate`               | Learning rate (required for non-Optuna mode) | null                            | float   |
| `warmup_steps`                | Warmup steps (required for non-Optuna mode)  | null                            | integer |

### Evaluation Configuration (evaluation) - Optional

| Parameter                    | Description                              | Default Value                      | Type    |
| ---------------------------- | ---------------------------------------- | ---------------------------------- | ------- |
| `eval_dataset`               | Evaluation dataset name                  | "activitynet_qa_val_500_limit_20s" | string  |
| `per_device_eval_batch_size` | Evaluation batch size per device         | 1                                  | integer |
| `predict_with_generate`      | Whether to use generation for prediction | true                               | boolean |
| `eval_cutoff_len`            | Evaluation sequence cutoff length        | 1024                               | integer |
| `max_new_tokens`             | Maximum number of new generated tokens   | 128                                | integer |
| `top_p`                      | Top-p sampling parameter                 | 0.7                                | float   |
| `temperature`                | Temperature parameter                    | 0.95                               | float   |
| `quantization_method`        | Quantization method                      | "bitsandbytes"                     | string  |
| `max_samples`                | Maximum number of samples                | 100000                             | integer |
| `video_fps`                  | Video frame rate                         | 0.1                                | float   |

## 🔍 Optuna features

### Supported Parameters

- `learning_rate` and `warmup_steps`.

**Recommended configuration example:**

```json
"optuna": {
  "n_trials": 20,
  "sampler_name": "TPESampler",
  "study_name": "my_tuning_experiment",
  "learning_rate_min": 5e-7,
  "learning_rate_max": 5e-4,
  "warmup_steps_min": 100,
  "warmup_steps_max": 800
}
```

### Supported Samplers

| Sampler Name          | Description                                     | Use Case                                                            |
| --------------------- | ----------------------------------------------- | ------------------------------------------------------------------- |
| `TPESampler`          | Tree-structured Parzen Estimator                | Default recommended, suitable for continuous parameter optimization |
| `RandomSampler`       | Random sampling                                 | Quick exploration, baseline comparison                              |
| `CmaEsSampler`        | Covariance Matrix Adaptation Evolution Strategy | Continuous optimization problems                                    |
| `GPSampler`           | Gaussian Process sampler                        | Expensive objective function evaluation                             |
| `PartialFixedSampler` | Partial fixed parameter sampler                 | Scenarios with some fixed parameters                                |
| `NSGAIISampler`       | Non-dominated Sorting Genetic Algorithm II      | Multi-objective optimization                                        |
| `QMCSampler`          | Quasi-Monte Carlo sampler                       | Low discrepancy sequences, uniform exploration                      |

## 💡 Log Control

Control subprocess output verbosity through the `print_detail_log` parameter:

- `true`: Show complete training and evaluation process logs of LLamaFactory.
- `false`: Hide detailed output, show only key information and errors (recommended for batch runs)

## 📊 Output and Results

### Optuna Optimization Mode Output

- Training results: `{base_output_dir}/{study_name}/trial_XXX/`
- Optuna database: `{base_output_dir}/optuna_study.db`
- Best parameters: `{base_output_dir}/{study_name}/best_params.json`

### Training Mode Output

- Training results: `{base_output_dir}/{model_name}/{timestamp}/`
- Evaluation results: `{base_output_dir}/{model_name}/{timestamp}/evaluation/`

### Evaluation Metrics

The script returns all evaluation metrics, including but not limited to:

- `predict_bleu-4`: BLEU-4 score
- `predict_rouge-1`: ROUGE-1 score
- `predict_rouge-2`: ROUGE-2 score
- `predict_rouge-l`: ROUGE-L score

## 📝 Complete Examples

Here are three complete configuration file examples:

### 1. Optuna Optimization Example

```bash
# Run 10 trials to optimize learning rate and warmup steps
python optuna_tuning.py --config_file ./cfgs/config_optuna.json
```

### 2. Fixed Parameter Train+Eval Example

```bash
# Train and evaluate with fixed parameters
python optuna_tuning.py --config_file ./cfgs/config_finetune_eval.json
```

### 3. Train-Only Example

```bash
# Only perform training without evaluation
python optuna_tuning.py --config_file ./cfgs/config_finetune_only.json
```

## Finetune qwen2-vl with logging eval loss

If you want to finetune with plotting eval loss, please set `eval_strategy` as `steps`, `eval_steps`and `eval_dataset`:
Please change the dateset to

```
export DATA='where you can find dataset_info.json'
export dataset=activitynet_qa_2000_limit_20s                    # to point which dataset llamafactory will use
export eval_dataset=activitynet_qa_val_500_limit_20s
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path $models/Qwen2-VL-7B-Instruct-GPTQ-Int8 \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen2_vl \
    --flash_attn auto \
    --dataset_dir $DATA \
    --dataset $dataset \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 20.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 100 \
    --warmup_steps 100 \
    --packing False \
    --report_to none \
    --output_dir saves/Qwen2-VL-7B-Instruct-GPTQ-Int8/lora/finetune_test_valmetrics_evalstep8 \
    --bf16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --optim adamw_torch \
    --video_fps 0.1 \
    --per_device_eval_batch_size 1 \
    --eval_strategy steps \
    --eval_steps 100 \
    --eval_dataset ${eval_dataset} \
    --predict_with_generate true \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all
```

### Evaluation metrics calculation and plotting

If you want to plot eval metrics:
Change `MODEL_NAME`,`EXPERIENT_NAME`,`EVAL_DATASET` as you need and run evaluation metrics calculation sctrpt:

```
export MODEL_DIR = where can find eval model
export MODEL_NAME="Qwen2-VL-2B-Instruct"
export EXPERIENT_NAME="finetune_onlyplot_evalloss_5e-6"
export EVAL_DATASET=activitynet_qa_val_500_limit_20s
chmod a+x ./doc/run_eval.sh
./doc/run_eval.sh
```

Change `model_name` and `experiment_name` then run:

```
python plot_metrics.py --model_name your_model_name --experiment_name your_experiment_name
```

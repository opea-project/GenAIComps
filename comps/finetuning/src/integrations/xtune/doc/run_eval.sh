#!/bin/bash

# Path to the checkpoints directory
MODEL_NAME=${MODEL_NAME: -"Qwen2-VL-2B-Instruct"}
EXPERIENT_NAME=${EXPERIENT_NAME: -"finetune_onlyplot_evalloss_5e-6"}
CHECKPOINTS_DIR="saves/${MODEL_NAME}/lora/${EXPERIENT_NAME}"
EVAL_STR="val_500_limit_20s"
EVAL_DATASET=${EVAL_DATASET: -"activitynet_qa_val_500_limit_20s"}
# Find all checkpoint directories
checkpoints=$(find "${CHECKPOINTS_DIR}" -maxdepth 1 -type d -name "checkpoint-*" | sort -V)
echo "Found checkpoints:"
echo "$checkpoints"

# Check if any checkpoints were found
if [ -z "$checkpoints" ]; then
    echo "No checkpoints found in ${CHECKPOINTS_DIR}"
    exit 1
fi
# If no checkpoints found, exit
if [ -z "$checkpoints" ]; then
    echo "No checkpoints found in ${CHECKPOINTS_DIR}"
    exit 1
fi

# Loop through each checkpoint and run evaluation sequentially
for checkpoint_path in $checkpoints; do
    # Extract checkpoint name (e.g., checkpoint-100)
    checkpoint=$(basename ${checkpoint_path})
    
    echo "Running evaluation on ${checkpoint}..."
    
    # Create output directory for this checkpoint's evaluation
    eval_output_dir="saves/${MODEL_NAME}/lora/${EXPERIENT_NAME}/eval_${EVAL_STR}"
    #if exists,skip this checkpoint
    if [ -d "${eval_output_dir}/eval_${checkpoint}" ]; then
        echo "Skipping ${checkpoint}, already evaluated."
        continue
    fi
    mkdir -p "${eval_output_dir}"
    output_dir="${eval_output_dir}/eval_${checkpoint}"
    # Run evaluation command and wait for it to complete
    llamafactory-cli train \
        --stage sft \
        --model_name_or_path ${MODEL_DIR}/${MODEL_NAME} \
        --preprocessing_num_workers 16 \
        --finetuning_type lora \
        --quantization_method bitsandbytes \
        --template qwen2_vl \
        --flash_attn auto \
        --dataset_dir data \
        --eval_dataset ${EVAL_DATASET} \
        --cutoff_len 1024 \
        --max_samples 100000 \
        --per_device_eval_batch_size 1 \
        --predict_with_generate True \
        --max_new_tokens 128 \
        --top_p 0.7 \
        --temperature 0.95 \
        --output_dir "${output_dir}" \
        --do_predict True \
        --adapter_name_or_path "${checkpoint_path}" \
        --video_fps 0.1 \
        --report_to none
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Evaluation completed successfully for ${checkpoint}"
    else
        echo "ERROR: Evaluation failed for ${checkpoint}"
        # You can choose to exit on failure or continue with the next checkpoint
        # exit 1  # Uncomment to exit on first failure
    fi
    
    echo "----------------------------------------"
    # Wait a moment before starting the next one
    sleep 5
done

echo "All evaluations completed!"
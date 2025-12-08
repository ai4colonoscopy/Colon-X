#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "============================================================"
echo "                    STARTING STAGE 1                        "
echo "============================================================"


IMAGE_ROOT=cache/data
S1_OUTPUT_FILE=ColonR1-Qwen2.5-VL-GRPO-thinking-StageI
S1_OUTPUT_DIR=cache/checkpoints/ft-exp/$S1_OUTPUT_FILE
S1_JSON_FILE=cache/data/JSON/Train-Val-merge/ColonReason_GRPO.json
S1_BASE_MODEL=cache/download-weights/Qwen2.5-VL-3B-Instruct

mkdir -p $S1_OUTPUT_DIR

torchrun --nproc_per_node=4 --master_port=29500 ColonR1/train/grpo_vqa_thinking.py \
    --deepspeed ColonR1/script/deepspeed_configs/zero3.json \
    --output_dir $S1_OUTPUT_DIR \
    --model_name_or_path $S1_BASE_MODEL \
    --dataset_name $S1_JSON_FILE \
    --image_root_dir $IMAGE_ROOT \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_generations 4 \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --save_steps 1000 \
    --save_only_model true \
    --learning_rate 2e-6 \
    --initial_beta 0.6 \
    --final_beta 0.01 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.01 \
    --bad_case_log_file $S1_OUTPUT_DIR/bad_case.json \
    --report_to none | tee 2>&1 cache/log/stdout-${S1_OUTPUT_FILE}.txt

echo "Stage 1 Training finished."

echo "Starting bad case merging process..."

BAD_CASE_JSON=$S1_OUTPUT_DIR/bad_case.json
ORIGINAL_DATA_FILE=$S1_JSON_FILE
S2_JSON_FILE=$S1_OUTPUT_DIR/bad_case_format.json

python ColonR1/serve/merge_bad_cases.py \
        --bad_case_file $BAD_CASE_JSON \
        --original_file $ORIGINAL_DATA_FILE \
        --output_file $S2_JSON_FILE

echo "Bad case merging finished. Stage 2 input JSON created at $S2_JSON_FILE"

echo "============================================================"
echo "                 STARTING STAGE 2                           "
echo "============================================================"

S2_OUTPUT_FILE=ColonR1-Qwen2.5-VL-GRPO-thinking-StageII
S2_OUTPUT_DIR=cache/checkpoints/ft-exp/$S2_OUTPUT_FILE
S2_BASE_MODEL=$S1_OUTPUT_DIR

mkdir -p $S2_OUTPUT_DIR

torchrun --nproc_per_node=4 --master_port=29510 ColonR1/train/grpo_vqa_thinking.py \
    --deepspeed ColonR1/script/zero3.json \
    --output_dir $S2_OUTPUT_DIR \
    --model_name_or_path $S2_BASE_MODEL \
    --dataset_name $S2_JSON_FILE \
    --image_root_dir $IMAGE_ROOT \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_generations 4 \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --save_steps 1000 \
    --save_only_model true \
    --learning_rate 2e-6 \
    --initial_beta 0.2 \
    --final_beta 0.01 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.01 \
    --bad_case_log_file $S2_OUTPUT_DIR/bad_case.json \
    --report_to none | tee 2>&1 cache/log/stdout-${S2_OUTPUT_FILE}.txt

echo "Stage 2 Training finished."
echo "============================================================"
echo "            All stages completed successfully."
echo "============================================================"
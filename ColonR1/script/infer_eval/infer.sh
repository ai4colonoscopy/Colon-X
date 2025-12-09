#!/bin/bash

IMAGE_BASE_PATH=cache/data
ROOT_PATH=cache/data/JSON/Test
EXP_MODEL_ID=cache/checkpoints/ft-exp/ColonR1-Qwen2.5-VL-GRPO-thinking-StageII

mkdir -p $EXP_MODEL_ID/pred

# --------------------------- GPU 0 --------------------------- #
export CUDA_VISIBLE_DEVICES=0

nohup python ColonR1/serve/inference.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_1_ColonEval.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_1_ColonEval.json > $EXP_MODEL_ID/pred/nohup-pred_task1.txt 2>&1 &

# --------------------------- GPU 1 --------------------------- #
export CUDA_VISIBLE_DEVICES=1

nohup python ColonR1/serve/inference.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_2_ColonEval.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_2_ColonEval.json > $EXP_MODEL_ID/pred/nohup-pred_task2.txt 2>&1 &

# --------------------------- GPU 2 --------------------------- #
export CUDA_VISIBLE_DEVICES=2

nohup python ColonR1/serve/inference.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_3_ColonEval.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_3_ColonEval.json > $EXP_MODEL_ID/pred/nohup-pred_task3.txt 2>&1 &

# --------------------------- GPU 3 --------------------------- #
export CUDA_VISIBLE_DEVICES=3

nohup python ColonR1/serve/inference.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_4_ColonEval.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_4_ColonEval.json > $EXP_MODEL_ID/pred/nohup-pred_task4.txt 2>&1 &

# --------------------------- GPU 4 --------------------------- #
export CUDA_VISIBLE_DEVICES=4

nohup python ColonR1/serve/inference.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_5_ColonEval.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_5_ColonEval.json > $EXP_MODEL_ID/pred/nohup-pred_task5.txt 2>&1 &

# --------------------------- GPU 5 --------------------------- #
export CUDA_VISIBLE_DEVICES=5

nohup python ColonR1/serve/inference.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_6_ColonEval.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_6_ColonEval.json > $EXP_MODEL_ID/pred/nohup-pred_task6.txt 2>&1 &

# --------------------------- GPU 6 --------------------------- #
export CUDA_VISIBLE_DEVICES=6

nohup python ColonR1/serve/inference.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_7_ColonEval.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_7_ColonEval.json > $EXP_MODEL_ID/pred/nohup-pred_task7.txt 2>&1 &

# --------------------------- GPU 7 --------------------------- #
export CUDA_VISIBLE_DEVICES=7

nohup python ColonR1/serve/inference.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_8_ColonEval.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_8_ColonEval.json > $EXP_MODEL_ID/pred/nohup-pred_task8.txt 2>&1 &
# ------------------------------------------------------------- #




# --------------------------- GPU 0 --------------------------- #
export CUDA_VISIBLE_DEVICES=0

nohup python ColonR1/serve/inference.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_9_ColonEval.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_9_ColonEval.json > $EXP_MODEL_ID/pred/nohup-pred_task9.txt 2>&1 &

# --------------------------- GPU 1 --------------------------- #
export CUDA_VISIBLE_DEVICES=1

nohup python ColonR1/serve/inference.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_10_ColonEval.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_10_ColonEval.json > $EXP_MODEL_ID/pred/nohup-pred_task10.txt 2>&1 &

# --------------------------- GPU 2 --------------------------- #
export CUDA_VISIBLE_DEVICES=2

nohup python ColonR1/serve/inference.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_11_ColonEval.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_11_ColonEval.json > $EXP_MODEL_ID/pred/nohup-pred_task11.txt 2>&1 &

# --------------------------- GPU 3 --------------------------- #
export CUDA_VISIBLE_DEVICES=3

nohup python ColonR1/serve/inference.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_12_ColonEval.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_12_ColonEval.json > $EXP_MODEL_ID/pred/nohup-pred_task12.txt 2>&1 &

# --------------------------- GPU 4 --------------------------- #
export CUDA_VISIBLE_DEVICES=4

nohup python ColonR1/serve/inference.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_14_ColonEval.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_14_ColonEval.json > $EXP_MODEL_ID/pred/nohup-pred_task14.txt 2>&1 &

# --------------------------- GPU 5 --------------------------- #
export CUDA_VISIBLE_DEVICES=5

nohup python ColonR1/serve/inference.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_15_ColonEval.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_15_ColonEval.json > $EXP_MODEL_ID/pred/nohup-pred_task15.txt 2>&1 &

# --------------------------- GPU 6 --------------------------- #
export CUDA_VISIBLE_DEVICES=6

nohup python ColonR1/serve/inference.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_16_ColonEval.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_16_ColonEval.json > $EXP_MODEL_ID/pred/nohup-pred_task16.txt 2>&1 &

# --------------------------- GPU 7 --------------------------- #
export CUDA_VISIBLE_DEVICES=7

nohup python ColonR1/serve/inference.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_17_ColonEval.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_17_ColonEval.json > $EXP_MODEL_ID/pred/nohup-pred_task17.txt 2>&1 &
# ------------------------------------------------------------- #

#!/bin/bash

MODEL=o4-mini-2025-04-16
IMAGE_BASE_PATH=cache/data
ROOT_PATH=cache/data/JSON/
OUT_PUT_BASE=cache/exp/ColonEval

mkdir -p $OUT_PUT_BASE/$MODEL/pred

nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_1_ColonEval.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_1_ColonEval.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task1.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_2_ColonEval.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_2_ColonEval.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task2.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_3_ColonEval.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_3_ColonEval.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task3.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_4_ColonEval.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_4_ColonEval.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task4.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_5_ColonEval.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_5_ColonEval.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task5.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_6_ColonEval.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_6_ColonEval.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task6.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_7_ColonEval.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_7_ColonEval.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task7.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_8_ColonEval.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_8_ColonEval.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task8.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_9_ColonEval.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_9_ColonEval.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task9.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_10_ColonEval.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_10_ColonEval.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task10.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_11_ColonEval.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_11_ColonEval.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task11.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_12_ColonEval.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_12_ColonEval.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task12.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_14_ColonEval.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_14_ColonEval.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task14.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_15_ColonEval.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_15_ColonEval.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task15.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_16_ColonEval.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_16_ColonEval.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task16.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASE_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_17_ColonEval.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_17_ColonEval.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task17.txt 2>&1 &


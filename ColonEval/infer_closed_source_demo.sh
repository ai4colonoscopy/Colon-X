#!/bin/bash

MODEL=o4-mini-2025-04-16
IMAGE_BASH_PATH=cache/data
ROOT_PATH=cache/data/JSON/
OUT_PUT_BASE=cache/exp/ColonEval

mkdir -p $OUT_PUT_BASE/$MODEL/pred

nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_1_zero_shot_Test_5000.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_1_5000_zero_shot_Test.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task1.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_2_zero_shot_Test_5000.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_2_5000_zero_shot_Test.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task2.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_3_zero_shot_Test_5000.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_3_5000_zero_shot_Test.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task3.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_4_zero_shot_Test_5000.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_4_5000_zero_shot_Test.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task4.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_5_zero_shot_Test_5000.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_5_5000_zero_shot_Test.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task5.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_6_zero_shot_Test_5000.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_6_5000_zero_shot_Test.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task6.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_7_zero_shot_Test_5000.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_7_5000_zero_shot_Test.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task7.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_8_zero_shot_Test_5000.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_8_5000_zero_shot_Test.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task8.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_9_zero_shot_Test_5000.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_9_5000_zero_shot_Test.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task9.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_10_zero_shot_Test_5000.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_10_5000_zero_shot_Test.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task10.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_11_zero_shot_Test_5000.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_11_5000_zero_shot_Test.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task11.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_12_zero_shot_Test_5000.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_12_5000_zero_shot_Test.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task12.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_14_zero_shot_Test_5000.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_14_5000_zero_shot_Test.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task14.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_15_zero_shot_Test_5000.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_15_5000_zero_shot_Test.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task15.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_16_zero_shot_Test_5000.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_16_5000_zero_shot_Test.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task16.txt 2>&1 &


nohup python ColonEval/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_17_zero_shot_Test_5000.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_17_5000_zero_shot_Test.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task17.txt 2>&1 &


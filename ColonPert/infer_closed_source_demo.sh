#!/bin/bash

MODEL=o4-mini-2025-04-16
IMAGE_BASH_PATH=cache/data
ROOT_PATH=cache/data/JSON/
OUT_PUT_BASE=cache/exp/ColonEval

mkdir -p $OUT_PUT_BASE/$MODEL/pred

nohup python ColonPert/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonPert/TestA_on_image_text_masking.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_A_Test.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task_A.txt 2>&1 &


nohup python ColonPert/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonPert/TestB_on_image_misleading_text.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_B_Test.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task_B.txt 2>&1 &


nohup python ColonPert/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonPert/TestC_case_contradicting_instruction.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_C_Test.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task_C.txt 2>&1 &


nohup python ColonPert/infer_demo/infer_o4_mini_pilot.py \
   --model $MODEL \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonPert/TestD_emotion_driven_decision_bias.json \
   --output_path $OUT_PUT_BASE/$MODEL/pred/pred_Task_D_Test.json > $OUT_PUT_BASE/$MODEL/pred/nohup-pred_task_D.txt 2>&1 &


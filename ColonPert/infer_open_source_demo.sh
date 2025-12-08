#!/bin/bash


EXP_MODEL_ID=cache/exp/ColonPert/medgemma-4b-it
IMAGE_BASH_PATH=cache/data
ROOT_PATH=cache/data/JSON

mkdir -p $EXP_MODEL_ID/pred

################################# GPU 0 #################################
export CUDA_VISIBLE_DEVICES=0

nohup python ColonPert/infer_demo/infer_medgemma_pilot.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonPert/TestA_on_image_text_masking.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_A_Test.json > $EXP_MODEL_ID/pred/nohup-pred_task_A.txt 2>&1 &

################################# GPU 1 #################################
export CUDA_VISIBLE_DEVICES=1

nohup python ColonPert/infer_demo/infer_medgemma_pilot.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonPert/TestB_on_image_misleading_text.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_B_Test.json > $EXP_MODEL_ID/pred/nohup-pred_task_B.txt 2>&1 &

################################# GPU 2 #################################
export CUDA_VISIBLE_DEVICES=2

nohup python ColonPert/infer_demo/infer_medgemma_pilot.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonPert/TestC_case_contradicting_instruction.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_C_Test.json > $EXP_MODEL_ID/pred/nohup-pred_task_C.txt 2>&1 &

################################# GPU 3 #################################
export CUDA_VISIBLE_DEVICES=3

nohup python ColonPert/infer_demo/infer_medgemma_pilot.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonPert/TestD_emotion_driven_decision_bias.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_D_Test.json > $EXP_MODEL_ID/pred/nohup-pred_task_D.txt 2>&1 &

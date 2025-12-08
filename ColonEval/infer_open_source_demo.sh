#!/bin/bash


EXP_MODEL_ID=cache/exp/ColonEval/medgemma-4b-it
IMAGE_BASH_PATH=cache/data
ROOT_PATH=cache/data/JSON

mkdir -p $EXP_MODEL_ID/pred

################################# GPU 0 #################################
export CUDA_VISIBLE_DEVICES=0

nohup python ColonEval/infer_demo/infer_medgemma_pilot.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_1_zero_shot_Test_5000.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_1_5000_zero_shot_Test.json > $EXP_MODEL_ID/pred/nohup-pred_task1.txt 2>&1 &

################################# GPU 1 #################################
export CUDA_VISIBLE_DEVICES=1

nohup python ColonEval/infer_demo/infer_medgemma_pilot.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_2_zero_shot_Test_5000.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_2_5000_zero_shot_Test.json > $EXP_MODEL_ID/pred/nohup-pred_task2.txt 2>&1 &

################################# GPU 2 #################################
export CUDA_VISIBLE_DEVICES=2

nohup python ColonEval/infer_demo/infer_medgemma_pilot.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_3_zero_shot_Test_5000.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_3_5000_zero_shot_Test.json > $EXP_MODEL_ID/pred/nohup-pred_task3.txt 2>&1 &

################################# GPU 3 #################################
export CUDA_VISIBLE_DEVICES=3

nohup python ColonEval/infer_demo/infer_medgemma_pilot.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_4_zero_shot_Test_5000.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_4_5000_zero_shot_Test.json > $EXP_MODEL_ID/pred/nohup-pred_task4.txt 2>&1 &

################################# GPU 4 #################################
export CUDA_VISIBLE_DEVICES=4

nohup python ColonEval/infer_demo/infer_medgemma_pilot.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_5_zero_shot_Test_5000.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_5_5000_zero_shot_Test.json > $EXP_MODEL_ID/pred/nohup-pred_task5.txt 2>&1 &

################################# GPU 5 #################################
export CUDA_VISIBLE_DEVICES=5

nohup python ColonEval/infer_demo/infer_medgemma_pilot.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_6_zero_shot_Test_5000.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_6_5000_zero_shot_Test.json > $EXP_MODEL_ID/pred/nohup-pred_task6.txt 2>&1 &

################################# GPU6 #################################
export CUDA_VISIBLE_DEVICES=6

nohup python ColonEval/infer_demo/infer_medgemma_pilot.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_7_zero_shot_Test_5000.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_7_5000_zero_shot_Test.json > $EXP_MODEL_ID/pred/nohup-pred_task7.txt 2>&1 &

################################# GPU7 #################################
export CUDA_VISIBLE_DEVICES=7

nohup python ColonEval/infer_demo/infer_medgemma_pilot.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_8_zero_shot_Test_5000.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_8_5000_zero_shot_Test.json > $EXP_MODEL_ID/pred/nohup-pred_task8.txt 2>&1 &




################################# GPU 0 #################################
export CUDA_VISIBLE_DEVICES=0

nohup python ColonEval/infer_demo/infer_medgemma_pilot.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_9_zero_shot_Test_5000.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_9_5000_zero_shot_Test.json > $EXP_MODEL_ID/pred/nohup-pred_task9.txt 2>&1 &

################################# GPU 1 #################################
export CUDA_VISIBLE_DEVICES=1

nohup python ColonEval/infer_demo/infer_medgemma_pilot.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_10_zero_shot_Test_5000.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_10_5000_zero_shot_Test.json > $EXP_MODEL_ID/pred/nohup-pred_task10.txt 2>&1 &

################################# GPU 2 #################################
export CUDA_VISIBLE_DEVICES=2

nohup python ColonEval/infer_demo/infer_medgemma_pilot.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_11_zero_shot_Test_5000.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_11_5000_zero_shot_Test.json > $EXP_MODEL_ID/pred/nohup-pred_task11.txt 2>&1 &

################################# GPU 3 #################################
export CUDA_VISIBLE_DEVICES=3

nohup python ColonEval/infer_demo/infer_medgemma_pilot.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_12_zero_shot_Test_5000.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_12_5000_zero_shot_Test.json > $EXP_MODEL_ID/pred/nohup-pred_task12.txt 2>&1 &

################################# GPU 4 #################################
export CUDA_VISIBLE_DEVICES=4

nohup python ColonEval/infer_demo/infer_medgemma_pilot.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_14_zero_shot_Test_5000.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_14_5000_zero_shot_Test.json > $EXP_MODEL_ID/pred/nohup-pred_task14.txt 2>&1 &

################################# GPU 5 #################################
export CUDA_VISIBLE_DEVICES=5

nohup python ColonEval/infer_demo/infer_medgemma_pilot.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_15_zero_shot_Test_5000.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_15_5000_zero_shot_Test.json > $EXP_MODEL_ID/pred/nohup-pred_task15.txt 2>&1 &

################################# GPU 6 #################################
export CUDA_VISIBLE_DEVICES=6

nohup python ColonEval/infer_demo/infer_medgemma_pilot.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_16_zero_shot_Test_5000.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_16_5000_zero_shot_Test.json > $EXP_MODEL_ID/pred/nohup-pred_task16.txt 2>&1 &

################################# GPU 7 #################################
export CUDA_VISIBLE_DEVICES=7

nohup python ColonEval/infer_demo/infer_medgemma_pilot.py \
   --model_path $EXP_MODEL_ID \
   --image_dir $IMAGE_BASH_PATH \
   --json_file $ROOT_PATH/ColonEval/Task_17_zero_shot_Test_5000.json \
   --output_path $EXP_MODEL_ID/pred/pred_Task_17_5000_zero_shot_Test.json > $EXP_MODEL_ID/pred/nohup-pred_task17.txt 2>&1 &

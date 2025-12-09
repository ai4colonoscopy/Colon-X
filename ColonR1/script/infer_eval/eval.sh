#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

EXP_MODEL_ID=cache/checkpoints/ft-exp/ColonR1-Qwen2.5-VL-GRPO-thinking-StageII
EVAL_MODE=pilot

python ColonR1/serve/eval_engine.py \
	--task_id 1 \
	--data_type reasoning \
	--eval_mode $EVAL_MODE \
	--input_file $EXP_MODEL_ID/pred/pred_Task_1_ColonEval.json \
	--output_file $EXP_MODEL_ID/pred/Task_1.txt > $EXP_MODEL_ID/pred/eval_task_1_log.txt 2>&1

python ColonR1/serve/eval_engine.py \
	--task_id 2 \
	--data_type reasoning \
	--eval_mode $EVAL_MODE \
	--input_file $EXP_MODEL_ID/pred/pred_Task_2_ColonEval.json \
	--output_file $EXP_MODEL_ID/pred/Task_2.txt > $EXP_MODEL_ID/pred/eval_task_2_log.txt 2>&1

python ColonR1/serve/eval_engine.py \
	--task_id 3 \
	--data_type reasoning \
	--eval_mode $EVAL_MODE \
	--input_file $EXP_MODEL_ID/pred/pred_Task_3_ColonEval.json \
	--output_file $EXP_MODEL_ID/pred/Task_3.txt > $EXP_MODEL_ID/pred/eval_task_3_log.txt 2>&1

python ColonR1/serve/eval_engine.py \
	--task_id 4 \
	--data_type reasoning \
	--eval_mode $EVAL_MODE \
	--input_file $EXP_MODEL_ID/pred/pred_Task_4_ColonEval.json \
	--output_file $EXP_MODEL_ID/pred/Task_4.txt > $EXP_MODEL_ID/pred/eval_task_4_log.txt 2>&1

python ColonR1/serve/eval_engine.py \
	--task_id 5 \
	--data_type reasoning \
	--eval_mode $EVAL_MODE \
	--input_file $EXP_MODEL_ID/pred/pred_Task_5_ColonEval.json \
	--output_file $EXP_MODEL_ID/pred/Task_5.txt > $EXP_MODEL_ID/pred/eval_task_5_log.txt 2>&1

python ColonR1/serve/eval_engine.py \
	--task_id 6 \
	--data_type reasoning \
	--eval_mode $EVAL_MODE \
	--input_file $EXP_MODEL_ID/pred/pred_Task_6_ColonEval.json \
	--output_file $EXP_MODEL_ID/pred/Task_6.txt > $EXP_MODEL_ID/pred/eval_task_6_log.txt 2>&1

python ColonR1/serve/eval_engine.py \
	--task_id 7 \
	--data_type reasoning \
	--eval_mode $EVAL_MODE \
	--input_file $EXP_MODEL_ID/pred/pred_Task_7_ColonEval.json \
	--output_file $EXP_MODEL_ID/pred/Task_7.txt > $EXP_MODEL_ID/pred/eval_task_7_log.txt 2>&1

python ColonR1/serve/eval_engine.py \
	--task_id 8 \
	--data_type reasoning \
	--eval_mode $EVAL_MODE \
	--input_file $EXP_MODEL_ID/pred/pred_Task_8_ColonEval.json \
	--output_file $EXP_MODEL_ID/pred/Task_8.txt > $EXP_MODEL_ID/pred/eval_task_8_log.txt 2>&1

python ColonR1/serve/eval_engine.py \
	--task_id 9 \
	--data_type reasoning \
	--eval_mode $EVAL_MODE \
	--input_file $EXP_MODEL_ID/pred/pred_Task_9_ColonEval.json \
	--output_file $EXP_MODEL_ID/pred/Task_9.txt > $EXP_MODEL_ID/pred/eval_task_9_log.txt 2>&1

python ColonR1/serve/eval_engine.py \
	--task_id 10 \
	--data_type reasoning \
	--eval_mode $EVAL_MODE \
	--input_file $EXP_MODEL_ID/pred/pred_Task_10_ColonEval.json \
	--output_file $EXP_MODEL_ID/pred/Task_10.txt > $EXP_MODEL_ID/pred/eval_task_10_log.txt 2>&1

python ColonR1/serve/eval_engine.py \
	--task_id 11 \
	--data_type reasoning \
	--eval_mode $EVAL_MODE \
	--input_file $EXP_MODEL_ID/pred/pred_Task_11_ColonEval.json \
	--output_file $EXP_MODEL_ID/pred/Task_11.txt > $EXP_MODEL_ID/pred/eval_task_11_log.txt 2>&1

python ColonR1/serve/eval_engine.py \
	--task_id 12 \
	--data_type reasoning \
	--eval_mode $EVAL_MODE \
	--input_file $EXP_MODEL_ID/pred/pred_Task_12_ColonEval.json \
	--output_file $EXP_MODEL_ID/pred/Task_12.txt > $EXP_MODEL_ID/pred/eval_task_12_log.txt 2>&1

python ColonR1/serve/eval_engine.py \
	--task_id 14 \
	--data_type reasoning \
	--eval_mode $EVAL_MODE \
	--input_file $EXP_MODEL_ID/pred/pred_Task_14_ColonEval.json \
	--output_file $EXP_MODEL_ID/pred/Task_14.txt > $EXP_MODEL_ID/pred/eval_task_14_log.txt 2>&1

python ColonR1/serve/eval_engine.py \
	--task_id 15 \
	--data_type reasoning \
	--eval_mode $EVAL_MODE \
	--input_file $EXP_MODEL_ID/pred/pred_Task_15_ColonEval.json \
	--output_file $EXP_MODEL_ID/pred/Task_15.txt > $EXP_MODEL_ID/pred/eval_task_15_log.txt 2>&1

python ColonR1/serve/eval_engine.py \
	--task_id 16 \
	--data_type reasoning \
	--eval_mode $EVAL_MODE \
	--input_file $EXP_MODEL_ID/pred/pred_Task_16_ColonEval.json \
	--output_file $EXP_MODEL_ID/pred/Task_16.txt > $EXP_MODEL_ID/pred/eval_task_16_log.txt 2>&1

python ColonR1/serve/eval_engine.py \
	--task_id 17 \
	--data_type reasoning \
	--eval_mode $EVAL_MODE \
	--input_file $EXP_MODEL_ID/pred/pred_Task_17_ColonEval.json \
	--output_file $EXP_MODEL_ID/pred/Task_17.txt > $EXP_MODEL_ID/pred/eval_task_17_log.txt 2>&1

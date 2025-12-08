#!/bin/bash

EXP_MODEL_ID=cache/exp/robust-exp/medgemma-4b-it
EVAL_MODE=pert


python ColonEval/eval_engine.py \
	--task_id A \
	--eval_mode $EVAL_MODE \
	--input_file $EXP_MODEL_ID/pred/TestA_on_image_text_masking.json \
	--output_file $EXP_MODEL_ID/pred/Task_A.txt > $EXP_MODEL_ID/pred/eval_task_A_log.txt 2>&1 &

python ColonEval/eval_engine.py \
	--task_id B \
	--eval_mode $EVAL_MODE \
	--input_file $EXP_MODEL_ID/pred/TestB_on_image_misleading_text.json \
	--output_file $EXP_MODEL_ID/pred/Task_B.txt > $EXP_MODEL_ID/pred/eval_task_B_log.txt 2>&1 &

python ColonEval/eval_engine.py \
	--task_id C \
	--eval_mode $EVAL_MODE \
	--input_file $EXP_MODEL_ID/pred/TestC_case_contradicting_instruction.json \
	--output_file $EXP_MODEL_ID/pred/Task_C.txt > $EXP_MODEL_ID/pred/eval_task_C_log.txt 2>&1 &

python ColonEval/eval_engine.py \
	--task_id D \
	--eval_mode $EVAL_MODE \
	--input_file $EXP_MODEL_ID/pred/TestD_emotion_driven_decision_bias.json \
	--output_file $EXP_MODEL_ID/pred/Task_D.txt > $EXP_MODEL_ID/pred/eval_task_D_log.txt 2>&1 &

#!/bin/bash

MODEL_PATH=ai4colonoscopy/ColonR1
IMAGE_PATH=assets/example.jpg

python ColonR1/serve/inference_single.py \
--model_path $MODEL_PATH \
--image_path $IMAGE_PATH

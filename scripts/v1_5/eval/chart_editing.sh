#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

#CHUNKS=${#GPULIST[@]}
CHUNKS=1
# /mnt/private_yucheng/chartgpt/LLaVA/checkpoints/final_merged_v1_plus_editing_3epochs
output_name=final_merged_v1_plus_editing_3epochs
IDX=0

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_lora \
    --model-path /mnt/private_yucheng/chartgpt/LLaVA/checkpoints/${output_name} \
    --question-file /mnt/private_yucheng/chartgpt/LLaVA/playground/data/ours/chart_editing_final_test.json \
    --image-folder ./playground/data/ \
    --answers-file ./playground/data/eval_res/ours/${output_name}_chart_editing.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --conv-mode vicuna_v1 

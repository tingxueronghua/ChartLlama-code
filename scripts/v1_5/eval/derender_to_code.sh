#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

#CHUNKS=${#GPULIST[@]}
CHUNKS=1
# /mnt/private_yucheng/chartgpt/LLaVA/checkpoints/ours_derender_to_code_plus_derender_to_csv_40epochs
output_name=ours_derender_to_code_plus_derender_to_csv_40epochs
IDX=0


CUDA_VISIBLE_DEVICES=1 python -m llava.eval.model_vqa_lora \
    --model-path /mnt/private_yucheng/chartgpt/LLaVA/checkpoints/${output_name} \
    --question-file /mnt/private_yucheng/chartgpt/LLaVA/playground/data/eval_json/ours_derender_to_code_eval.json \
    --image-folder ./playground/data/ \
    --answers-file ./playground/data/eval_res/derender_to_code/${output_name}_ours_ans.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --conv-mode vicuna_v1 &

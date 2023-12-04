#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

#CHUNKS=${#GPULIST[@]}
CHUNKS=1
# /mnt/private_yucheng/chartgpt/LLaVA/checkpoints/ours_derender_plus_chartqa_table_sharp_v3_3epochs
output_name=ours_derender_plus_chartqa_table_sharp_v3_3epochs
IDX=0

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_lora \
    --model-path /mnt/private_yucheng/chartgpt/LLaVA/checkpoints/${output_name} \
    --question-file /mnt/private_yucheng/chartgpt/chartqa/chartqa_test_human_table.json \
    --image-folder ./playground/data/ \
    --answers-file ./playground/data/eval_res/chartqa_derender/${output_name}_chartqa_human_ans.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --conv-mode vicuna_v1 &
sleep 60
CUDA_VISIBLE_DEVICES=1 python -m llava.eval.model_vqa_lora \
    --model-path /mnt/private_yucheng/chartgpt/LLaVA/checkpoints/${output_name} \
    --question-file /mnt/private_yucheng/chartgpt/chartqa/chartqa_test_augmented_table.json \
    --image-folder ./playground/data/ \
    --answers-file ./playground/data/eval_res/chartqa_derender/${output_name}_chartqa_augmented_ans.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --conv-mode vicuna_v1 &





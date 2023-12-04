#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

#CHUNKS=${#GPULIST[@]}
CHUNKS=1
# /mnt/private_yucheng/chartgpt/LLaVA/checkpoints/ours_qa_and_chartqa_sharp_v8_3epochs
output_name=ours_qa_and_chartqa_sharp_v10_3epochs
IDX=0

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_lora \
    --model-path /mnt/private_yucheng/chartgpt/LLaVA/checkpoints/${output_name} \
    --question-file /mnt/private_yucheng/chartgpt/LLaVA/playground/data/eval_json/chartqa/chartqa_test_human.json \
    --image-folder ./playground/data/ \
    --answers-file ./playground/data/eval_res/chartqa/chartqa_plus_ours_linechart/${output_name}_chartqa_human_ans.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --conv-mode vicuna_v1 &
sleep 60
CUDA_VISIBLE_DEVICES=1 python -m llava.eval.model_vqa_lora \
    --model-path /mnt/private_yucheng/chartgpt/LLaVA/checkpoints/${output_name} \
    --question-file /mnt/private_yucheng/chartgpt/LLaVA/playground/data/eval_json/chartqa/chartqa_test_augmented.json \
    --image-folder ./playground/data/ \
    --answers-file ./playground/data/eval_res/chartqa/chartqa_plus_ours_linechart/${output_name}_chartqa_augmented_ans.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --conv-mode vicuna_v1 &
#sleep 60
#CUDA_VISIBLE_DEVICES=2 python -m llava.eval.model_vqa_lora \
    #--model-path /mnt/private_yucheng/chartgpt/LLaVA/checkpoints/${output_name} \
    #--question-file /mnt/private_yucheng/chartgpt/LLaVA/playground/data/ours/linechart/ours_simplified_qa_test.json \
    #--image-folder ./playground/data/ \
    #--answers-file ./playground/data/eval_res/chartqa/chartqa_plus_ours_linechart/${output_name}_ours_linechart_simplified_qa_ans.jsonl \
    #--num-chunks $CHUNKS \
    #--chunk-idx $IDX \
    #--temperature 0 \
    #--conv-mode vicuna_v1 &





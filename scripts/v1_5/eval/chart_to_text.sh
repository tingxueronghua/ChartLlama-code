#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

#CHUNKS=${#GPULIST[@]}
CHUNKS=1
# /mnt/private_yucheng/chartgpt/LLaVA/checkpoints/statista_sharp_v5_5epochs
output_name=statista_sharp_v5_5epochs
IDX=0

#CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_lora \
    #--model-path /mnt/private_yucheng/chartgpt/LLaVA/checkpoints/${output_name} \
    #--question-file /mnt/private_yucheng/chartgpt/chart-to-text/pew_dataset_test.json \
    #--image-folder ./playground/data/ \
    #--answers-file ./playground/data/eval_res/chart-to-text/${output_name}_pew_dataset_ans.jsonl \
    #--num-chunks $CHUNKS \
    #--chunk-idx $IDX \
    #--temperature 0 \
    #--conv-mode vicuna_v1 
#sleep 60
CUDA_VISIBLE_DEVICES=1 python -m llava.eval.model_vqa_lora \
    --model-path /mnt/private_yucheng/chartgpt/LLaVA/checkpoints/${output_name} \
    --question-file /mnt/private_yucheng/chartgpt/chart-to-text/statista_dataset_test.json \
    --image-folder ./playground/data/ \
    --answers-file ./playground/data/eval_res/chart-to-text/${output_name}_statista_dataset_ans.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --conv-mode vicuna_v1 
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





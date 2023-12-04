#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

#CHUNKS=${#GPULIST[@]}
CHUNKS=1
# /mnt/private_yucheng/chartgpt/LLaVA/checkpoints/final_merged_v6_3epochs
output_name=vanilla
IDX=0

#CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_vanilla \
    #--model-path /mnt/private_yucheng/huggingface_hub/llava-v1.5-13b \
    #--question-file /mnt/private_yucheng/chartgpt/chartqa/chartqa_test_human.json \
    #--image-folder ./playground/data/ \
    #--answers-file ./playground/data/eval_res/chartqa/chartqa_plus_ours_linechart/${output_name}_chartqa_human_ans.jsonl \
    #--num-chunks $CHUNKS \
    #--chunk-idx $IDX \
    #--temperature 0 \
    #--conv-mode vicuna_v1 &
#sleep 60
#CUDA_VISIBLE_DEVICES=1 python -m llava.eval.model_vqa_vanilla \
    #--model-path /mnt/private_yucheng/huggingface_hub/llava-v1.5-13b \
    #--question-file /mnt/private_yucheng/chartgpt/chartqa/chartqa_test_augmented.json \
    #--image-folder ./playground/data/ \
    #--answers-file ./playground/data/eval_res/chartqa/chartqa_plus_ours_linechart/${output_name}_chartqa_augmented_ans.jsonl \
    #--num-chunks $CHUNKS \
    #--chunk-idx $IDX \
    #--temperature 0 \
    #--conv-mode vicuna_v1 &
#sleep 60
#CUDA_VISIBLE_DEVICES=2 python -m llava.eval.model_vqa_vanilla \
    #--model-path /mnt/private_yucheng/huggingface_hub/llava-v1.5-13b \
    #--question-file /mnt/private_yucheng/chartgpt/chart-to-text/pew_dataset_test.json \
    #--image-folder ./playground/data/ \
    #--answers-file ./playground/data/eval_res/chart-to-text/${output_name}_pew_dataset_ans.jsonl \
    #--num-chunks $CHUNKS \
    #--chunk-idx $IDX \
    #--temperature 0 \
    #--conv-mode vicuna_v1 &
#sleep 60
##CUDA_VISIBLE_DEVICES=3 python -m llava.eval.model_vqa_vanilla \
    ##--model-path /mnt/private_yucheng/chartgpt/LLaVA/checkpoints/${output_name} \
    ##--question-file /mnt/private_yucheng/chartgpt/chart-to-text/statista_dataset_test.json \
    ##--image-folder ./playground/data/ \
    ##--answers-file ./playground/data/eval_res/chart-to-text/${output_name}_statista_dataset_ans.jsonl \
    ##--num-chunks $CHUNKS \
    ##--chunk-idx $IDX \
    ##--temperature 0 \
    ##--conv-mode vicuna_v1 &
##sleep 60
#CUDA_VISIBLE_DEVICES=4 python -m llava.eval.model_vqa_vanilla \
    #--model-path /mnt/private_yucheng/huggingface_hub/llava-v1.5-13b \
    #--question-file /mnt/private_yucheng/chartgpt/chartqa/chartqa_test_human_table.json \
    #--image-folder ./playground/data/ \
    #--answers-file ./playground/data/eval_res/chartqa_derender/${output_name}_chartqa_human_ans.jsonl \
    #--num-chunks $CHUNKS \
    #--chunk-idx $IDX \
    #--temperature 0 \
    #--conv-mode vicuna_v1 &
#sleep 60
#CUDA_VISIBLE_DEVICES=5 python -m llava.eval.model_vqa_vanilla \
    #--model-path /mnt/private_yucheng/huggingface_hub/llava-v1.5-13b \
    #--question-file /mnt/private_yucheng/chartgpt/chartqa/chartqa_test_augmented_table.json \
    #--image-folder ./playground/data/ \
    #--answers-file ./playground/data/eval_res/chartqa_derender/${output_name}_chartqa_augmented_ans.jsonl \
    #--num-chunks $CHUNKS \
    #--chunk-idx $IDX \
    #--temperature 0 \
    #--conv-mode vicuna_v1 &
#sleep 60
#CUDA_VISIBLE_DEVICES=6 python -m llava.eval.model_vqa_vanilla \
    #--model-path /mnt/private_yucheng/huggingface_hub/llava-v1.5-13b \
    #--question-file /mnt/private_yucheng/chartgpt/LLaVA/playground/data/eval_json/ours_detailed_description_test \
    #--image-folder ./playground/data/ \
    #--answers-file ./playground/data/eval_res/ours/${output_name}_detailed_description_ans.jsonl \
    #--num-chunks $CHUNKS \
    #--chunk-idx $IDX \
    #--temperature 0 \
    #--conv-mode vicuna_v1 &
#sleep 60
CUDA_VISIBLE_DEVICES=7 python -m llava.eval.model_vqa_vanilla \
    --model-path /mnt/private_yucheng/huggingface_hub/llava-v1.5-13b \
    --question-file /mnt/private_yucheng/chartgpt/LLaVA/playground/data/eval_json/ours_text_to_chart_test \
    --image-folder ./playground/data/ \
    --answers-file ./playground/data/eval_res/ours/${output_name}_text_to_chart_ans.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --conv-mode vicuna_v1 &
#sleep 60
#CUDA_VISIBLE_DEVICES=3 python -m llava.eval.model_vqa_vanilla \
    #--model-path /mnt/private_yucheng/huggingface_hub/llava-v1.5-13b \
    #--question-file /mnt/private_yucheng/chartgpt/LLaVA/playground/data/eval_json/ours_derender_to_code_test \
    #--image-folder ./playground/data/ \
    #--answers-file ./playground/data/eval_res/ours/${output_name}_derender_to_code_ans.jsonl \
    #--num-chunks $CHUNKS \
    #--chunk-idx $IDX \
    #--temperature 0 \
    #--conv-mode vicuna_v1 &

#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

#CHUNKS=${#GPULIST[@]}
CHUNKS=1
# /mnt/private_yucheng/chartgpt/LLaVA/checkpoints/final_merged_v5_1epochs
output_name=final_merged_v5_1epochs

chart_type_list=(box_chart candlestick_chart funnel_chart gantt_chart heatmap_chart polar_chart scatter_chart)
IDX=0


for chart_type in ${chart_type_list[@]}
do
CUDA_VISIBLE_DEVICES=${IDX} python -m llava.eval.model_vqa_lora \
    --model-path /mnt/private_yucheng/chartgpt/LLaVA/checkpoints/${output_name} \
    --question-file /mnt/private_yucheng/chartgpt/LLaVA/playground/data/ours/${chart_type}_100examples_qa_test.json \
    --image-folder ./playground/data/ \
    --answers-file ./playground/data/eval_res/chartqa/chartqa_plus_ours_linechart/${output_name}_${chart_type}_chartqa_human_ans.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --conv-mode vicuna_v1 & 
let IDX++
sleep 30
done



#for chart_type in ${chart_type_list[@]}
#do
#CUDA_VISIBLE_DEVICES=1 python -m llava.eval.model_vqa_vanilla \
    #--model-path /mnt/private_yucheng/huggingface_hub/llava-v1.5-13b \
    #--question-file /mnt/private_yucheng/chartgpt/LLaVA/playground/data/ours/${chart_type}_100examples_qa_test.json \
    #--image-folder ./playground/data/ \
    #--answers-file ./playground/data/eval_res/chartqa/chartqa_plus_ours_linechart/vanilla_${chart_type}_chartqa_human_ans.jsonl \
    #--num-chunks $CHUNKS \
    #--chunk-idx $IDX \
    #--temperature 0 \
    #--conv-mode vicuna_v1 
#sleep 60
#done

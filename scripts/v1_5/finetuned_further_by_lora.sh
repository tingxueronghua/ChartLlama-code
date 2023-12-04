#!/bin/bash


WANDB_MODE=offline deepspeed llava/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --lora_enable True \
    --model_name_or_path /mnt/private_yucheng/huggingface_hub/llava-v1.5-13b \
    --version v1 \
    --data_path /mnt/private_yucheng/chartgpt/LLaVA/playground/data/ours/publish_v1.json \
    --image_folder /mnt/private_yucheng/chartgpt/LLaVA/playground/data \
    --vision_tower /mnt/share_1227775/yandali/multimodal/models/ft_local/clip-vit-large-patch14-336/ \
    --lora_further_tune_finetuned \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --fp16 True \
    --lora_r 64 \
    --output_dir ./checkpoints/publish_v1_3epochs \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 

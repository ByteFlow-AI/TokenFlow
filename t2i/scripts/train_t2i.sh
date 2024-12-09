#!/bin/bash


export MODEL_LOG_PATH=./checkpoints/inital-try
export DATA_PATH="your/data/path"
export VISION_TOWER_CKPT="../pretrained_ckpts/tokenflow_clipb_32k_enhanced.pt"
# export MODEL_PATH="TinyLlama/TinyLlama_v1.1"
export MODEL_PATH="meta-llama/Llama-2-7b-hf"

deepspeed  \
  llava_t2i/train/train_plain.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $MODEL_PATH \
    --version plain_img \
    --data_path $DATA_PATH \
    --vision_tower $VISION_TOWER_CKPT \
    --mm_vision_vq_type TOKEN_FLOW \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_vq_token True \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --ignore_text_loss True \
    --bf16 True \
    --output_dir $MODEL_LOG_PATH \
    --max_grad_norm 0.5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --eval_steps 2000 \
    --save_total_limit 50 \
    --num_train_epochs 10 \
    --learning_rate 5e-4 \
    --weight_decay 0.1 \
    --warmup_steps 5000 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --max_text_token_num 128 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb

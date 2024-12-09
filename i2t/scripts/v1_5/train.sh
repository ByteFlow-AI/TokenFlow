#!/bin/bash
# pip3 install -e . 
# pip3 install -e ".[train]" 

CUR_DIR=$(cd `dirname $0`; pwd)

cd ${CUR_DIR}/../..

export FAST_TRANSFORMER=O0
export BYTED_TORCH_BYTECCL=O0
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0

export OMP_NUM_THREADS=8

export PRETRAIN_OUT_PATH=checkpoints/i2t_pre
export SFT_OUT_PATH=checkpoints/i2t_sft
export VISION_TOWER_CKPT="PATH_TO_VISION_TOWER"

PRETRAIN_TASK_NAME=$(basename "${PRETRAIN_OUT_PATH%/}")
SFT_TASK_NAME=$(basename "${SFT_OUT_PATH%/}")

### pretrain ####

torchrun \
--nnodes $WORKER_NUM \
--node_rank $NODE_RANK \
--nproc_per_node $NPROC_PER_NODE \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --version plain \
    --data_path /YOUR_DATA_PATH/Cambrian-Alignment/jsons/alignment_2.5m.jsonl \
    --image_folder /YOUR_DATA_PATH/cambrian/Cambrian-Alignment \
    --vision_tower $VISION_TOWER_CKPT \
    --mm_vision_vq_type LLAMAGEN_VQKD \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_tuning_embedding False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $PRETRAIN_OUT_PATH \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${PRETRAIN_TASK_NAME}

sleep 10m

### SFT finetune ####

torchrun \
--nnodes $WORKER_NUM \
--node_rank $NODE_RANK \
--nproc_per_node $NPROC_PER_NODE \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --version qwen_2_5 \
    --data_path /YOUR_DATA_PATH/Cambrian-10M/jsons/Cambrian10M.jsonl \
    --image_folder /YOUR_DATA_PATH/Cambrian-10M/ \
    --vision_tower $VISION_TOWER_CKPT \
    --mm_vision_vq_type LLAMAGEN_VQKD \
    --pretrain_mm_mlp_adapter $PRETRAIN_OUT_PATH/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $SFT_OUT_PATH \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${SFT_TASK_NAME}


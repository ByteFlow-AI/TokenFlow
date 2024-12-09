# !/bin/bash
set -x
pip3 install -r requirements.txt

export WANDB_API_KEY="your wandb key"

torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... \
tokenizer/vq_train.py "$@" \
--dataset "imagenet" \
--teacher "clipb_224" \
--finetune_decoder \
--enhanced_decoder \
--image-size 224 \
--codebook-size 32768 \
--epochs 100 \
--global-batch-size 32 \
--cloud-save-path "./logs/your-log-path/" \
--vq-ckpt="/path/to/your/ckpt"
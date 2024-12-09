# !/bin/bash
set -x
pip3 install -r requirements.txt

# export WANDB_API_KEY="your wandb key"

export OMP_NUM_THREAD
torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... \
tokenizer/vq_train.py "$@" \
--dataset "imagenet" \
--data-path "/path/to/your/dataset" \
--teacher "clipb_224" \
--image-size 224 \
--codebook-size 32768 \
--epochs 100 \
--global-batch-size 32 \
--ckpt-every 100 \
--cloud-save-path "./logs/clipb-training/"



# torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... \
# tokenizer/vq_train.py "$@" \
# --dataset "imagenet" \
# --data-path "/path/to/your/dataset" \
# --teacher "vitamin_xlarge_256" \
# --image-size 256 \
# --codebook-size 32768 \
# --epochs 100 \
# --global-batch-size 32 \
# --cloud-save-path "./logs/vitamin-256-training/"


# torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... \
# tokenizer/vq_train.py "$@" \
# --dataset "imagenet" \
# --data-path "/path/to/your/dataset" \
# --teacher "siglip_384" \
# --image-size 384 \
# --codebook-size 32768 \
# --epochs 100 \
# --global-batch-size 32 \
# --cloud-save-path "./logs/siglip-training/"
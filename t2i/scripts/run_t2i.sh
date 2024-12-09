# !/bin/bash

python3 llava_t2i/eval/run_llava_samples.py \
--model-path "ByteFlow-AI/TokenFlow-t2i" \
--tokenizer-path "../pretrained_ckpts/tokenflow_clipb_32k_enhanced.pt" \
--output-path "generations/" \
--cfg 7.5 \
--loop 1 \
--mixed_precision bf16 \
--batch_size 20 \
# --g_seed 0





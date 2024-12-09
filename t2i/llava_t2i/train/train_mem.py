import os
from llava_t2i.train.train import train


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")

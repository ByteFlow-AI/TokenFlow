from torch import nn
import torch
import torch.nn.functional as F
import math
from functools import partial
import torch.distributed as distributed
from einops import rearrange

import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_dir,'../../../../tokenflow/'))
from tokenizer.vq_model import VQ_models

import types
# define a nothing train method
def no_op_train(self, mode=True):
    pass

# process_type
def tokenflow_model(model_name, pretrain_path=None):
    vq_model = VQ_models[model_name](codebook_size=32768, 
                                     codebook_embed_dim=8, 
                                     teacher='clipb_224', 
                                     enhanced_decoder=True, 
                                     infer_interpolate=True)
    vq_model.train = types.MethodType(no_op_train, vq_model)
    if pretrain_path:
        checkpoint = torch.load(pretrain_path, map_location="cpu")
        if "ema" in checkpoint:  # ema
            model_weight = checkpoint["ema"]
        elif "model" in checkpoint:  # ddp
            model_weight = checkpoint["model"]
        elif "state_dict" in checkpoint:
            model_weight = checkpoint["state_dict"]
        else:
            raise Exception("please check model weight")
        vq_model.load_state_dict(model_weight)
        vq_model.quantize.training = False

    vq_model.eval()
    vq_model.training = False
    # exit()

    return vq_model







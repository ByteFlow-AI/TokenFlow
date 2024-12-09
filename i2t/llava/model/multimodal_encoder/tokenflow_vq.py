from torch import nn
import torch
import torch.nn.functional as F
import math
from functools import partial
import torch.distributed as distributed
from einops import rearrange
from transformers.modeling_utils import get_parameter_dtype
import sys
from pathlib import Path
cur_file_path = Path(__file__).resolve()
sys.path.append(str(cur_file_path.parent.parent.parent.parent.parent))
from tokenflow.tokenizer.vq_model import VQ_models
import types

def no_op_train(self, mode=True):
    pass

# process_type
def tokenflow_model(model_name, codebook_size, teacher, pretrain_path=None):
    vq_model = VQ_models[model_name](codebook_size=codebook_size, teacher=teacher)
    vq_model.train = types.MethodType(no_op_train, vq_model)

    if pretrain_path is not None:
        print("tokenflow load from:", pretrain_path)
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
        print("tokenflow model load success!!")
        vq_model.eval()
        vq_model.training = False
        vq_model.quantize.training = False

    return vq_model
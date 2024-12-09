# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers
#   maskgit: https://github.com/google-research/maskgit
#   LlamaGen: https://github.com/FoundationVision/LlamaGen/
#   VAR: https://github.com/FoundationVision/VAR

import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir)
from dataclasses import dataclass, field
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .vqkd_model import VisionTransformer
import math
from einops import rearrange
from .clip import clip
import numpy as np
from transformers.modeling_utils import get_parameter_device, get_parameter_dtype
from .norm_ema_quantizer import EmbeddingEMA, l2norm, norm_ema_inplace, kmeans
import torch.distributed as dist
import random

from timm.models.layers import trunc_normal_
from timm import create_model

from transformers import SiglipImageProcessor, SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipVisionModel


def copy_new_embedding(old_embedding, requires_grad=True):
    new_embedding = nn.Embedding(old_embedding.weight.size(0), old_embedding.weight.size(1))
    new_embedding.weight = nn.Parameter(old_embedding.weight.clone())
    new_embedding.weight.requires_grad = requires_grad
    return new_embedding

def drop_scale(original_scales, num_to_drop=1):
    """
    Randomly remove scales from scale list.
    
    Args:
        original_scales: list of scales
        num_to_drop: Number of scales to randomly remove (default 1)
        
    Returns:
        New scale list
    """
    if num_to_drop >= len(original_scales) - 1:
        raise ValueError("Cannot drop that many items")
    
    drop_candidates = list(range(1, len(original_scales)))
    indices_to_drop = set(random.sample(drop_candidates, num_to_drop))
    return [item for i, item in enumerate(original_scales) if i not in indices_to_drop]


@dataclass
class ModelArgs:
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    semantic_code_dim: int = 32
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 1.0
    entropy_loss_ratio: float = 0.0
    
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0
    kmeans: bool = False
    teacher: str = None # option: ["clipb_224", "vitamin_xlarge_256", "siglip_384"]
    enhanced_decoder: bool = False
    infer_interpolate: bool = False

def get_model_default_params():
    return dict(img_size=256, patch_size=16, in_chans=3, num_classes=1000, embed_dim=1152, depth=12, num_heads=12,  
                             mlp_ratio=4., qkv_bias=True,  qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                             norm_layer='LayerNorm', init_values=0., use_abs_pos_emb=True, 
                             use_rel_pos_bias=False, use_shared_rel_pos_bias=False, use_mean_pooling=True, init_scale=0.001)


class TokenFlow(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        ### VQGAN encoder decoder definitions ###
        self.encoder = Encoder(ch_mult=config.encoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
        if not config.enhanced_decoder:
            print("Using normal pixel decoder")
            self.decoder = Decoder(ch_mult=config.decoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
        else:
            print("Using 2 times enhanced pixel decoder")
            self.decoder = Decoder(
                ch_mult=config.decoder_ch_mult, 
                z_channels=config.z_channels, 
                dropout=config.dropout_p,
                ch=256,
                num_res_blocks=4,
        )
        ### VQGAN encoder decoder definitions ###

        self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)

        ### VQKD semantic teacher choice ###
        ### option: ["clipb_224", "vitamin_xlarge_256", "siglip_384"]
        self.teacher = config.teacher

        ### multi-scale quantizer ###
        self.enable_var = True
        if self.enable_var:
            if self.teacher == 'clipb_224':
                if not config.infer_interpolate:
                    self.scale_rq_layers = [1,2,4,6,8,10,12,14]
                else: # directly interpolate the positional encoding of clip for higher resolution inference
                    print("using interpolated scales")
                    self.scale_rq_layers = [1,2,4,6,8,10,12,14,16]
            elif self.teacher == 'vitamin_xlarge_256':
                self.scale_rq_layers = [1,2,4,6,8,10,12,14,16]
            elif self.teacher == 'siglip_384':
                self.scale_rq_layers = [1,2,3,4,5,6,7,8,9,10,12,14,17,22,27]
            else:
                raise NotImplementedError
            self.rq_depth = len(self.scale_rq_layers)
        ### multi-scale quantizer ###

        encoder_config, decoder_config = get_model_default_params(), get_model_default_params()
        if self.teacher == 'clipb_224':
            encoder_config["embed_dim"] = decoder_config["embed_dim"] = 768
            img_size = 224
            self.decoder_out_dim = 512
        elif self.teacher == 'vitamin_xlarge_256':
            img_size = 256
            self.decoder_out_dim = 1152
        elif self.teacher == 'siglip_384':
            img_size = 27 * 2**4 # 384
            self.decoder_out_dim = 1152
        else:
            raise NotImplementedError
        
        semantic_code_dim = config.semantic_code_dim # 32 by default
            
        encoder_config['img_size'] = img_size
        encoder_config['num_classes'] = 0

        # decoder settings
        decoder_config['img_size'] = img_size // decoder_config['patch_size']
        decoder_config['patch_size'] = 1
        decoder_config['in_chans'] = semantic_code_dim
        decoder_config['num_classes'] = 0
        decoder_config['depth'] = 3

        print('Final encoder config', encoder_config)

        ### VQKD encoder decoder definition ###
        ### define semantic encoder, initialize it with pretrained clip ###
        if self.teacher == 'clipb_224':
            self.encoder_vqkd = clip.load_clip_vision("ViT-B/16", download_root='./clip_model')
        elif self.teacher == 'vitamin_xlarge_256':
            print('Current teacher is vitamin_xlarge_256')
            self.encoder_vqkd = create_model(self.teacher, pretrained=True)
            del self.encoder_vqkd.head
            del self.encoder_vqkd.fc_norm
        elif self.teacher == 'siglip_384':
            print('Current teacher is google/siglip-so400m-patch14-384')
            self.encoder_vqkd = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
            # Freeze the head's parameters and 26th layer's parameters
            for name, param in self.encoder_vqkd.named_parameters():
                if 'head' in name:
                    param.requires_grad = False
                if 'encoder.layers.26' in name:
                    param.requires_grad = False
                if 'post_layernorm' in name:
                    param.requires_grad = False
        else:
            raise ValueError(f'teacher {self.teacher} not supported')
        
        print('Final decoder config', decoder_config)
        self.decoder_vqkd = VisionTransformer(**decoder_config)
        ### VQKD encoder decoder definition ###


        ### task layer ###
        self.encode_task_layer = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(encoder_config['embed_dim'], semantic_code_dim) # for quantize
        )
        self.decode_task_layer = nn.Sequential(
            nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
        )
        self.encode_task_layer.apply(self._init_weights)
        self.decode_task_layer.apply(self._init_weights)
        ### task layer ###


        # scaling layers
        if self.teacher == 'clipb_224':
            self.scaling_layer = None # can be changed to ScalingLayerForClip()
        elif self.teacher == 'vitamin_xlarge_256':
            self.scaling_layer = ScalingLayerForClip()
        elif self.teacher == 'siglip_384':
            self.scaling_layer = ScalingLayerForSigLip()
        else:
            raise ValueError(f'teacher {self.teacher} not supported')        

        ### parameters required by llava ###
        self.semantic_code_dim = semantic_code_dim
        self.vqgan_code_dim = config.codebook_embed_dim
        self.code_dim = self.semantic_code_dim + self.vqgan_code_dim

        self.embed_dim = self.code_dim
        self.n_embed = config.codebook_size
        self.compression = 2**(len(config.encoder_ch_mult) - 1)
        ### parameters required by llava ###
        
        # quantizer
        self.use_kmeans = config.kmeans
        self.quantize = VectorQuantizer(config.codebook_size, self.code_dim, 
                                config.commit_loss_beta, config.entropy_loss_ratio,
                                config.codebook_l2_norm, config.codebook_show_usage, split=[self.semantic_code_dim, self.vqgan_code_dim], kmeans=config.kmeans)


        ### whether using shared conv in VAR ###
        self.using_shared_conv = False # default not use.
        if self.using_shared_conv:
            quant_resi = 0.5
            share_quant_resi = 4
            self.vqkd_quant_resi = PhiPartiallyShared(nn.ModuleList([(Phi(self.vqkd_code_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(share_quant_resi)]))
            self.vqgan_quant_resi = PhiPartiallyShared(nn.ModuleList([(Phi(self.vqgan_code_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(share_quant_resi)]))
            print("quant conv initialized")

        ### whether using random scale drop when training ###
        self.random_scale_drop = True
        self.random_scale_drop_ratio = 0.1 # default 10%
        self.num_of_random_drop = 1

        print(f'Current model is: TokenFlow with teacher {self.teacher}, initialization finished.')
    
    def clone_vq_codebook(self, requires_grad):
        cloned_vqkd_embedding = copy_new_embedding(self.quantize.embedding_vqkd, requires_grad)
        cloned_vqgan_embedding = copy_new_embedding(self.quantize.embedding_vqgan, requires_grad)
        return (cloned_vqkd_embedding, cloned_vqgan_embedding)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def device(self):
        return get_parameter_device(self)

    @property
    def dtype(self):
        return get_parameter_dtype(self)

    def encode(self, x):

        if self.teacher == "clipb_224":
            vqkd_feature = self.encoder_vqkd(x, return_patch_tokens=True)
            vqkd_feature = self.encode_task_layer(vqkd_feature.type_as(self.encode_task_layer[-1].weight))
            N = vqkd_feature.shape[1]
            B = vqkd_feature.shape[0]
            h, w = int(math.sqrt(N)), int(math.sqrt(N))
            vqkd_feature = rearrange(vqkd_feature, 'b (h w) c -> b c h w', h=h, w=w) # reshape for quantizer
        elif self.teacher == 'vitamin_xlarge_256':
            vqkd_x = self.scaling_layer(x)
            # vqkd
            vqkd_feature = self.encoder_vqkd.forward_features(vqkd_x)
            vqkd_feature = self.encode_task_layer(vqkd_feature.type_as(self.encode_task_layer[-1].weight))
            N = vqkd_feature.shape[1]
            h, w = int(math.sqrt(N)), int(math.sqrt(N))
            vqkd_feature = rearrange(vqkd_feature, 'b (h w) c -> b c h w', h=h, w=w) # reshape for quantizer
        elif self.teacher == 'siglip_384':
            vqkd_x = self.scaling_layer(x)
            vqkd_feature = self.encoder_vqkd(vqkd_x, output_hidden_states=True).hidden_states[-2]
            vqkd_feature = self.encode_task_layer(vqkd_feature.type_as(self.encode_task_layer[-1].weight))
            N = vqkd_feature.shape[1]
            h, w = int(math.sqrt(N)), int(math.sqrt(N))
            vqkd_feature = rearrange(vqkd_feature, 'b (h w) c -> b c h w', h=h, w=w) # reshape for quantizer
        else:
            raise ValueError(f'teacher {self.teacher} not supported')

        # vqgan
        if self.teacher in ['clipb_224', 'vitamin_xlarge_256']:
            vqgan_features = self.encoder(x)
            vqgan_features = self.quant_conv(vqgan_features)
        elif self.teacher == 'siglip_384':
            upscale_reso = 2**4 * vqkd_feature.shape[-1]
            vqgan_x = F.interpolate(x, size=(upscale_reso, upscale_reso), mode='bicubic')
            vqgan_features = self.encoder(vqgan_x)
            vqgan_features = self.quant_conv(vqgan_features)
        else:
            raise ValueError(f'teacher {self.teacher} not supported')

        h = torch.cat([vqkd_feature, vqgan_features], dim=1)
        if self.enable_var:
            # VAR
            residual = h
            final_quantize = 0.
            all_loss = []
            all_inds = []

            new_layers = self.scale_rq_layers
            if self.training and self.random_scale_drop:
                if random.random() < self.random_scale_drop_ratio:
                    new_layers = drop_scale(self.scale_rq_layers, self.num_of_random_drop)

            for i, scale_size in enumerate(new_layers): # TODO add random drop
                residual_si = F.interpolate(residual, size=(scale_size, scale_size), mode='area')
                quantize, emb_loss, info = self.quantize(residual_si)
                quantize = F.interpolate(quantize, size=(self.scale_rq_layers[-1], self.scale_rq_layers[-1]), mode='bicubic')
                
                if self.using_shared_conv: # default not use
                    quantize_vqkd, quantize_vqgan = torch.split(quantize, split_size_or_sections=[self.semantic_code_dim, self.vqgan_code_dim], dim=1)
                    quantize_vqkd = self.vqkd_quant_resi[i/(self.rq_depth-1)](quantize_vqkd)
                    quantize_vqgan = self.vqgan_quant_resi[i/(self.rq_depth-1)](quantize_vqgan)
                    quantize = torch.cat([quantize_vqkd, quantize_vqgan], dim=1)

                final_quantize += quantize
                residual = residual - quantize.detach()

                all_loss.append(emb_loss)
                inds = info[-1].reshape([-1, scale_size ** 2])
                all_inds.append(inds)

            all_loss = zip(*all_loss)
            all_loss = [sum(group) / len(group) if group[0] is not None else None for group in all_loss]
            all_inds = torch.cat(all_inds, dim=1)

            return final_quantize, all_loss, all_inds
        else:
            quant, emb_loss, info = self.quantize(h)
            return quant, emb_loss, info

    def decode(self, quant):
        vqkd_quant, vqgan_quant = torch.split(quant, [self.semantic_code_dim, self.vqgan_code_dim], dim=1)
        vqkd_recon = self.decoder_vqkd(vqkd_quant, return_patch_tokens=True)
        vqkd_recon = self.decode_task_layer(vqkd_recon)

        vqgan_recon = self.post_quant_conv(vqgan_quant)
        vqgan_recon = self.decoder(vqgan_recon)

        if self.teacher == 'siglip_384':
            vqgan_recon = F.interpolate(vqgan_recon, size=(384, 384), mode='bicubic')

        dec = (vqkd_recon, vqgan_recon)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        if not self.enable_var and self.use_kmeans:
            self.enable_var = True # kmeans use last scale features instead of 1*1, and change to multiscale setting after kmeans init. modify this for different teacher
            self.scale_rq_layers = [1,2,4,6,8,10,12,14,16] 
            self.rq_depth = len(self.scale_rq_layers)
        return dec, diff



class ScalingLayerForClip(nn.Module):
    def __init__(self):
        super(ScalingLayerForClip, self).__init__()
        self.register_buffer('shift', torch.Tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None])

    def forward(self, inp):
        inp = ((inp + 1.) * 127.5).clamp(0, 255.) / 255. # rescale to [0, 1.]
        return (inp - self.shift) / self.scale

class ScalingLayerForSigLip(nn.Module):
    def __init__(self):
        super(ScalingLayerForSigLip, self).__init__()
        self.register_buffer('shift', torch.Tensor([0.5, 0.5, 0.5])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([0.5, 0.5, 0.5])[None, :, None, None])

    def forward(self, inp):
        inp = ((inp + 1.) * 127.5).clamp(0, 255.) / 255. # rescale to [0, 1.]
        return (inp - self.shift) / self.scale



class Encoder(nn.Module):
    def __init__(self, in_channels=3, ch=128, ch_mult=(1,1,2,2,4), num_res_blocks=2, 
                 norm_type='group', dropout=0.0, resamp_with_conv=True, z_channels=256):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        # downsampling
        in_ch_mult = (1,) + tuple(ch_mult)
        self.conv_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # downsample
            if i_level != self.num_resolutions-1:
                conv_block.downsample = Downsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # middle
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        h = self.conv_in(x)
        # downsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.downsample(h)
        
        # middle
        for mid_block in self.mid:
            h = mid_block(h)
        
        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class Decoder(nn.Module):
    def __init__(self, z_channels=256, ch=128, ch_mult=(1,1,2,2,4), num_res_blocks=2, norm_type="group",
                 dropout=0.0, resamp_with_conv=True, out_channels=3):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch*ch_mult[self.num_resolutions-1]
        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

       # middle
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # upsampling
        self.conv_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # downsample
            if i_level != 0:
                conv_block.upsample = Upsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    @property
    def last_layer(self):
        return self.conv_out.weight
    
    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)

        # middle
        for mid_block in self.mid:
            h = mid_block(h)
        
        # upsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks + 1):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


### shared quantizer
class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, entropy_loss_ratio, l2_norm, show_usage, split, kmeans=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.split = split

        self.kmeans_init = kmeans
        self.initted = False
        
        if self.kmeans_init: # default not use
            print("using kmeans init")
            self.embedding_vqkd = EmbeddingEMA(self.n_e, self.split[0])
            self.embedding_vqkd.weight.requires_grad = False

            self.embedding_vqgan = EmbeddingEMA(self.n_e, self.split[1])
            self.embedding_vqgan.weight.requires_grad = False
        else:
            print("no kmeans init")
            self.embedding_vqkd = nn.Embedding(self.n_e, self.split[0])
            self.embedding_vqkd.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
            if self.l2_norm:
                self.embedding_vqkd.weight.data = F.normalize(self.embedding_vqkd.weight.data, p=2, dim=-1)

            self.embedding_vqgan = nn.Embedding(self.n_e, self.split[1])
            self.embedding_vqgan.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
            if self.l2_norm:
                self.embedding_vqgan.weight.data = F.normalize(self.embedding_vqgan.weight.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(131072)))

        

    def forward(self, z):
        z = torch.einsum('b c h w -> b h w c', z).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        z_vqkd, z_vqgan = torch.split(z, split_size_or_sections=self.split, dim=-1)
        z_flattened_vqkd, z_flattened_vqgan = torch.split(z_flattened, split_size_or_sections=self.split, dim=-1)

        if self.l2_norm:
            z_flattened_vqkd = F.normalize(z_flattened_vqkd, p=2, dim=-1)
            z_flattened_vqgan = F.normalize(z_flattened_vqgan, p=2, dim=-1)
            z_flattened = torch.cat([z_flattened_vqkd, z_flattened_vqgan], dim=-1)
        if self.kmeans_init and not self.initted: # default not use kmeans
            with torch.no_grad():
                # Gather outputs from all GPUs
                z_flattened_vqkds = [torch.zeros_like(z_flattened_vqkd) for _ in range(torch.distributed.get_world_size())]
                dist.all_gather(z_flattened_vqkds, z_flattened_vqkd)
                combined_z_flatteneds = torch.cat(z_flattened_vqkds, dim=0)
                print("combined_z_flatteneds.shape", combined_z_flatteneds.shape)

                self.embedding_vqkd.init_embed_(combined_z_flatteneds)
                self.initted = True # kmeans from the first batch

        if self.l2_norm:
            z_vqkd = F.normalize(z_vqkd, p=2, dim=-1)
            embedding_vqkd = F.normalize(self.embedding_vqkd.weight, p=2, dim=-1)

            z_vqgan = F.normalize(z_vqgan, p=2, dim=-1)
            embedding_vqgan = F.normalize(self.embedding_vqgan.weight, p=2, dim=-1)

            z = torch.cat([z_vqkd, z_vqgan], dim=-1)

        else:
            embedding_vqkd = self.embedding.weight[:, :self.split[0]]
            embedding_vqgan = self.embedding.weight[:, self.split[0]:]

        d_vqkd = torch.sum(z_flattened_vqkd ** 2, dim=1, keepdim=True) + \
                torch.sum(embedding_vqkd**2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn', z_flattened_vqkd, torch.einsum('n d -> d n', embedding_vqkd))
        d_vqgan = torch.sum(z_flattened_vqgan ** 2, dim=1, keepdim=True) + \
                torch.sum(embedding_vqgan**2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn', z_flattened_vqgan, torch.einsum('n d -> d n', embedding_vqgan))
        
        vqkd_d_norm = torch.mean(torch.sum(d_vqkd**2, dim=-1))
        vqgan_d_norm = torch.mean(torch.sum(d_vqgan**2, dim=-1))
        
        ### shared mapping ###
        d = d_vqkd + 1.0 * d_vqgan
        min_encoding_indices = torch.argmin(d, dim=1)
        ### shared mapping ###

        aggregate_usage = False
        if aggregate_usage:
            with torch.no_grad():
                min_encoding_indices_all = [torch.zeros_like(min_encoding_indices) for _ in range(torch.distributed.get_world_size())]
                dist.all_gather(min_encoding_indices_all, min_encoding_indices)
                min_encoding_indices_all = torch.cat(min_encoding_indices_all, dim=0)

        all_embedding = torch.cat([embedding_vqkd, embedding_vqgan], dim=-1)

        z_q = all_embedding[min_encoding_indices].view(z.shape)
        z_q_vqkd = embedding_vqkd[min_encoding_indices].view(z_vqkd.shape)
        z_q_vqgan = embedding_vqgan[min_encoding_indices].view(z_vqgan.shape)


        perplexity = None
        min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None
        codebook_usage = 0

        if self.show_usage and self.training:
            if aggregate_usage:
                cur_len = min_encoding_indices_all.shape[0]
                self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
                self.codebook_used[-cur_len:] = min_encoding_indices_all
                codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e
            else:
                cur_len = min_encoding_indices.shape[0]
                self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
                self.codebook_used[-cur_len:] = min_encoding_indices
                codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e
            

        encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype)     

        if self.kmeans_init and self.training: # default not use, if using kmeans, remember to remove the vq loss and commit loss
            bins = encodings.sum(0)
            dist.all_reduce(bins)

            zero_mask = (bins == 0)
            bins = bins.masked_fill(zero_mask, 1.)

            embed_sum = z_flattened_vqkd.t() @ encodings
            dist.all_reduce(embed_sum)
                        
            embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
            embed_normalized = l2norm(embed_normalized)
            
            embed_normalized = torch.where(zero_mask[..., None], self.embedding_vqkd.weight,
                                           embed_normalized)
            
            norm_ema_inplace(self.embedding_vqkd.weight, embed_normalized, 0.99)
        
        # compute loss for embedding
        if self.training:
            if self.kmeans_init:
                vq_loss = torch.mean((z_q_vqgan - z_vqgan.detach()) ** 2) 
                commit_loss = self.beta * torch.mean((z_q_vqkd.detach() - z_vqkd) ** 2) + self.beta * torch.mean((z_q_vqgan.detach() - z_vqgan) ** 2)
                entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)
            else:
                vq_loss = torch.mean((z_q - z.detach()) ** 2) 
                commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) 
                entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = torch.einsum('b h w c -> b c h w', z_q)

        return z_q, (vq_loss, commit_loss, entropy_loss, codebook_usage, vqkd_d_norm, vqgan_d_norm), (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
        if self.l2_norm:
            embedding_vqkd = F.normalize(self.embedding_vqkd.weight, p=2, dim=-1)
            embedding_vqgan = F.normalize(self.embedding_vqgan.weight, p=2, dim=-1)
            embedding = torch.cat([embedding_vqkd, embedding_vqgan], dim=-1)
        else:
            embedding = self.embedding.weight
        z_q = embedding[indices]  # (b*h*w, c)

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                # reshape back to match original input shape
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q
    
    def get_codebook_entry_outside(self, indices, outside_embedding, shape=None, channel_first=True):
        # shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
        if self.l2_norm:
            embedding_vqkd = F.normalize(outside_embedding[0].weight, p=2, dim=-1)
            embedding_vqgan = F.normalize(outside_embedding[1].weight, p=2, dim=-1)
            embedding = torch.cat([embedding_vqkd, embedding_vqgan], dim=-1)
        else:
            embedding = self.embedding.weight
        z_q = embedding[indices]  # (b*h*w, c)

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels, norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type='group'):
        super().__init__()
        self.norm = Normalize(in_channels, norm_type)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


class Phi(nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks//2)
        self.resi_ratio = abs(quant_resi)
    
    def forward(self, h_BChw):
        return h_BChw.mul(1-self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)

class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, norm_type='group'):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return nn.SyncBatchNorm(in_channels)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss

#################################################################################
#                              VQ Model Configs                                 #
#################################################################################

def TokenFlowFunc(**kwargs):
    return TokenFlow(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

VQ_models = {'TokenFlow': TokenFlowFunc}
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import math
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
import random
from llava_t2i.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VQ_TOKEN_TEMPLATE

from llava_t2i.mm_utils import get_anyres_image_grid_shape
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            # self.vision_tower.load_model()
            config.mm_hidden_size = self.vision_tower.hidden_size

            self.mm_projector = build_vision_projector(config)
            # add scale embedding， add extra 1 for text token
            self.lvl_embed = nn.Embedding(len(self.vision_tower.var_scales) + 1, config.hidden_size)
            init_std = math.sqrt(1 / config.hidden_size / 3)
            nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

            # 3. absolute position embedding
            pos_1LC = []
            for i, pn in enumerate(self.vision_tower.var_scales):
                pe = torch.empty(pn, config.hidden_size)
                nn.init.trunc_normal_(pe, mean=0, std=init_std)
                pos_1LC.append(pe)
            pos_1LC = torch.cat(pos_1LC, dim=0)     # 1, L, C
            assert tuple(pos_1LC.shape) == (self.vision_tower.num_patches, config.hidden_size)
            self.pos_1LC = nn.Parameter(pos_1LC)

            for lid in range(len(self.layers)):
                # qk norm
                self.layers[lid].self_attn.qknorm = QKNorm(dim=config.hidden_size//config.num_attention_heads)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
        vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.vq_codebook_size = getattr(vision_tower, 'num_codebook_tokens', -1)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        init_std = math.sqrt(1 / self.config.hidden_size / 3)

        if getattr(self, 'lvl_embed', None) is None:
            self.lvl_embed = nn.Embedding(len(self.vision_tower.var_scales) + 1, self.config.hidden_size)
            nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

        else:
            # In case it is frozen by LoRA
            for p in self.lvl_embed.parameters():
                p.requires_grad = True

        # add modulated_layers in decoder_layers
        for lid in range(len(self.layers)):
            if getattr(self.layers[lid].self_attn, 'qknorm', None) is None:
                # qk norm
                self.layers[lid].self_attn.qknorm = QKNorm(dim=self.config.hidden_size//self.config.num_attention_heads)
            else:
                for p in self.layers[lid].self_attn.qknorm.parameters():
                    p.requires_grad = True

        if getattr(self, 'pos_1LC', None) is None:
            pos_1LC = []
            for i, pn in enumerate(self.vision_tower.var_scales):
                pe = torch.empty(pn, self.config.hidden_size)
                nn.init.trunc_normal_(pe, mean=0, std=init_std)
                pos_1LC.append(pe)
            pos_1LC = torch.cat(pos_1LC, dim=0)     # 1, L, C

            assert tuple(pos_1LC.shape) == (self.vision_tower.num_patches, self.config.hidden_size), (pos_1LC.shape, (self.vision_tower.num_patches, self.config.hidden_size))
            self.pos_1LC = nn.Parameter(pos_1LC)
        else:
            # In case it is frozen by LoRA
            self.pos_1LC.requires_grad = True
        
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            self.lvl_embed.load_state_dict(get_w(mm_projector_weights, 'lvl_embed'))
            self.pos_1LC.load_state_dict(get_w(mm_projector_weights, 'pos_1LC'))
            print('resume pretrained mm_projector, lvl_embed')


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, vision_code_inds, pred_for_curr_shape=None, visualize=False):
        # output: b n c
        image_features = self.get_model().get_vision_tower()(vision_code_inds, pred_for_curr_shape=pred_for_curr_shape)
        image_features = self.get_model().mm_projector(image_features)

        return image_features, None

    def initialize_lm_head(self, model_args):
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.lm_head.load_state_dict(get_w(mm_projector_weights, 'lm_head'))
            print('resume pretrained lm head')


    def prepare_inputs_labels_for_multimodal_generation(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, training=True, img_ind_list = (), text_labels=None, pred_for_curr_shape=False
    ):
        # TokenFlow # for vq tower encoder, we have replaced the image token with special tokens("<vq_token_%d>"), and are added to the text tokenizer;
        # all images have been preprocessed into special tokens(including "<img>", "<\img>"), so the input_ids already include image;
        # here we need replace the embedding of the image token with the feature in vq codebook;
        vision_tower = self.get_vision_tower()

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
            text_labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        text_labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(text_labels, attention_mask)]

        # replace vision inds
        if images is not None:
            vision_inds = self.get_model().get_vision_tower().get_vq_codes(images)

            assert len(vision_inds) == len(input_ids), (vision_inds.shape, input_ids.shape)
            for batch_idx, cur_input_ids in enumerate(input_ids):
                if len(img_ind_list) == len(input_ids):
                    img_inds = img_ind_list[batch_idx]
                else:
                    condition = (cur_input_ids >= self.image_token_start) & (cur_input_ids <= self.image_token_end)
                    img_inds = torch.where(condition)[0]
                    # assert len(img_inds) == img_inds[-1]-img_inds[0] + 1
                if len(img_inds) !=img_inds[-1]-img_inds[0] + 1:
                    print(f'warning: len(img_inds) == img_inds[-1]-img_inds[0] + 1, {len(img_inds)} vs {img_inds[-1], img_inds[0] }. Skip replacing vision embedding.')
                    continue
                # assert len(img_inds) == vision_tower.num_patches
                if len(img_inds) != vision_tower.num_patches:
                    print(f'warning: len(img_inds) != vision_tower.num_patches, {len(img_inds)} vs {vision_tower.num_patches}. Skip replacing vision embedding.')
                    continue
                
                cur_input_ids[img_inds] = vision_inds[batch_idx] + self.image_token_start
                labels[batch_idx][img_inds] = vision_inds[batch_idx] + self.image_token_start

        new_input_embeds = []
        new_labels = []
        new_input_ids = []
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)

        fisrt_level_len = (vision_tower.scale_rq_layers[0]) ** 2
        for batch_idx, cur_input_ids in enumerate(input_ids):

            if len(img_ind_list) == len(input_ids):
                img_inds = img_ind_list[batch_idx]
            else:
                condition = (cur_input_ids >= self.image_token_start) & (cur_input_ids <= self.image_token_end)
                img_inds = torch.where(condition)[0]
                # assert len(img_inds) == img_inds[-1]-img_inds[0] + 1
            if len(img_inds) !=img_inds[-1]-img_inds[0] + 1:
                print(f'warning: len(img_inds) == img_inds[-1]-img_inds[0] + 1, {len(img_inds)} vs {img_inds[-1], img_inds[0] }. Skip replacing vision embedding.')
                continue
            # assert len(img_inds) == vision_tower.num_patches
            if len(img_inds) != vision_tower.num_patches:
                print(f'warning: len(img_inds) != vision_tower.num_patches, {len(img_inds)} vs {vision_tower.num_patches}. Skip replacing vision embedding.')
                continue

            vision_code_inds = cur_input_ids[img_inds] - self.image_token_start
            if (vision_code_inds < 0).any():
                print(f'warning: values in vision_code_inds is negative, force them into 0')
                vision_code_inds[vision_code_inds<0] = 0
            img_embeds, all_samples = self.encode_images(
                vision_code_inds.unsqueeze(0), 
                pred_for_curr_shape=pred_for_curr_shape, 
                visualize=False
                )  # N c
            img_embeds = img_embeds[0]

            # # an workaround for the shift operation on lables and logits when calculating next token prediction loss,
            # # WE append an empty token behind, this makes the inputs length equals to the labels
            extra_last_token_placeholder = torch.zeros_like(img_embeds[:fisrt_level_len, :])

            text_cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
            text_embeds, no_use_vision_embeds, text_embeds_after = torch.split(text_cur_input_embeds, 
                                                                            [img_inds[0], len(img_inds), len(text_cur_input_embeds)-img_inds[-1]-1],
                                                                            dim=0)
            cur_input_embeds = torch.cat([text_embeds, img_embeds, extra_last_token_placeholder, text_embeds_after], dim=0)
            assert len(cur_input_embeds) == len(text_cur_input_embeds)

            scale_embedding_inds = torch.zeros([len(cur_input_embeds)], dtype=torch.long, device=cur_input_embeds.device)
            vision_scale_inds = torch.cat([torch.full((pn*pn,), i+1, device=scale_embedding_inds.device) for i, pn in enumerate(vision_tower.scale_rq_layers)])[fisrt_level_len:]
            scale_embedding_inds[img_inds[:-fisrt_level_len]] = vision_scale_inds.long()
            scale_embedding = self.get_model().lvl_embed(scale_embedding_inds)

            cur_input_embeds += scale_embedding

            # abs positional encoding
            abs_positional_embedding = torch.zeros([len(cur_input_embeds), self.get_model().pos_1LC.shape[-1]]).to(self.get_model().pos_1LC)
            abs_positional_embedding[img_inds-1] = self.get_model().pos_1LC
            cur_input_embeds += abs_positional_embedding

            new_input_embeds.append(cur_input_embeds)
            # （---）no need the shift, it will be conducted when calculating loss
            new_labels.append(labels[batch_idx])
            new_input_ids.append(cur_input_ids)

        # Truncate sequences to max length as image embeddings can make the sequence longer

        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            text_labels = [x[:tokenizer_model_max_length] for x in text_labels]
            new_input_ids = [x[:tokenizer_model_max_length] for x in new_input_ids]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        new_text_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    new_text_labels_padded[i, -len(text_labels[i]):] = text_labels[i]
                    attention_mask[i, -cur_len:] = True
                    attention_mask[i, -cur_len:] = torch.where(new_input_ids[i]==self.pad_token_id, False, attention_mask[i, -cur_len:])
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    new_text_labels_padded[i, :len(text_labels[i])] = text_labels[i]
                    attention_mask[i, :cur_len] = True
                    attention_mask[i, :cur_len] = torch.where(new_input_ids[i]==self.pad_token_id, False, attention_mask[i, :cur_len])
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        # TODO convert attention mask to 4d VAR mask
        d = torch.cat([torch.full((pn*pn,), i, device=new_input_embeds.device) for i, pn in enumerate(vision_tower.scale_rq_layers)]).view(1, vision_tower.num_patches, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        attn_bias_for_masking = torch.where(d >= dT, 0., torch.finfo(new_input_embeds.dtype).min).reshape(1, vision_tower.num_patches, vision_tower.num_patches).to(new_input_embeds)

        attn_mask_converter = AttentionMaskConverter(is_causal=True)
        attention_mask = attn_mask_converter.to_4d(
            attention_mask, max_len, key_value_length=max_len, dtype=new_input_embeds.dtype
        )
        for i in range(batch_size):
            # assert len(img_ind_list) == batch_size, (len(img_ind_list), batch_size)
            if len(img_ind_list) == batch_size:
                img_inds = img_ind_list[i]
            else:
                condition = (new_input_ids[i] >= self.image_token_start) & (new_input_ids[i] <= self.image_token_end)
                img_inds = torch.where(condition)[0]
            img_inds = img_inds -1

            attention_mask[i, :, img_inds[:, None], img_inds] = attn_bias_for_masking

        # the model will convert to zero and minus-infinity in forward
        # for transformers >= 4.43.4, no need to invert the attention mask
        # attention_mask = (attention_mask > -1).to(new_input_embeds.dtype)

        new_text_labels = new_text_labels_padded
        new_labels = new_labels_padded

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, new_text_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        # if model_args.mm_use_im_patch_token:
        #     tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        #     self.resize_token_embeddings(len(tokenizer))
        if model_args.mm_use_vq_token:
            assert self.config.vq_codebook_size > 0
            add_tokens = [DEFAULT_VQ_TOKEN_TEMPLATE % i for i in range(self.config.vq_codebook_size)]
            tokenizer.add_tokens(add_tokens, special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            image_token_start = tokenizer.convert_tokens_to_ids(add_tokens[0])
            image_token_end = tokenizer.convert_tokens_to_ids(add_tokens[-1])
            self.register_buffer('image_token_start', torch.tensor(image_token_start).to(self.get_vision_tower().device))
            self.register_buffer('image_token_end', torch.tensor(image_token_end).to(self.get_vision_tower().device))

            # var_scales = self.get_vision_tower().var_scales
            # self.register_buffer('var_scales', var_scales)

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token or model_args.mm_use_vq_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

        self.initialize_lm_head(model_args)

    def reinit_image_token_start_end(self, tokenizer):
        num_new_tokens = tokenizer.add_special_tokens(dict(pad_token="[PAD]"))
        pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.pad_token_id = pad_token_id
        add_tokens = [DEFAULT_VQ_TOKEN_TEMPLATE % i for i in range(self.config.vq_codebook_size)]

        self.image_token_start = tokenizer.convert_tokens_to_ids(add_tokens[0])
        self.image_token_end = tokenizer.convert_tokens_to_ids(add_tokens[-1])
        print(self.image_token_start , self.image_token_end )


from llava_t2i.model.language_model.modeling_llama import LlamaRMSNorm
from torch import Tensor, nn
class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = LlamaRMSNorm(dim, eps=1e-6)
        self.key_norm = LlamaRMSNorm(dim, eps=1e-6)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)

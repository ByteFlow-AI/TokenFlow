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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig
from .modeling_llama import LlamaForCausalLM, LlamaModel
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass

from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

@dataclass
class CloudCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    loss_vision: Optional[torch.FloatTensor] = None
    loss_text: Optional[torch.FloatTensor] = None

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        # cache_position: Optional[torch.LongTensor] = None,  # only for transformers version 4.38.2
        text_labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CloudCausalLMOutputWithPast]:
        # print('output_hidden_states', output_hidden_states, self.config.output_hidden_states)
        # print('return_dict', return_dict, self.config.use_return_dict)
        # print('use_cache', use_cache, self.config.use_cache)
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                text_labels
            ) = self.prepare_inputs_labels_for_multimodal_generation(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                text_labels=text_labels
            )

        #  copied and modified from LlamaForCausalLM forward
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # cache_position=cache_position, # only for transformers version 4.38.2
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss_vision = loss_fct(shift_logits, shift_labels)

            if text_labels is not None:
                shift_text_labels = text_labels[..., 1:].contiguous()
                shift_text_labels = shift_text_labels.view(-1)
                # Enable model parallelism
                shift_text_labels = shift_text_labels.to(shift_logits.device)
                loss_text = loss_fct(shift_logits, shift_text_labels)
                loss = loss_vision + loss_text
            else:
                loss = loss_vision
                loss_text = 0. * loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output + (loss_vision, loss_text) if loss is not None else output

        return CloudCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            loss_vision=loss_vision,
            loss_text=loss_text
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs
    
    def prepare_generation_inputs(self, cur_inputs_ids, input_labels=None, img_ind_list=(), padding_value=0, pred_for_curr_shape=None):
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        assert (isinstance(cur_inputs_ids, list) and cur_inputs_ids[0].ndim == 1) or cur_inputs_ids.ndim == 2
        
        full_target_inputs = torch.zeros((len(cur_inputs_ids), tokenizer_model_max_length), dtype=torch.long, device=cur_inputs_ids[0].device)
        for i in range(len(cur_inputs_ids)):
            if len(img_ind_list) == len(cur_inputs_ids):
                full_target_inputs[i][img_ind_list[i]] = padding_value
            full_target_inputs[i, :len(cur_inputs_ids[i])] = cur_inputs_ids[i]
            # print('fill length: %d, img start ind: %d' % (len(cur_inputs_ids[i]), img_ind_list[i][0]))
        cur_attention_mask = torch.ones_like(full_target_inputs, dtype=torch.bool)

        (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                text_labels
            ) = self.prepare_inputs_labels_for_multimodal_generation(full_target_inputs, labels=input_labels, 
                                                                      position_ids=None, attention_mask=cur_attention_mask, past_key_values=None, images=None,
                                                                      training=False, 
                                                                      img_ind_list=img_ind_list,
                                                                      pred_for_curr_shape=pred_for_curr_shape,
                                                                      )
        return inputs_embeds, attention_mask, labels
    
    @torch.no_grad()
    def autoregressive_infer_cfg(self, B, prefix_text_codes=None, g_seed=None, 
                                 cfg=1.5, topk_list=[600], topp_list=[0.6]
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param prefix_text_codes: text prompt code
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        """

        vision_tower = self.get_vision_tower()
        device = vision_tower.device

        if g_seed is None: 
            rng = None
        else:
            rng = torch.Generator(device=device)
            rng.manual_seed(g_seed)

        B = len(prefix_text_codes)//2
        input_codes = prefix_text_codes
        cur_Ls = [len(i)-1 for i in prefix_text_codes]

        img_ind_list = []
        for i in range(len(input_codes)):
            start = len(input_codes[i])
            img_inds = torch.arange(start, vision_tower.num_patches + start).to(input_codes[i].device)
            img_ind_list.append(img_inds)

        image_token_start = getattr(self, 'image_token_start', 0)
        print('image_token_start', image_token_start)
        print('scales:', vision_tower.scale_rq_layers)
        ctx_length = getattr(self.config, 'tokenizer_model_max_length', None)
        print('context length: ', ctx_length)

        # multi_step_inferience strategy
        multi_step_infer_start = 1
        # topk_list = [top_k, 100, 1]
        # topp_list = [top_p, 0.8, 0]
        for si, pn in enumerate(vision_tower.scale_rq_layers):   # si: i-th segment
            ratio = si / (len(vision_tower.scale_rq_layers) - 1)

            for loop in range(len(topk_list)):
                inputs_embeds, \
                attention_mask, \
                labels = self.prepare_generation_inputs(input_codes, 
                                                        img_ind_list=img_ind_list,
                                                        padding_value=image_token_start+1,
                                                        pred_for_curr_shape=None if loop == 0 else si-1)
                if loop == 0:
                    max_len = max([cur_Ls[i]+pn*pn for i in range(2*B)])
                else:
                    max_len = max([cur_Ls[i] for i in range(2*B)])
                inputs_embeds = inputs_embeds[:, :max_len]
                attention_mask = attention_mask[:, :, :max_len, :max_len]

                logits = self.forward(
                                        input_ids=None,
                                        attention_mask=attention_mask,
                                        position_ids=None,
                                        past_key_values=None,
                                        inputs_embeds=inputs_embeds,
                                        labels=None,
                                        use_cache=False,
                                        output_attentions=False,
                                        output_hidden_states=False,
                                        return_dict=False
                                    )[0]
                if loop == 0:
                    logits = torch.stack([logits[i, cur_Ls[i]:cur_Ls[i]+pn*pn] for i in range(2*B)], dim=0)  # no use cache
                else:
                    logits = torch.stack([logits[i, cur_Ls[i]-pn*pn:cur_Ls[i]] for i in range(2*B)], dim=0)  # no use cache

                t = cfg * ratio
                logits_BlV = (1+t) * logits[:B] - t * logits[B:]
                logits_BlV[..., :image_token_start] = -torch.inf

                idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=topk_list[loop], top_p=topp_list[loop], num_samples=1)[:, :, 0]
                
                if loop == 0:
                    for i in range(B):
                        input_codes[i] =  torch.cat([input_codes[i], idx_Bl[i]], dim=0)
                        input_codes[i+B] =  torch.cat([input_codes[i+B], idx_Bl[i]], dim=0)
                    cur_Ls = [i + pn*pn for i in cur_Ls]
                else:
                    for i in range(B):    
                        input_codes[i][-len(idx_Bl[i]):] = idx_Bl[i]
                        input_codes[i+B][-len(idx_Bl[i]):] = idx_Bl[i]       

                # jump out while loop
                if si < multi_step_infer_start:
                    break              
            
        gene_codes = torch.stack([input_codes[i][img_ind_list[i]] for i in range(B)], dim=0)
        gene_codes = gene_codes - image_token_start
        samples = vision_tower.decode_to_img_pt(gene_codes)

        samples = torch.clamp(samples.permute(0, 2, 3, 1).contiguous() * 127.5 + 128.0, 0, 255).to("cpu", dtype=torch.uint8)

        return samples   # de-normalize, from [-1, 1] to [0, 1]


def sample_with_top_k_top_p_(logits_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = logits_BlV.shape
    if top_k > 0:
        idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
    # sample (have to squeeze cuz torch.multinomial can only be used for 2D tensor)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)

AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)

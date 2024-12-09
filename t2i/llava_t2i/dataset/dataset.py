import os
import io
import math
import re
import random
import numpy as np
import copy
import json
from typing import Dict, Optional, Sequence, List
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from dataclasses import dataclass, field

from PIL import PngImagePlugin, Image
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)
import transformers
import datasets
from datasets import load_dataset
from llava_t2i.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
DEFAULT_IM_END_TOKEN, IMAGE_START_TOKEN_GEN, IMAGE_END_TOKEN_GEN, DEFAULT_VQ_TOKEN_TEMPLATE_PREFIX, DEFAULT_VQ_TOKEN_TEMPLATE
from .process import preprocess
import re


def split_and_rejoin(sentence, length):
    parts = re.split('([.,])', sentence)

    new_sentence = ""
    current_length = 0

    for part in parts:
        current_length += len(part.split(' '))
        if current_length <= length:
            new_sentence += part
        else:
            break

    return new_sentence


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])



def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])

    
class ExampleDataset(Dataset):
    def __init__(self, 
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args,
                 image_size,
                 patch_num,
                 training=True,
                 split='train'):

        self.ds = load_dataset("sayakpaul/coco-30-val-2014", split=split)

        self.transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])


        self.tokenizer = tokenizer
        self.data_args = data_args
        self.patch_num = patch_num
        self.text_drop_rate = 0.1
    def __len__(self):
        return len(self.ds)
    
    @property
    def lengths(self):
        length_list = [500] * len(self)
        return length_list

    @property
    def modality_lengths(self):
        length_list = [500] * len(self)
        return length_list

    def __getitem__(self, idx):
        try:
            item = self.ds[idx]
            image = item['image'].convert("RGB")
            caption = item['caption']
        except:
            idx = random.randint(0, len(self) - 1)
            item = self.ds[idx]
            image = item['image'].convert("RGB")
            caption = item['caption']

        cls_label = 0
    
        if self.transform:
            image = self.transform(image)

        inds_str = (DEFAULT_VQ_TOKEN_TEMPLATE%0)*self.patch_num
        sources = {
            'text': caption,
            'image': inds_str,
            'label': cls_label
        }
        # print(caption)
        
        if isinstance(idx, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        sources = copy.deepcopy(sources)
        
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=False,
            max_text_token_num=self.data_args.max_text_token_num)
        if isinstance(idx, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],
                             text_labels=data_dict["text_labels"][0],
                             )

        data_dict['image'] = image

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        padding_value = 1000000000  # we dose't use pad_token, worried its index collapse with vision code index
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=padding_value)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(padding_value),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        if 'text_labels' in instances[0] and instances[0]['text_labels'] is not None:
            text_labels = [instance['text_labels'] for instance in instances]
            text_labels = torch.nn.utils.rnn.pad_sequence(text_labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
            text_labels = text_labels[:, :self.tokenizer.model_max_length]
            batch['text_labels'] = text_labels

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args,
                                image_size,
                                patch_num) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = ExampleDataset(tokenizer=tokenizer,
            data_path=data_args.data_path,
            data_args=data_args,
            image_size=image_size,
            patch_num=patch_num
            )
  
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                data_collator=data_collator)

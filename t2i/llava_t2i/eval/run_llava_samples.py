import argparse
import torch
import numpy as np

from llava_t2i.dataset.process import crop_and_encode_text_and_img

from llava_t2i.utils import disable_torch_init

from tqdm import tqdm
import json
from PIL import Image
import os
from PIL import Image
from llava_t2i.model import *
import transformers


multi_step_infer_strategy = {
    1: {
        'topk_list': [600],
        'topp_list': [0.6],
    },
    2: {
        'topk_list': [1200, 1],
        'topp_list': [0.8, 0],
    },
    3: {
        'topk_list': [1200, 100, 1],
        'topp_list': [0.8, 0.8, 0],
    },
}

def eval_model(args):
    cur_output_path = os.path.join(args.output_path, f'loop{args.loop}_cfg{args.cfg}')
    os.makedirs(cur_output_path, exist_ok=True)
    repeat = 4
    prompts = [
        "A realistic landscape shot of theNorthern Lights dancing over asnowy mountain range in Iceland.",
        "A picture of the head of a browncow wearing a halter.",
        "A portrait of a woman.",
        "An elephant walking under the sea.",
    ]
    prompts = [i.strip() for i in prompts for _ in range(repeat)]

    negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
    batch_size = args.batch_size
    topk_list = multi_step_infer_strategy[args.loop]['topk_list']
    topp_list = multi_step_infer_strategy[args.loop]['topp_list']

    # Model
    disable_torch_init()
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("No GPU available. Using CPU instead.")

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    model = LlavaLlamaForCausalLM.from_pretrained(
                args.model_path,
                attn_implementation='eager',
                mm_vision_tower=args.tokenizer_path
            )
    model = model.eval()
    model=model.to(ptdtype).cuda()
    vision_tower = model.get_vision_tower()
    vision_tower.to(ptdtype)

    model.config.mm_vision_vq_type = str(model.config.mm_vision_vq_type)
    
    mm_use_vq_token = getattr(model.config, "mm_use_vq_token", False)
    assert mm_use_vq_token
    model.config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=model.config.tokenizer_model_max_length,
        padding_side="right",
        use_fast=False,
    )
    print('model.config.tokenizer_model_max_length', model.config.tokenizer_model_max_length)
    print(tokenizer.vocab_size)
    print('lm head shape and tokenizer size: ', model.lm_head.weight.shape, len(tokenizer))
    model.reinit_image_token_start_end(tokenizer)


    if batch_size > len(prompts):
        batch_size = len(prompts)
    
    total_num = len(prompts)
    print("length of prompts:", total_num)
    # assert total_num%batch_size == 0
    json.dump(prompts, open(os.path.join(cur_output_path, 'prompts.json'), 'w'))
    for i in tqdm(range(total_num // batch_size+1), desc='Generating images'):
        cur_prompts = prompts[i * batch_size: (i + 1) * batch_size]
        if len(cur_prompts) == 0:
            break
        prefix_text_codes = []
        for pind, p in enumerate(cur_prompts):
            input_id, prefix_len = crop_and_encode_text_and_img(tokenizer, p, image=None, max_text_token_num=128)
            prefix_text_codes += [input_id]
        
        uncondition_input_id, _ = crop_and_encode_text_and_img(tokenizer, negative_prompt, image=None, max_text_token_num=128)
        
        prefix_text_codes += [uncondition_input_id] * len(cur_prompts)
    
        with torch.inference_mode():
            samples = model.autoregressive_infer_cfg(B=batch_size,
                                                    prefix_text_codes=prefix_text_codes, 
                                                    cfg=args.cfg, topk_list=topk_list, topp_list=topp_list, 
                                                    g_seed=None)

        # B H W 3
        # save all images
        for iid, img in enumerate(samples):
            Image.fromarray(img.numpy().astype(np.uint8)).save(os.path.join(cur_output_path,"sample_%d_%d.png" % (i, iid)))
        # break
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--tokenizer-path", type=str)

    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--loop", type=int, default=1)
    parser.add_argument("--output-path", type=str, default='./generation')
    parser.add_argument("--mixed_precision", type=str, default='bf16')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--g_seed", type=int, default=None)
    args = parser.parse_args()

    eval_model(args)

import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .vision_tokenizer import VQTower
from llava_t2i.mm_utils import VQType


def build_vision_tower(vision_tower_cfg, **kwargs):

    mm_vision_vq_type = getattr(vision_tower_cfg, 'mm_vision_vq_type', VQType.CLIP)
    if isinstance(mm_vision_vq_type, str):
        mm_vision_vq_type = vision_tower_cfg.mm_vision_vq_type = VQType[mm_vision_vq_type]
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if mm_vision_vq_type == VQType.CLIP or vision_tower.startswith("openai"):
        if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
            if use_s2:
                return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
            else:
                return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    # elif mm_vision_vq_type in [VQType.VQGAN, VQType.VQKD, VQType.LFQGAN, VQType.VQGANLC, VQType.LLAMAGEN_VQKD, VQType.LLAMAGEN_VQKD_INFER]:
    else:
        if is_absolute_path_exists:
            return VQTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')

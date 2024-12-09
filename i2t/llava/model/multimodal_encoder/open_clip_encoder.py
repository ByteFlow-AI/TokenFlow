import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipVisionModel

from transformers.modeling_utils import get_parameter_device, get_parameter_dtype


class SigLipVisionTower(nn.Module):
    def __init__(self):
        super().__init__()

        self.is_loaded = False

        self.load_model()
        self.embed_dim = 1152
        self.n_embed = 256
        self.compression = 2**4

    @property
    def dtype(self):
        return get_parameter_dtype(self)

    @property
    def device(self):
        return get_parameter_device(self)
    
    def load_model(self, device_map=None):
        if self.is_loaded:
            print('openclip model is already loaded')
            return

        self.vision_tower = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    @torch.no_grad()
    def encode(self, images):
        # image_features = self.vision_tower.forward_intermediates(images, intermediates_only=True)[-1]
        image_features = self.vision_tower(images, output_hidden_states=True).hidden_states[-2]

        return image_features, [], [0]

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = image_forward_out.to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = image_forward_outs.to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def hidden_size(self):
        return 1152
        # return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return 16
        # return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (256 // 16) ** 2

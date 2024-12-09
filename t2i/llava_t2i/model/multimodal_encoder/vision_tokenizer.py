import torch
import torch.nn as nn
from .tokenflow import tokenflow_model
import os
from llava_t2i.mm_utils import VQType
import torch.nn.functional as F


class VQTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        
        self.vision_tower_name = vision_tower
        self.vq_type = getattr(args, 'mm_vision_vq_type', VQType.TOKEN_FLOW)
        if self.vq_type == VQType.TOKEN_FLOW:
            vq_model = tokenflow_model('TokenFlow', pretrain_path=self.vision_tower_name)
            vq_model.eval()
            vq_model.requires_grad_(False)

            self.vq_model = [vq_model]
        else:
            raise NotImplementedError

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        assert os.path.exists(self.vision_tower_name), "VQGAN model path is invalid: %s" % self.vision_tower_name
       
        if self.vq_type == VQType.TOKEN_FLOW:
            img_size = 256
            vq_model = tokenflow_model('TokenFlow', pretrain_path=self.vision_tower_name)
            vq_model.eval()
            vq_model.requires_grad_(False)
            if device_map:
                vq_model = vq_model.to(device_map)
            self.vq_model = [vq_model]
        else:
            raise NotImplementedError
            
        self.image_size = img_size
        self.is_loaded = True
    
    def get_vq_model(self):
        if isinstance(self.vq_model, list):
            return self.vq_model[0]
        else:
            return self.vq_model

    def to(self, *args, **kwargs):
        self.vq_model = [self.vq_model[0].to(*args, **kwargs)]
        super(VQTower, self).to(*args, **kwargs)
    
    @torch.no_grad()
    def get_vq_codes(self, images):
        image_features, _, inds = self.get_vq_model().encode(images.to(device=self.device, dtype=self.dtype))
        return inds
    
    def get_vq_features(self, inds):
        image_features = self.get_vq_model().quantize.get_codebook_entry(inds)
        return image_features

    def forward(self, vision_inds, pred_for_curr_shape=None):
        if type(vision_inds) is list:
            raise NotImplementedError
        else:
            assert self.vq_type == VQType.TOKEN_FLOW
            inds = vision_inds
            assert len(inds.shape) == 2
            flattend_inds = inds.flatten().to(device=self.device, dtype=torch.int32)
            image_features = self.get_vq_features(flattend_inds)
            image_features = image_features.reshape([inds.shape[0], inds.shape[1], -1])

            assert inds.shape[1] == self.num_patches, (inds.shape, self.num_patches)
            feature_without_first_level = []
            curr_scales = []
            cur = 0
            f_hat = 0.
            for i in range(len(self.get_vq_model().scale_rq_layers)-1):
                scale = self.get_vq_model().scale_rq_layers[i]
                cur_feature = image_features[:, cur:cur+scale**2].reshape([inds.shape[0], scale, scale, -1]).permute(0, 3, 1, 2)
                next_scale = self.get_vq_model().scale_rq_layers[i+1]
                f_hat += cur_feature

                # before resize, append it to curr scales
                curr_scales.append(f_hat.permute([0, 2, 3, 1]).flatten(1, 2).clone())
                f_hat = F.interpolate(f_hat, size=(next_scale, next_scale), mode='bicubic')
                feature_without_first_level.append(f_hat.permute([0, 2, 3, 1]).flatten(1, 2).clone())
                cur += scale**2
            
            last_feat = image_features[:, cur:].reshape([inds.shape[0], self.get_vq_model().scale_rq_layers[-1], self.get_vq_model().scale_rq_layers[-1], -1]).permute(0, 3, 1, 2)
            f_hat += last_feat
            
            curr_scales.append(f_hat.permute([0, 2, 3, 1]).flatten(1, 2).clone())

            if pred_for_curr_shape is not None and pred_for_curr_shape is not False:
                feature_without_first_level[pred_for_curr_shape] = curr_scales[pred_for_curr_shape+1]

            image_features = torch.cat(feature_without_first_level, dim=1).to(self.dtype)

        return image_features
    
    def decode_to_img_pt(self, codes_Bl):

        assert codes_Bl.ndim == 2
        assert codes_Bl.shape[1] == self.num_patches

        flattend_inds = codes_Bl.flatten().to(device=self.device, dtype=torch.int32)
        image_features = self.get_vq_features(flattend_inds)
        image_features = image_features.reshape([codes_Bl.shape[0], codes_Bl.shape[1], -1])
        f_hat = 0.
        final_scale = self.get_vq_model().scale_rq_layers[-1]
        cur = 0
        for i in range(len(self.get_vq_model().scale_rq_layers)):
            cur_scale = self.get_vq_model().scale_rq_layers[i]
            cur_feature = image_features[:, cur:cur+cur_scale**2].reshape([codes_Bl.shape[0], cur_scale, cur_scale, image_features.shape[-1]]).permute(0, 3, 1, 2)
            cur_feature = F.interpolate(cur_feature, size=(final_scale, final_scale), mode='bicubic')

            f_hat += cur_feature
            cur += cur_scale**2

        semantic_recon, img_recon = self.get_vq_model().decode(f_hat)
        # B x 3 x H x W, value range: -1 ~ 1
        return img_recon

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.get_vq_model().dtype

    @property
    def device(self):
        return self.get_vq_model().device

    @property
    def hidden_size(self):
        return self.get_vq_model().embed_dim

    @property
    def num_patches_per_side(self):
        return self.image_size // self.get_vq_model().compression

    @property
    def num_patches(self):
        return sum(x**2 for x in self.get_vq_model().scale_rq_layers)
    
    @property
    def var_scales(self):
        return [x**2 for x in self.get_vq_model().scale_rq_layers]
        
    @property
    def num_codebook_tokens(self):
        return self.get_vq_model().n_embed

    @property
    def scale_rq_layers(self):
        return self.get_vq_model().scale_rq_layers


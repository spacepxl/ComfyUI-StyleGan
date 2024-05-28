import os
import sys
import numpy as np
import pickle
import torch

from .slerp import slerp

from . import dnnlib
from . import torch_utils
sys.modules["dnnlib"] = dnnlib
sys.modules["torch_utils"] = torch_utils

import folder_paths

# set the models directory
if "stylegan" not in folder_paths.folder_names_and_paths:
    current_paths = [os.path.join(folder_paths.models_dir, "stylegan")]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["stylegan"]
folder_paths.folder_names_and_paths["stylegan"] = (current_paths, folder_paths.supported_pt_extensions)

class LoadStyleGAN:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stylegan_file": (folder_paths.get_filename_list("stylegan"), ),
            },
        }
    
    RETURN_TYPES = ("STYLEGAN",)
    FUNCTION = "load_stylegan"
    CATEGORY = "StyleGAN"
    
    def load_stylegan(self, stylegan_file):
        with open(folder_paths.get_full_path("stylegan", stylegan_file), 'rb') as f:
            G = pickle.load(f)['G_ema'].cuda()
        return (G,)

class GenerateStyleGANLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stylegan_model": ("STYLEGAN", ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 1024}),
            },
        }
    
    RETURN_TYPES = ("STYLEGAN_LATENT",)
    FUNCTION = "generate_latent"
    CATEGORY = "StyleGAN"
    
    def generate_latent(self, stylegan_model, seed, batch_size):
        torch.manual_seed(seed)
        z = torch.randn([batch_size, stylegan_model.z_dim]).cuda()
        return (z, )

class StyleGANSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stylegan_model": ("STYLEGAN", ),
                "stylegan_latent": ("STYLEGAN_LATENT", ),
                "class_label": ("INT", {"default": -1, "min": -1}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "StyleGAN"
    
    def generate_image(self, stylegan_model, stylegan_latent, class_label):
        if class_label < 0:
            class_label = None
        
        img = stylegan_model(stylegan_latent, class_label)
        img = torch.permute(img, (0, 2, 3, 1)) # BCHW -> BHWC
        img = torch.clip(img / 2 + 0.5, 0, 1)  # [-1, 1] -> [0, 1]
        
        return (img, )

class BlendStyleGANLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_1": ("STYLEGAN_LATENT", ),
                "latent_2": ("STYLEGAN_LATENT", ),
                "blend": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.01}),
                "mode": (["lerp", "slerp"],),
            },
        }
    
    RETURN_TYPES = ("STYLEGAN_LATENT",)
    FUNCTION = "generate_latent"
    CATEGORY = "StyleGAN/extra"
    
    def generate_latent(self, latent_1, latent_2, blend, mode):
        if latent_1.shape != latent_2.shape:
            raise Exception(f"latent_1 shape {latent_1.shape} and latent_2 shape {latent_2.shape} do not match!")
        
        if mode == "slerp":
            z = slerp(latent_1, latent_2, blend)
        else:
            z = torch.lerp(latent_1, latent_2, blend)
        
        return (z, )

class BatchAverageStyleGANLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stylegan_latent": ("STYLEGAN_LATENT", ),
            },
        }
    
    RETURN_TYPES = ("STYLEGAN_LATENT",)
    FUNCTION = "generate_latent"
    CATEGORY = "StyleGAN/extra"
    
    def generate_latent(self, stylegan_latent):
        z = torch.mean(stylegan_latent, dim=0, keepdim=True)
        std, mean = torch.std_mean(z)
        z = (z - mean) / std
        
        return (z, )

class TweakStyleGANLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stylegan_latent": ("STYLEGAN_LATENT", ),
                "index_1": ("INT", {"default": 0, "min": 0, "max": 511}),
                "offset_1": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "index_2": ("INT", {"default": 1, "min": 0, "max": 511}),
                "offset_2": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "index_3": ("INT", {"default": 2, "min": 0, "max": 511}),
                "offset_3": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "index_4": ("INT", {"default": 3, "min": 0, "max": 511}),
                "offset_4": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
        }
    
    RETURN_TYPES = ("STYLEGAN_LATENT",)
    FUNCTION = "generate_latent"
    CATEGORY = "StyleGAN/extra"
    
    def generate_latent(
        self,
        stylegan_latent,
        index_1, offset_1,
        index_2, offset_2,
        index_3, offset_3,
        index_4, offset_4,
        ):
        
        z = stylegan_latent.detach().clone()
        
        indices = [index_1, index_2, index_3, index_4]
        offsets = [offset_1, offset_2, offset_3, offset_4]
        
        for i in range(4):
            z[:, indices[i]] += offsets[i]
        
        return (z, )

class StyleGANLatentFromBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stylegan_latent": ("STYLEGAN_LATENT", ),
                "index": ("INT", {"default": 0, "min": 0}),
            },
        }
    
    RETURN_TYPES = ("STYLEGAN_LATENT",)
    FUNCTION = "generate_latent"
    CATEGORY = "StyleGAN/extra"
    
    def generate_latent(self, stylegan_latent, index):
        clipped_index = min(index, stylegan_latent.size(0) - 1)
        z = stylegan_latent[clipped_index].unsqueeze(0).detach().clone()
        
        return (z, )

NODE_CLASS_MAPPINGS = {
    "LoadStyleGAN": LoadStyleGAN,
    "GenerateStyleGANLatent": GenerateStyleGANLatent,
    "StyleGANSampler": StyleGANSampler,
    "BlendStyleGANLatents": BlendStyleGANLatents,
    "BatchAverageStyleGANLatents": BatchAverageStyleGANLatents,
    "TweakStyleGANLatents": TweakStyleGANLatents,
    "StyleGANLatentFromBatch": StyleGANLatentFromBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadStyleGAN": "Load StyleGAN Model",
    "GenerateStyleGANLatent": "Generate StyleGAN Latent",
    "StyleGANSampler": "StyleGAN Sampler",
    "BlendStyleGANLatents": "Blend StyleGAN Latents (lerp or slerp)",
    "BatchAverageStyleGANLatents": "Batch Average StyleGAN Latents",
    "TweakStyleGANLatents": "Tweak StyleGAN Latents",
    "StyleGANLatentFromBatch": "StyleGAN Latent From Batch",
}
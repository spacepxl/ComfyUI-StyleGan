import os
import sys
import numpy as np
import pickle
import torch

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
            },
        }
    
    RETURN_TYPES = ("STYLEGAN_LATENT",)
    FUNCTION = "generate_latent"
    CATEGORY = "StyleGAN"
    
    def generate_latent(self, stylegan_model, seed):
        torch.manual_seed(seed)
        z = torch.randn([1, stylegan_model.z_dim]).cuda()
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

NODE_CLASS_MAPPINGS = {
    "LoadStyleGAN": LoadStyleGAN,
    "GenerateStyleGANLatent": GenerateStyleGANLatent,
    "StyleGANSampler": StyleGANSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadStyleGAN": "Load StyleGAN Model",
    "GenerateStyleGANLatent": "Generate StyleGAN Latent",
    "StyleGANSampler": "StyleGAN Sampler",
}
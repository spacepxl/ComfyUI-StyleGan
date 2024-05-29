# Modified from https://github.com/ouhenio/stylegan3-projector/blob/main/projector.py

import os
import io
import copy
import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from . import dnnlib

import folder_paths
from comfy.utils import PROGRESS_BAR_ENABLED, ProgressBar

def load_vgg(device):
    # set the models directory
    if "VGG" not in folder_paths.folder_names_and_paths:
        current_paths = [os.path.join(folder_paths.models_dir, "VGG")]
        if not os.path.exists(current_paths[0]):
            os.mkdir(current_paths[0])
    else:
        current_paths, _ = folder_paths.folder_names_and_paths["VGG"]
    folder_paths.folder_names_and_paths["VGG"] = (current_paths, folder_paths.supported_pt_extensions)
    
    vgg_file = None
    if "vgg16.pt" in folder_paths.get_filename_list("VGG"):
        vgg_file = folder_paths.get_full_path("VGG", "vgg16.pt")
    
    if vgg_file is not None:
        with open(vgg_file, 'rb') as fv:
            vgg16 = torch.jit.load(fv).eval().to(device)
    else:
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        print("downloading VGG16")
        with dnnlib.util.open_url(url=url, cache=False) as fd:
            filename = os.path.join(current_paths[0], "vgg16.pt")
            with open(filename, "wb") as fv:
                print(f"saving VGG16 to {filename}")
                fv.write(fd.getvalue())
            vgg16 = torch.jit.load(fd).eval().to(device)
    
    return vgg16

# @torch.enable_grad()
@torch.inference_mode(mode=False)
def project(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    seed                       = 0,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    print(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    np.random.seed(seed)
    torch.manual_seed(seed)
    z_samples = np.random.RandomState(seed).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    vgg16 = load_vgg(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    pbar = None
    if PROGRESS_BAR_ENABLED and num_steps > 1:
        pbar = ProgressBar(num_steps)
    tq_bar = tqdm.trange(num_steps, desc="Projecting")
    for step in tq_bar:
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight
        # loss.requires_grad = True

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        tq_bar.set_postfix_str(f"dist: {dist:.2f} total: {float(loss):<5.2f}")

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()
        
        if pbar is not None:
            pbar.update(1)

    return w_out.repeat([1, G.mapping.num_ws, 1])
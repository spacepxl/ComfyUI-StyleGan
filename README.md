# ComfyUI-StyleGan

Basic support for StyleGAN2 and StyleGAN3 models.  
![workflow](https://github.com/spacepxl/ComfyUI-StyleGan/assets/143970342/f9aaf427-f66d-4651-82aa-3c7b3ede6e26)

Original:  
https://github.com/NVlabs/stylegan3

Models:  
https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan2/files  
https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan3/files  

Place any models you want to use in `ComfyUI/models/stylegan/*.pkl` (create the folder if it doesn't exist)

## Installation

StyleGAN uses custom CUDA extensions which are compiled at runtime, so unfortunately the setup process can be a bit of a pain.

You need CUDA Toolkit, ninja, and either GCC (Linux) or Visual Studio (Windows). Tested on Windows with CUDA Toolkit 11.7 and VS2019 Community. You may also need to add paths to the system PATH, CUDA_HOME, and LD_LIBRARY_PATH.

```
PATH:
C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64
C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build

CUDA_HOME:
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7

LD_LIBRARY_PATH:
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\lib\x64
```

If you're using ComfyUI portable, the embedded python installation is probably also missing some necessary files. The only solution I found to this was to just copy them from a full system installation of python 3.10.x to the embedded installation.

From `C:/Users/username/AppData/Local/Programs/Python/Python310/include/*`  
to `ComfyUI_windows_portable/python_embeded/Include/*`  
(make sure you don't overwrite any file/folders that are already there)

And from `C:/Users/username/AppData/Local/Programs/Python/Python310/libs/*`  
to `ComfyUI_windows_portable/python_embeded/libs/*`

If all of that is set up correctly, when you run a StyleGAN workflow, it will first build the necessary PyTorch plugins (should take 30-60s), then generate an image. There will be a message in the console, and then subsequent images will be much faster to generate, about 0.1-0.3s on a 3090.

StyleGAN2:  
```
Setting up PyTorch plugin "bias_act_plugin"... Done.
Setting up PyTorch plugin "upfirdn2d_plugin"... Done.
```
StyleGAN3:  
```
Setting up PyTorch plugin "bias_act_plugin"... Done.
Setting up PyTorch plugin "filtered_lrelu_plugin"... Done.
```  

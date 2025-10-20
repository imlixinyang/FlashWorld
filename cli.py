try:
    import spaces
    GPU = spaces.GPU
    mode = "Spaces"
except ImportError:
    def GPU(func):
        return func
    mode = "Local"

import os
import subprocess

import tqdm

try:
    import gsplat
except ImportError:
    def install_cuda_toolkit():
        # CUDA_TOOLKIT_URL = "https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run"
        CUDA_TOOLKIT_URL = "https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run"
        CUDA_TOOLKIT_FILE = "/tmp/%s" % os.path.basename(CUDA_TOOLKIT_URL)
        subprocess.call(["wget", "-q", CUDA_TOOLKIT_URL, "-O", CUDA_TOOLKIT_FILE])
        subprocess.call(["chmod", "+x", CUDA_TOOLKIT_FILE])
        subprocess.call([CUDA_TOOLKIT_FILE, "--silent", "--toolkit"])
        
        os.environ["CUDA_HOME"] = "/usr/local/cuda"
        os.environ["PATH"] = "%s/bin:%s" % (os.environ["CUDA_HOME"], os.environ["PATH"])
        os.environ["LD_LIBRARY_PATH"] = "%s/lib:%s" % (
            os.environ["CUDA_HOME"],
            "" if "LD_LIBRARY_PATH" not in os.environ else os.environ["LD_LIBRARY_PATH"],
        )
        # Fix: arch_list[-1] += '+PTX'; IndexError: list index out of range
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0+PTX"

        print("Successfully installed CUDA toolkit at: ", os.environ["CUDA_HOME"])

        subprocess.call('rm /usr/bin/gcc', shell=True)
        subprocess.call('rm /usr/bin/g++', shell=True)

        subprocess.call('ln -s /usr/bin/gcc-11 /usr/bin/gcc', shell=True)
        subprocess.call('ln -s /usr/bin/g++-11 /usr/bin/g++', shell=True)

        subprocess.call('gcc --version', shell=True)
        subprocess.call('g++ --version', shell=True)

    if mode == "Spaces":
        install_cuda_toolkit()

        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0+PTX"
        os.environ["CUDA_HOME"] = "/usr/local/cuda"
        os.environ["PATH"] = "/usr/local/cuda/bin/:" + os.environ["PATH"]

        subprocess.run('pip install git+https://github.com/nerfstudio-project/gsplat.git@32f2a54d21c7ecb135320bb02b136b7407ae5712', 
            env={'CUDA_HOME': "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "9.0+PTX", "PATH": "/usr/local/cuda/bin/:" + os.environ["PATH"]}, shell=True)
    
    else:
        print("Gsplat is not installed.")
        exit()
        
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import gradio as gr
import base64
import io
from PIL import Image
import torch
import numpy as np
import os
import argparse
import imageio
import json
import time
import tempfile
import shutil
import threading

from huggingface_hub import hf_hub_download

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import imageio

from models import *
from utils import *

from transformers import T5TokenizerFast, UMT5EncoderModel

from diffusers import FlowMatchEulerDiscreteScheduler

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.enable_flash_sdp(True)

from app import GenerationSystem

@GPU
def process_generation_request(data, generation_system, out_dir, video=False, spz=False, ply=False, video_fps=15, video_path=None):
    image_prompt = data.get('image_prompt', None)
    text_prompt = data.get('text_prompt', "")
    cameras = data.get('cameras')
    resolution = data.get('resolution')
    image_index = data.get('image_index', 0)

    n_frame, image_height, image_width = resolution

    if not image_prompt and text_prompt == "":
        return {'error': 'No Prompts provided'}

    if image_prompt:
        # image_prompt可以是路径和base64
        if os.path.exists(image_prompt):
            image_prompt = Image.open(image_prompt)
        else:
            # image_prompt 可能是 "data:image/png;base64,...."
            if ',' in image_prompt:
                image_prompt = image_prompt.split(',', 1)[1]
            
            try:
                image_bytes = base64.b64decode(image_prompt)
                image_prompt = Image.open(io.BytesIO(image_bytes))
            except Exception as img_e:
                return {'error': f'Image decode error: {str(img_e)}'}

        image = image_prompt.convert('RGB')

        w, h = image.size

        # center crop
        if image_height / h > image_width / w:
            scale = image_height / h
        else:
            scale = image_width / w
            
        new_h = int(image_height / scale)
        new_w = int(image_width / scale)

        image = image.crop(((w - new_w) // 2, (h - new_h) // 2, 
                            new_w + (w - new_w) // 2, new_h + (h - new_h) // 2)).resize((image_width, image_height))

        for camera in cameras:
            camera['fx'] = camera['fx'] * scale 
            camera['fy'] = camera['fy'] * scale 
            camera['cx'] = (camera['cx'] - (w - new_w) // 2) * scale
            camera['cy'] = (camera['cy'] - (h - new_h) // 2) * scale

        image = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0 * 2 - 1
    else:
        image = None

    cameras = torch.stack([
        torch.from_numpy(np.array([camera['quaternion'][0], camera['quaternion'][1], camera['quaternion'][2], camera['quaternion'][3], camera['position'][0], camera['position'][1], camera['position'][2], camera['fx'] / image_width, camera['fy'] / image_height, camera['cx'] / image_width, camera['cy'] / image_height], dtype=np.float32))
        for camera in cameras
    ], dim=0)

    start_time = time.time()
    scene_params, ref_w2c, T_norm = generation_system.generate(cameras, n_frame, image, text_prompt, image_index, image_height, image_width, video_path=os.path.join(out_dir, 'video.mp4') if video else None)
    end_time = time.time()

    scene_params = scene_params.detach().cpu()

    export_gaussians(scene_params, 
                    opacity_threshold=0.000, 
                    T_norm=T_norm, 
                    ply_path=os.path.join(out_dir, 'gaussians.ply') if ply else None, 
                    spz_path=os.path.join(out_dir, 'gaussians.spz') if spz else None)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument("--ckpt", default=None)

    parser.add_argument("--offload_t5", action="store_true")
    parser.add_argument("--offload_vae", action="store_true")
    parser.add_argument("--offload_transformer_during_vae", action="store_true")

    parser.add_argument("--input_dir", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default=None, required=True)

    parser.add_argument("--video", action="store_true")
    parser.add_argument("--spz", action="store_true")
    parser.add_argument("--ply", action="store_true")

    parser.add_argument('--video_fps', type=int, default=15)
    args = parser.parse_args()

    # Ensure model.ckpt exists, download if not present
    if args.ckpt is None:
        from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
        ckpt_path = os.path.join(HUGGINGFACE_HUB_CACHE, "models--imlixinyang--FlashWorld", "snapshots", "6a8e88c6f88678ac098e4c82675f0aee555d6e5d", "model.ckpt")
        if not os.path.exists(ckpt_path):
            hf_hub_download(repo_id="imlixinyang/FlashWorld", filename="model.ckpt", local_dir_use_symlinks=False)
    else:
        ckpt_path = args.ckpt

    # Initialize GenerationSystem
    device = torch.device("cuda")
    generation_system = GenerationSystem(ckpt_path=ckpt_path, device=device, offload_t5=args.offload_t5, offload_vae=args.offload_vae, offload_transformer_during_vae=args.offload_transformer_during_vae)

    print("GenerationSystem initialized!")

    for json_file in tqdm.tqdm(sorted(os.listdir(args.input_dir))):
        if json_file.endswith('.json'):
            json_path = os.path.join(args.input_dir, json_file)
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            out_dir = os.path.join(args.output_dir, json_file.replace('.json', ''))
            os.makedirs(out_dir, exist_ok=True)
            result = process_generation_request(json_data, generation_system, out_dir, video=args.video, spz=args.spz, ply=args.ply, video_fps=args.video_fps)
            # print(f'{json_file} generated successfully')
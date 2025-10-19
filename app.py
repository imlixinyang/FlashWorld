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

class MyFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        return torch.argmin(
            (timestep - schedule_timesteps.to(timestep.device)).abs(), dim=0).item()

class GenerationSystem(nn.Module):
    def __init__(self, ckpt_path=None, device="cuda:0", offload_t5=False, offload_vae=False):
        super().__init__()
        self.device = device
        self.offload_t5 = offload_t5
        self.offload_vae = offload_vae

        self.latent_dim = 48
        self.temporal_downsample_factor = 4
        self.spatial_downsample_factor = 16

        self.feat_dim = 1024

        self.latent_patch_size = 2

        self.denoising_steps = [0, 250, 500, 750]

        model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

        self.vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float).eval()

        from models.autoencoder_kl_wan import WanCausalConv3d
        with torch.no_grad():
            for name, module in self.vae.named_modules():
                if isinstance(module, WanCausalConv3d):
                    time_pad = module._padding[4]
                    module.padding = (0, module._padding[2], module._padding[0])
                    module._padding = (0, 0, 0, 0, 0, 0)
                    module.weight = torch.nn.Parameter(module.weight[:, :, time_pad:].clone())

        self.vae.requires_grad_(False)

        self.register_buffer('latents_mean', torch.tensor(self.vae.config.latents_mean).float().view(1, self.vae.config.z_dim, 1, 1, 1).to(self.device))
        self.register_buffer('latents_std', torch.tensor(self.vae.config.latents_std).float().view(1, self.vae.config.z_dim, 1, 1, 1).to(self.device))

        self.tokenizer = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer")

        self.text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float32).eval().requires_grad_(False).to(self.device if not self.offload_t5 else "cpu")

        self.transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.float32).train().requires_grad_(False)
        
        self.transformer.patch_embedding.weight = nn.Parameter(F.pad(self.transformer.patch_embedding.weight, (0, 0, 0, 0, 0, 0, 0, 6 + self.latent_dim)))
        # self.transformer.rope.freqs_f[:] = self.transformer.rope.freqs_f[:1]

        weight = self.transformer.proj_out.weight.reshape(self.latent_patch_size ** 2, self.latent_dim, self.transformer.proj_out.weight.shape[1])
        bias = self.transformer.proj_out.bias.reshape(self.latent_patch_size ** 2, self.latent_dim)

        extra_weight = torch.randn(self.latent_patch_size ** 2, self.feat_dim, self.transformer.proj_out.weight.shape[1]) * 0.02
        extra_bias = torch.zeros(self.latent_patch_size ** 2, self.feat_dim)
 
        self.transformer.proj_out.weight = nn.Parameter(torch.cat([weight, extra_weight], dim=1).flatten(0, 1).detach().clone())
        self.transformer.proj_out.bias = nn.Parameter(torch.cat([bias, extra_bias], dim=1).flatten(0, 1).detach().clone())

        self.recon_decoder = WANDecoderPixelAligned3DGSReconstructionModel(self.vae, self.feat_dim, use_render_checkpointing=True, use_network_checkpointing=False).train().requires_grad_(False)

        self.scheduler = MyFlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", shift=3)

        self.register_buffer('timesteps', self.scheduler.timesteps.clone().to(self.device))

        self.transformer.disable_gradient_checkpointing()
        self.transformer.gradient_checkpointing = False

        self.add_feedback_for_transformer()

        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.transformer.load_state_dict(state_dict["transformer"])
            self.recon_decoder.load_state_dict(state_dict["recon_decoder"])
            print(f"Loaded {ckpt_path}.")

        from quant import FluxFp8GeMMProcessor

        FluxFp8GeMMProcessor(self.transformer)

        del self.vae.post_quant_conv, self.vae.decoder
        self.vae.to(self.device if not self.offload_vae else "cpu").to(torch.bfloat16)
        self.recon_decoder.to(self.device if not self.offload_vae else "cpu").to(torch.bfloat16)

        self.transformer.to(self.device)

    def latent_scale_fn(self, x):
        return (x - self.latents_mean) / self.latents_std

    def latent_unscale_fn(self, x):
        return x * self.latents_std + self.latents_mean

    def add_feedback_for_transformer(self):
        self.use_feedback = True
        self.transformer.patch_embedding.weight = nn.Parameter(F.pad(self.transformer.patch_embedding.weight, (0, 0, 0, 0, 0, 0, 0, self.feat_dim + self.latent_dim)))
    
    def encode_text(self, texts):
        max_sequence_length = 512

        text_inputs = self.tokenizer(
            texts,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        if getattr(self, "offload_t5", False):
            text_input_ids = text_inputs.input_ids.to("cpu")
            mask = text_inputs.attention_mask.to("cpu")
        else:
            text_input_ids = text_inputs.input_ids.to(self.device)
            mask = text_inputs.attention_mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()

        if getattr(self, "offload_t5", False):
            with torch.no_grad():
                text_embeds = self.text_encoder(text_input_ids, mask).last_hidden_state.to(self.device)
        else:
            text_embeds = self.text_encoder(text_input_ids, mask).last_hidden_state
        text_embeds = [u[:v] for u, v in zip(text_embeds, seq_lens)]
        text_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in text_embeds], dim=0
        )
        return text_embeds.float()

    def forward_generator(self, noisy_latents, raymaps, condition_latents, t, text_embeds, cameras, render_cameras, image_height, image_width, need_3d_mode=True):

        out = self.transformer(
            hidden_states=torch.cat([noisy_latents, raymaps, condition_latents], dim=1),
            timestep=t,
            encoder_hidden_states=text_embeds,
            return_dict=False,
        )[0]

        v_pred, feats = out.split([self.latent_dim, self.feat_dim], dim=1)
               
        sigma = torch.stack([self.scheduler.sigmas[self.scheduler.index_for_timestep(_t)] for _t in t.unbind(0)], dim=0).to(self.device)
        latents_pred_2d = noisy_latents - sigma * v_pred

        if need_3d_mode:
            scene_params = self.recon_decoder(
                                einops.rearrange(feats, 'B C T H W -> (B T) C H W').unsqueeze(2).to(self.device if not self.offload_vae else "cpu").to(torch.bfloat16), 
                                einops.rearrange(self.latent_unscale_fn(latents_pred_2d.detach()), 'B C T H W -> (B T) C H W').unsqueeze(2).to(self.device if not self.offload_vae else "cpu").to(torch.bfloat16), 
                                cameras.to(self.device if not self.offload_vae else "cpu").float()
                            ).flatten(1, -2).to(self.device).float()

            images_pred, _ = self.recon_decoder.render(scene_params.unbind(0), render_cameras, image_height, image_width, bg_mode="white")

            latents_pred_3d = einops.rearrange(self.latent_scale_fn(self.vae.encode(
                            einops.rearrange(images_pred, 'B T C H W -> (B T) C H W', T=images_pred.shape[1]).unsqueeze(2).to(self.device if not self.offload_vae else "cpu").to(torch.bfloat16), 
                        ).latent_dist.sample().to(self.device)).squeeze(2), '(B T) C H W -> B C T H W', T=images_pred.shape[1]).to(noisy_latents.dtype)

        return {
            '2d': latents_pred_2d,
            '3d': latents_pred_3d if need_3d_mode else None,
            'rgb_3d': images_pred if need_3d_mode else None,
            'scene': scene_params if need_3d_mode else None,
            'feat': feats
        }

    @torch.no_grad()
    def generate(self, cameras, n_frame, image=None, text="", image_index=0, image_height=480, image_width=704, video_path=None, video_fps=15):  

        if mode == "Spaces":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
            self.vae.to(self.device)
            self.text_encoder.to(self.device if not self.offload_t5 else "cpu")
            self.transformer.to(self.device)
            self.recon_decoder.to(self.device)
            self.timesteps = self.timesteps.to(self.device)
            self.latents_mean = self.latents_mean.to(self.device)
            self.latents_std = self.latents_std.to(self.device)

        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
            batch_size = 1
            
            cameras = cameras.to(self.device).unsqueeze(0)

            if cameras.shape[1] != n_frame:
                cameras = sample_from_dense_cameras(cameras.squeeze(0), torch.linspace(0, 1, n_frame, device=self.device)).unsqueeze(0)

            if video_path is not None:
                render_cameras = sample_from_dense_cameras(cameras.squeeze(0), torch.linspace(0, 1, (n_frame - 1) * video_fps + 1, device=self.device)).unsqueeze(0)
            else:
                render_cameras = None
            
            cameras, ref_w2c, T_norm = normalize_cameras(cameras, return_meta=True, n_frame=None)

            render_cameras = normalize_cameras(render_cameras, ref_w2c=ref_w2c, T_norm=T_norm, n_frame=None) if render_cameras is not None else None

            text = "[Static] " + text

            text_embeds = self.encode_text([text])
            # neg_text_embeds = self.encode_text([""]).repeat(batch_size, 1, 1)

            masks = torch.zeros(batch_size, n_frame, device=self.device)

            condition_latents = torch.zeros(batch_size, self.latent_dim, n_frame, image_height // self.spatial_downsample_factor, image_width // self.spatial_downsample_factor, device=self.device)

            if image is not None:
                image = image.to(self.device)

                latent = self.latent_scale_fn(self.vae.encode(
                        image.unsqueeze(0).unsqueeze(2).to(self.device if not self.offload_vae else "cpu").to(torch.bfloat16)
                    ).latent_dist.sample().to(self.device)).squeeze(2)

                masks[:, image_index] = 1
                condition_latents[:, :, image_index] = latent

            raymaps = create_raymaps(cameras, image_height // self.spatial_downsample_factor, image_width // self.spatial_downsample_factor)
            raymaps = einops.rearrange(raymaps, 'B T H W C -> B C T H W', T=n_frame)
            
            noise = torch.randn(batch_size, self.latent_dim, n_frame, image_height // self.spatial_downsample_factor, image_width // self.spatial_downsample_factor, device=self.device)

            noisy_latents = noise 

            if self.use_feedback:
                prev_latents_pred = torch.zeros(batch_size, self.latent_dim, n_frame, image_height // self.spatial_downsample_factor, image_width // self.spatial_downsample_factor, device=self.device)

                prev_feats = torch.zeros(batch_size, self.feat_dim, n_frame, image_height // self.spatial_downsample_factor, image_width // self.spatial_downsample_factor, device=self.device)

            for i in range(len(self.denoising_steps)):
                t_ids = torch.full((noisy_latents.shape[0],), self.denoising_steps[i], device=self.device)

                t = self.timesteps[t_ids]

                if self.use_feedback:
                    _condition_latents = torch.cat([condition_latents, prev_feats, prev_latents_pred], dim=1)
                else:
                    _condition_latents = condition_latents

                if i < len(self.denoising_steps) - 1:
                    out = self.forward_generator(noisy_latents, raymaps, _condition_latents, t, text_embeds, cameras, cameras, image_height, image_width, need_3d_mode=True)

                    latents_pred = out["3d"]

                    if self.use_feedback:
                        prev_latents_pred = latents_pred
                        prev_feats = out['feat']
                   
                    noisy_latents = self.scheduler.scale_noise(latents_pred, self.timesteps[torch.full((noisy_latents.shape[0],), self.denoising_steps[i + 1], device=self.device)], torch.randn_like(noise))
                    
                else:
                    out = self.transformer(
                        hidden_states=torch.cat([noisy_latents, raymaps, _condition_latents], dim=1),
                        timestep=t,
                        encoder_hidden_states=text_embeds,
                        return_dict=False,
                    )[0]

                    v_pred, feats = out.split([self.latent_dim, self.feat_dim], dim=1)
                        
                    sigma = torch.stack([self.scheduler.sigmas[self.scheduler.index_for_timestep(_t)] for _t in t.unbind(0)], dim=0).to(self.device)
                    latents_pred = noisy_latents - sigma * v_pred

                    scene_params = self.recon_decoder(
                                        einops.rearrange(feats, 'B C T H W -> (B T) C H W').unsqueeze(2).to(self.device if not self.offload_vae else "cpu").to(torch.bfloat16), 
                                        einops.rearrange(self.latent_unscale_fn(latents_pred.detach()), 'B C T H W -> (B T) C H W').unsqueeze(2).to(self.device if not self.offload_vae else "cpu").to(torch.bfloat16), 
                                        cameras.to(self.device if not self.offload_vae else "cpu").float()
                                    ).flatten(1, -2).to(self.device).float()

            if video_path is not None:
                interpolated_images_pred, _ = self.recon_decoder.render(scene_params.unbind(0), render_cameras, image_height, image_width, bg_mode="white")

                interpolated_images_pred = einops.rearrange(interpolated_images_pred[0].clamp(-1, 1).add(1).div(2), 'T C H W -> T H W C')

                interpolated_images_pred = [torch.cat([img], dim=1).detach().cpu().mul(255).numpy().astype(np.uint8) for i, img in enumerate(interpolated_images_pred.unbind(0))]

                imageio.mimwrite(video_path, interpolated_images_pred, fps=video_fps, quality=8, macro_block_size=1) 

        scene_params = scene_params[0]

        return scene_params, ref_w2c, T_norm

@GPU
def process_generation_request(data, generation_system, cache_dir):
    """
    Process the generation request with the same logic as Flask version
    """
    try:
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

        file_id = str(int(time.time() * 1000))

        start_time = time.time()
        scene_params, ref_w2c, T_norm = generation_system.generate(cameras, n_frame, image, text_prompt, image_index, image_height, image_width, video_output_path=os.path.join(cache_dir, f'{file_id}.mp4'))
        end_time = time.time()
        print(f'生成时间: {end_time - start_time} 秒')

        scene_params = scene_params.detach().cpu()

        with open(os.path.join(cache_dir, f'{file_id}.json'), 'w') as f:
            json.dump(data, f)

        splat_path = os.path.join(cache_dir, f'{file_id}.spz')

        export_gaussians(scene_params, opacity_threshold=0.00025, T_norm=T_norm, spz_path=splat_path)

        if not os.path.exists(splat_path):
            return {'error': f'{splat_path} not found'}

        file_size = os.path.getsize(splat_path)
        
        response_data = {
            'success': True,
            'file_id': file_id,
            'file_path': splat_path,
            'file_size': file_size,
            'download_url': f'/download/{file_id}',
            'generation_time': end_time - start_time,
        }
        
        return response_data

    except Exception as e:
        print(f'Processing error: {str(e)}')
        return {'error': f'Processing error: {str(e)}'}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--offload_t5", action="store_true")
    parser.add_argument("--offload_vae", action="store_true")
    args = parser.parse_args()

    # Ensure model.ckpt exists, download if not present
    if args.ckpt is None:
        from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
        ckpt_path = os.path.join(HUGGINGFACE_HUB_CACHE, "models--imlixinyang--FlashWorld", "snapshots", "6a8e88c6f88678ac098e4c82675f0aee555d6e5d", "model.ckpt")
        if not os.path.exists(ckpt_path):
            hf_hub_download(repo_id="imlixinyang/FlashWorld", filename="model.ckpt", local_dir_use_symlinks=False)
    else:
        ckpt_path = args.ckpt

    if args.cache_dir is None or args.cache_dir == "":
        GRADIO_TEMP_DIR = tempfile.gettempdir()
        cache_dir = os.path.join(GRADIO_TEMP_DIR, "flashworld_gradio")
    else:
        cache_dir = args.cache_dir

    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)

    # Initialize GenerationSystem
    device = torch.device("cpu") if mode == "Spaces" else torch.device("cuda")
    generation_system = GenerationSystem(ckpt_path=ckpt_path, device=device, offload_t5=args.offload_t5, offload_vae=args.offload_vae)

    print("GenerationSystem initialized!")

    # Create Gradio interface
    with gr.Blocks(title="FlashWorld Backend") as demo:
        gr.Markdown("FlashWorld Generation Backend — API only. This service powers the FlashWorld Web Demo and is intended for programmatic/API access. The UI is intentionally hidden.")
        
        with gr.Row(visible=False):
            with gr.Column():
                json_input = gr.Textbox(
                    label="JSON Input",
                    placeholder="Enter JSON request here...",
                    lines=10,
                    value='{"image_prompt": null, "text_prompt": "A beautiful landscape", "cameras": [...], "resolution": [16, 480, 704], "image_index": 0}'
                )
                
                generate_btn = gr.Button("Generate", variant="primary")
                
            with gr.Column():
                json_output = gr.Textbox(
                    label="JSON Output",
                    lines=10,
                    interactive=False
                )
        
        # File download section
        gr.Markdown("## File Download", visible=False)
        with gr.Row(visible=False):
            file_id_input = gr.Textbox(
                label="File ID",
                placeholder="Enter file ID to download..."
            )
            download_btn = gr.Button("Download SPZ File")
            download_output = gr.File(label="Downloaded File")

        
        def gradio_generate(json_input):
            """
            Gradio interface function that processes JSON input and returns JSON output
            """
            try:
                # Parse JSON input
                if isinstance(json_input, str):
                    data = json.loads(json_input)
                else:
                    data = json_input
                    
                # Process the request
                result = process_generation_request(data, generation_system, cache_dir)
                
                # Return JSON response
                return json.dumps(result, indent=2)
                
            except Exception as e:
                error_response = {'error': f'JSON processing error: {str(e)}'}
                return json.dumps(error_response, indent=2)

        def download_file(file_id):
            """
            Download generated SPZ file
            """
            file_path = os.path.join(cache_dir, f'{file_id}.spz')
            
            if not os.path.exists(file_path):
                return None
            
            return file_path

        def gradio_delete(file_id):
            """
            Delete generated artifacts by file_id (.spz/.json/.mp4)
            """
            deleted = False
            try:
                for ext in (".spz", ".json", ".mp4"):
                    p = os.path.join(cache_dir, f"{file_id}{ext}")
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                            deleted = True
                        except Exception:
                            pass
                return {"success": True, "deleted": deleted}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        # Event handlers
        generate_btn.click(
            fn=gradio_generate,
            inputs=[json_input],
            outputs=[json_output]
        )
        
        download_btn.click(
            fn=download_file,
            inputs=[file_id_input],
            outputs=[download_output]
        )

        # Hidden API hook for deletion to expose /gradio_api/call/gradio_delete
        _hidden_delete_in = gr.Textbox(visible=False)
        _hidden_delete_btn = gr.Button(visible=False)
        _hidden_delete_btn.click(fn=gradio_delete, inputs=[_hidden_delete_in], outputs=[])
        
        # Example JSON format
        gr.Markdown("""
        ## Example JSON Input Format:
        ```json
        {
            "image_prompt": null,
            "text_prompt": "A beautiful landscape with mountains and trees",
            "cameras": [
                {
                    "quaternion": [0, 0, 0, 1],
                    "position": [0, 0, 5],
                    "fx": 500,
                    "fy": 500,
                    "cx": 240,
                    "cy": 240
                },
                {
                    "quaternion": [0, 0, 0, 1],
                    "position": [0, 0, 5],
                    "fx": 500,
                    "fy": 500,
                    "cx": 240,
                    "cy": 240
                }
            ],
            "resolution": [16, 480, 704],
            "image_index": 0
        }
        ```
        """, visible=False)

    # Background periodic cleanup thread (no FastAPI app lifecycle)
    def _cleanup_loop(directory: str, max_age_seconds: int = 15 * 60, interval_seconds: int = 300):
        while True:
            try:
                now = time.time()
                for name in os.listdir(directory):
                    path = os.path.join(directory, name)
                    try:
                        if os.path.isfile(path):
                            mtime = os.path.getmtime(path)
                            if (now - mtime) > max_age_seconds:
                                try:
                                    os.remove(path)
                                except Exception:
                                    pass
                    except Exception:
                        pass
            except Exception:
                pass
            time.sleep(interval_seconds)

    threading.Thread(target=_cleanup_loop, args=(cache_dir,), daemon=True).start()

    if mode == "Spaces":
        demo.launch(allowed_paths=[cache_dir])
    else:

        app = FastAPI()

        # 在/app端口下用fastapi直接返回index.html文件
        @app.get("/app")
        def index():
            return FileResponse("index.html")

        app = gr.mount_gradio_app(app, demo, path="/")

        app.mount("/examples", StaticFiles(directory=os.path.abspath("./examples")), name="examples")

        uvicorn.run(app, port=args.port)

    

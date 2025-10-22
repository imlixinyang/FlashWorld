#!/usr/bin/env python3
"""
Low memory FlashWorld generation test script
Usage: python test_generate_minimal.py --trajectory cameras.json --text "a beautiful landscape" --output output.ply
"""

import os
import sys

# This allows importing models, utils, etc. when running from examples/minimal/
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, root_dir)

import torch
import numpy as np
import argparse
import json
from PIL import Image
import gc

from models import *
from utils import *
from transformers import T5TokenizerFast, UMT5EncoderModel
from diffusers import FlowMatchEulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
import einops
import torch.nn as nn
import torch.nn.functional as F

class MyFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        return torch.argmin(
            (timestep - schedule_timesteps.to(timestep.device)).abs(), dim=0).item()

class GenerationSystem(nn.Module):
    def __init__(self, ckpt_path=None, device="cuda:0", offload_t5=True, offload_vae=True, 
                 reduce_frames=True, use_fewer_steps=False):
        super().__init__()
        self.device = device
        self.offload_t5 = offload_t5
        self.offload_vae = offload_vae
        self.reduce_frames = reduce_frames
        
        self.latent_dim = 48
        self.temporal_downsample_factor = 4
        self.spatial_downsample_factor = 16
        self.feat_dim = 1024
        self.latent_patch_size = 2
        
        # Option to use fewer denoising steps for faster/lower memory
        if use_fewer_steps:
            self.denoising_steps = [0, 500]  # Just 2 steps instead of 4
            print("Using 2 denoising steps (faster, uses less memory)")
        else:
            self.denoising_steps = [0, 250, 500, 750]
            print("Using 4 denoising steps (full quality)")

        print("Loading VAE...")
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

        self.latent_scale_fn = lambda x: (x - self.latents_mean) / self.latents_std
        self.latent_unscale_fn = lambda x: x * self.latents_std + self.latents_mean

        print("Loading text encoder...")
        self.tokenizer = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer")
        # Offload T5 to CPU to save VRAM
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.float32
        ).eval().requires_grad_(False).to("cpu" if offload_t5 else self.device)

        print("Loading transformer...")
        self.transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.float32).train().requires_grad_(False)
        
        self.transformer.patch_embedding.weight = nn.Parameter(F.pad(self.transformer.patch_embedding.weight, (0, 0, 0, 0, 0, 0, 0, 6 + self.latent_dim)))

        weight = self.transformer.proj_out.weight.reshape(self.latent_patch_size ** 2, self.latent_dim, self.transformer.proj_out.weight.shape[1])
        bias = self.transformer.proj_out.bias.reshape(self.latent_patch_size ** 2, self.latent_dim)

        extra_weight = torch.randn(self.latent_patch_size ** 2, self.feat_dim, self.transformer.proj_out.weight.shape[1]) * 0.02
        extra_bias = torch.zeros(self.latent_patch_size ** 2, self.feat_dim)
 
        self.transformer.proj_out.weight = nn.Parameter(torch.cat([weight, extra_weight], dim=1).flatten(0, 1).detach().clone())
        self.transformer.proj_out.bias = nn.Parameter(torch.cat([bias, extra_bias], dim=1).flatten(0, 1).detach().clone())

        print("Loading reconstruction decoder...")
        self.recon_decoder = WANDecoderPixelAligned3DGSReconstructionModel(
            self.vae, self.feat_dim, use_render_checkpointing=True, use_network_checkpointing=False
        ).train().requires_grad_(False).to(self.device)

        self.scheduler = MyFlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", shift=3)
        self.register_buffer('timesteps', self.scheduler.timesteps.clone().to(self.device))

        self.transformer.disable_gradient_checkpointing()
        self.transformer.gradient_checkpointing = False

        self.use_feedback = True
        self.transformer.patch_embedding.weight = nn.Parameter(F.pad(self.transformer.patch_embedding.weight, (0, 0, 0, 0, 0, 0, 0, self.feat_dim + self.latent_dim)))

        if ckpt_path is not None:
            print(f"Loading checkpoint: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.transformer.load_state_dict(state_dict["transformer"])
            self.recon_decoder.load_state_dict(state_dict["recon_decoder"])
            print(f"Loaded {ckpt_path}.")

        from quant import FluxFp8GeMMProcessor
        FluxFp8GeMMProcessor(self.transformer)

        # Delete unused VAE parts to save memory
        del self.vae.post_quant_conv, self.vae.decoder
        # Offload VAE encoder to CPU to save VRAM
        self.vae.to("cpu" if offload_vae else self.device)
        
        self.transformer.to(self.device)
        
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
        
        print("Model loaded successfully!")
        print(f"T5 offloaded to CPU: {offload_t5}")
        print(f"VAE offloaded to CPU: {offload_vae}")

    def encode_text(self, texts):
        max_sequence_length = 512
        text_inputs = self.tokenizer(
            texts, padding="max_length", max_length=max_sequence_length,
            truncation=True, add_special_tokens=True, return_attention_mask=True,
            return_tensors="pt",
        )
        
        if self.offload_t5:
            text_input_ids = text_inputs.input_ids.to("cpu")
            mask = text_inputs.attention_mask.to("cpu")
        else:
            text_input_ids = text_inputs.input_ids.to(self.device)
            mask = text_inputs.attention_mask.to(self.device)
        
        seq_lens = mask.gt(0).sum(dim=1).long()

        if self.offload_t5:
            with torch.no_grad():
                text_embeds = self.text_encoder(text_input_ids, mask).last_hidden_state.to(self.device)
        else:
            text_embeds = self.text_encoder(text_input_ids, mask).last_hidden_state
            
        text_embeds = [u[:v] for u, v in zip(text_embeds, seq_lens)]
        text_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in text_embeds], dim=0
        )
        return text_embeds.float()

    @torch.no_grad()
    @torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda")
    def generate(self, cameras, n_frame, image=None, text="", image_index=0, image_height=480, image_width=704):
        # Reduce frame count if requested
        if self.reduce_frames and n_frame > 16:
            original_n_frame = n_frame
            n_frame = 16
            print(f"Reducing frame count from {original_n_frame} to {n_frame} to save memory")
        
        print(f"\nGenerating scene with {n_frame} frames...")
        print(f"Resolution: {image_height}x{image_width}")
        
        batch_size = 1
        
        cameras = cameras.to(self.device).unsqueeze(0)

        if cameras.shape[1] != n_frame:
            render_cameras = cameras.clone()
            cameras = sample_from_dense_cameras(cameras.squeeze(0), torch.linspace(0, 1, n_frame, device=self.device)).unsqueeze(0)
        else:
            render_cameras = cameras
        
        cameras, ref_w2c, T_norm = normalize_cameras(cameras, return_meta=True, n_frame=None)
        render_cameras = normalize_cameras(render_cameras, ref_w2c=ref_w2c, T_norm=T_norm, n_frame=None)

        text = "[Static] " + text
        print(f"Text prompt: {text}")

        text_embeds = self.encode_text([text])
        
        # Clear cache after text encoding
        torch.cuda.empty_cache()
        
        masks = torch.zeros(batch_size, n_frame, device=self.device)
        condition_latents = torch.zeros(batch_size, self.latent_dim, n_frame, 
                                       image_height // self.spatial_downsample_factor, 
                                       image_width // self.spatial_downsample_factor, 
                                       device=self.device)

        if image is not None:
            print("Encoding image...")
            image = image.to(self.device)
            
            # Encode on CPU if VAE is offloaded
            if self.offload_vae:
                latent = self.latent_scale_fn(self.vae.encode(
                        image.unsqueeze(0).unsqueeze(2).to("cpu").float()
                    ).latent_dist.sample().to(self.device)).squeeze(2)
            else:
                latent = self.latent_scale_fn(self.vae.encode(
                        image.unsqueeze(0).unsqueeze(2).to(self.device).float()
                    ).latent_dist.sample().to(self.device)).squeeze(2)
                
            masks[:, image_index] = 1
            condition_latents[:, :, image_index] = latent
            
            torch.cuda.empty_cache()

        raymaps = create_raymaps(cameras, image_height // self.spatial_downsample_factor, 
                                 image_width // self.spatial_downsample_factor)
        raymaps = einops.rearrange(raymaps, 'B T H W C -> B C T H W', T=n_frame)
        
        noise = torch.randn(batch_size, self.latent_dim, n_frame, 
                           image_height // self.spatial_downsample_factor, 
                           image_width // self.spatial_downsample_factor, 
                           device=self.device)
        noisy_latents = noise 
        
        torch.cuda.empty_cache()
        gc.collect()

        prev_latents_pred = torch.zeros(batch_size, self.latent_dim, n_frame, 
                                        image_height // self.spatial_downsample_factor, 
                                        image_width // self.spatial_downsample_factor, 
                                        device=self.device)
        prev_feats = torch.zeros(batch_size, self.feat_dim, n_frame, 
                                 image_height // self.spatial_downsample_factor, 
                                 image_width // self.spatial_downsample_factor, 
                                 device=self.device)

        for i in range(len(self.denoising_steps)):
            print(f"Denoising step {i+1}/{len(self.denoising_steps)}...")
            
            # Print memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                print(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            t_ids = torch.full((noisy_latents.shape[0],), self.denoising_steps[i], device=self.device)
            t = self.timesteps[t_ids]

            _condition_latents = torch.cat([condition_latents, prev_feats, prev_latents_pred], dim=1)

            if i < len(self.denoising_steps) - 1:
                out = self.transformer(
                    hidden_states=torch.cat([noisy_latents, raymaps, _condition_latents], dim=1),
                    timestep=t, encoder_hidden_states=text_embeds, return_dict=False,
                )[0]

                v_pred, feats = out.split([self.latent_dim, self.feat_dim], dim=1)
                sigma = torch.stack([self.scheduler.sigmas[self.scheduler.index_for_timestep(_t)] 
                                    for _t in t.unbind(0)], dim=0).to(self.device)
                latents_pred_2d = noisy_latents - sigma * v_pred

                scene_params = self.recon_decoder(
                    einops.rearrange(feats, 'B C T H W -> (B T) C H W').unsqueeze(2), 
                    einops.rearrange(self.latent_unscale_fn(latents_pred_2d.detach()), 
                                    'B C T H W -> (B T) C H W').unsqueeze(2), 
                    cameras
                ).flatten(1, -2)

                images_pred, _ = self.recon_decoder.render(scene_params.unbind(0), cameras, 
                                                           image_height, image_width, bg_mode="white")

                # Encode on CPU if VAE is offloaded
                if self.offload_vae:
                    latents_pred_3d = einops.rearrange(self.latent_scale_fn(self.vae.encode(
                        einops.rearrange(images_pred, 'B T C H W -> (B T) C H W', T=images_pred.shape[1]).unsqueeze(2).to("cpu").float()
                    ).latent_dist.sample().to(self.device)).squeeze(2), 
                    '(B T) C H W -> B C T H W', T=images_pred.shape[1]).to(noisy_latents.dtype)
                else:
                    latents_pred_3d = einops.rearrange(self.latent_scale_fn(self.vae.encode(
                        einops.rearrange(images_pred, 'B T C H W -> (B T) C H W', T=images_pred.shape[1]).unsqueeze(2).to(self.device).float()
                    ).latent_dist.sample().to(self.device)).squeeze(2), 
                    '(B T) C H W -> B C T H W', T=images_pred.shape[1]).to(noisy_latents.dtype)

                prev_latents_pred = latents_pred_3d
                prev_feats = feats
               
                noisy_latents = self.scheduler.scale_noise(
                    latents_pred_3d, 
                    self.timesteps[torch.full((noisy_latents.shape[0],), self.denoising_steps[i + 1], device=self.device)], 
                    torch.randn_like(noise)
                )
                
                # Clean up intermediate results
                del out, v_pred, feats, latents_pred_2d, scene_params, images_pred, latents_pred_3d
                torch.cuda.empty_cache()
                
            else:
                out = self.transformer(
                    hidden_states=torch.cat([noisy_latents, raymaps, _condition_latents], dim=1),
                    timestep=t, encoder_hidden_states=text_embeds, return_dict=False,
                )[0]

                v_pred, feats = out.split([self.latent_dim, self.feat_dim], dim=1)
                sigma = torch.stack([self.scheduler.sigmas[self.scheduler.index_for_timestep(_t)] 
                                    for _t in t.unbind(0)], dim=0).to(self.device)
                latents_pred = noisy_latents - sigma * v_pred

                scene_params = self.recon_decoder(
                    einops.rearrange(feats, 'B C T H W -> (B T) C H W').unsqueeze(2), 
                    einops.rearrange(self.latent_unscale_fn(latents_pred.detach()), 
                                    'B C T H W -> (B T) C H W').unsqueeze(2), 
                    cameras
                ).flatten(1, -2)

        scene_params = scene_params[0].detach().cpu()
        print("Generation complete!")
        
        return scene_params, ref_w2c, T_norm


def main():
    parser = argparse.ArgumentParser(description='Test FlashWorld generation (low memory)')
    parser.add_argument('--trajectory', type=str, required=True, help='Path to camera trajectory JSON file')
    parser.add_argument('--image', type=str, default=None, help='Path to input image (optional)')
    parser.add_argument('--text', type=str, default="", help='Text prompt')
    parser.add_argument('--image_index', type=int, default=0, help='Frame index for image condition')
    parser.add_argument('--output', type=str, default='output.ply', help='Output PLY file path')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--no-offload-t5', action='store_true', help='Keep T5 on GPU (uses more VRAM)')
    parser.add_argument('--no-offload-vae', action='store_true', help='Keep VAE on GPU (uses more VRAM)')
    parser.add_argument('--no-reduce-frames', action='store_true', help='Use full frame count (uses more VRAM)')
    parser.add_argument('--fewer-steps', action='store_true', help='Use only 2 denoising steps instead of 4 (faster, less quality)')
    args = parser.parse_args()

    # Load checkpoint
    if args.ckpt is None:
        ckpt_path = os.path.join(HUGGINGFACE_HUB_CACHE, 
                                "models--imlixinyang--FlashWorld", "snapshots", 
                                "6a8e88c6f88678ac098e4c82675f0aee555d6e5d", "model.ckpt")
        if not os.path.exists(ckpt_path):
            print("Downloading model checkpoint...")
            hf_hub_download(repo_id="imlixinyang/FlashWorld", filename="model.ckpt", local_dir_use_symlinks=False)
    else:
        ckpt_path = args.ckpt

    # Load trajectory
    print(f"Loading trajectory from {args.trajectory}")
    with open(args.trajectory, 'r') as f:
        traj_data = json.load(f)
    
    cameras_data = traj_data['cameras']
    resolution = traj_data.get('resolution', [24, 480, 704])
    n_frame, image_height, image_width = resolution

    # Convert cameras to tensor
    cameras = torch.stack([
        torch.from_numpy(np.array([
            cam['quaternion'][0], cam['quaternion'][1], cam['quaternion'][2], cam['quaternion'][3],
            cam['position'][0], cam['position'][1], cam['position'][2],
            cam.get('fx', image_width) / image_width, 
            cam.get('fy', image_height) / image_height,
            cam.get('cx', image_width/2) / image_width,
            cam.get('cy', image_height/2) / image_height
        ], dtype=np.float32))
        for cam in cameras_data
    ], dim=0)

    # Load image if provided
    image = None
    if args.image:
        print(f"Loading image from {args.image}")
        img = Image.open(args.image).convert('RGB')
        w, h = img.size
        
        # Center crop and resize
        if image_height / h > image_width / w:
            scale = image_height / h
        else:
            scale = image_width / w
        
        new_h = int(image_height / scale)
        new_w = int(image_width / scale)
        
        img = img.crop(((w - new_w) // 2, (h - new_h) // 2, 
                       new_w + (w - new_w) // 2, new_h + (h - new_h) // 2)).resize((image_width, image_height))
        
        image = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0 * 2 - 1

    # Initialize model
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = GenerationSystem(
        ckpt_path=ckpt_path, 
        device=device,
        offload_t5=not args.no_offload_t5,
        offload_vae=not args.no_offload_vae,
        reduce_frames=not args.no_reduce_frames,
        use_fewer_steps=args.fewer_steps
    )

    # Generate
    scene_params, ref_w2c, T_norm = model.generate(
        cameras, n_frame, image, args.text, args.image_index, image_height, image_width
    )

    # Export PLY
    print(f"Exporting to {args.output}")
    export_ply_for_gaussians(args.output, scene_params, opacity_threshold=0.001, T_norm=T_norm)
    
    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Success! Generated {args.output} ({file_size:.2f} MB)")


if __name__ == "__main__":
    main()
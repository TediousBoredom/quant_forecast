"""
Inference script for Quantized Diffusion Model
量化扩散模型推理脚本

This script performs inference with a trained quantized diffusion model.
It supports:
1. Text-to-video generation
2. Batch inference
3. Quantized latent visualization
4. Multiple sampling strategies (DDIM, DDPM)
"""

import os
import argparse
import json
from pathlib import Path
from typing import Optional, List, Tuple
import time

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import imageio

from quantized_diffusion_model import (
    QuantizedDiffusionPredictor,
    QuantizedDiffusionDMD,
)


class QuantizedDiffusionInference:
    """
    Inference engine for Quantized Diffusion Model.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize inference engine.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to use ('cuda' or 'cpu')
            dtype: Data type for inference
        """
        self.device = torch.device(device)
        self.dtype = dtype
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.config = checkpoint['config']
        
        # Create model
        self.model = QuantizedDiffusionPredictor(
            latent_channels=self.config['model']['latent_channels'],
            num_embeddings=self.config['model']['num_embeddings'],
            embedding_dim=self.config['model']['embedding_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            num_heads=self.config['model']['num_heads'],
            dropout=0.0,  # No dropout during inference
            max_frames=self.config['model']['max_frames'],
        ).to(self.device).to(dtype)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Create DMD wrapper for sampling
        self.dmd = QuantizedDiffusionDMD(
            model=self.model,
            device=self.device,
            num_train_timesteps=self.config['diffusion']['num_train_timesteps'],
            min_step=self.config['diffusion']['min_step'],
            max_step=self.config['diffusion']['max_step'],
            beta_schedule=self.config['diffusion']['beta_schedule'],
        )
        
        print(f"Model loaded successfully!")
        print(f"  - Codebook size: {self.config['model']['num_embeddings']}")
        print(f"  - Embedding dim: {self.config['model']['embedding_dim']}")
        print(f"  - Hidden dim: {self.config['model']['hidden_dim']}")
    
    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        num_frames: int = 16,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        return_indices: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate video latents.
        
        Args:
            batch_size: Number of videos to generate
            num_frames: Number of frames
            height: Video height (in latent space)
            width: Video width (in latent space)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale (not used in current implementation)
            seed: Random seed for reproducibility
            return_indices: Whether to return quantized indices
            
        Returns:
            latents: Generated latents [B, C, F, H, W]
            indices: Quantized indices [B, F, H, W] (if return_indices=True)
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Calculate latent dimensions
        latent_h = height // 8  # Assuming 8x downsampling
        latent_w = width // 8
        
        shape = (
            batch_size,
            self.config['model']['latent_channels'],
            num_frames,
            latent_h,
            latent_w,
        )
        
        print(f"Generating {batch_size} video(s) with shape {shape}")
        print(f"Using {num_inference_steps} inference steps")
        
        start_time = time.time()
        
        # Sample using DDIM
        latents = self.dmd.sample(
            shape=shape,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        
        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f}s")
        
        # Get quantized indices if requested
        indices = None
        if return_indices:
            # Create dummy timesteps (use min_step for final prediction)
            timesteps = torch.full(
                (batch_size,),
                self.config['diffusion']['min_step'],
                device=self.device,
                dtype=torch.long,
            )
            indices = self.model.encode_to_indices(latents, timesteps)
        
        return latents, indices
    
    @torch.no_grad()
    def encode_to_codes(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode latents to discrete codes.
        
        Args:
            latents: Input latents [B, C, F, H, W]
            
        Returns:
            indices: Discrete codes [B, F, H, W]
        """
        B = latents.shape[0]
        timesteps = torch.full(
            (B,),
            self.config['diffusion']['min_step'],
            device=self.device,
            dtype=torch.long,
        )
        
        indices = self.model.encode_to_indices(latents, timesteps)
        return indices
    
    @torch.no_grad()
    def decode_from_codes(
        self,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode discrete codes to latents.
        
        Args:
            indices: Discrete codes [B, F, H, W]
            
        Returns:
            latents: Decoded latents [B, C, F, H, W]
        """
        latents = self.model.decode_from_indices(indices)
        return latents
    
    def visualize_latents(
        self,
        latents: torch.Tensor,
        output_path: str,
        normalize: bool = True,
    ):
        """
        Visualize latents as video.
        
        Args:
            latents: Latents to visualize [B, C, F, H, W]
            output_path: Path to save video
            normalize: Whether to normalize latents
        """
        B, C, F, H, W = latents.shape
        
        # Take first batch and first 3 channels for RGB visualization
        latent = latents[0, :3].cpu().numpy()  # [3, F, H, W]
        
        if normalize:
            # Normalize to [0, 1]
            latent = (latent - latent.min()) / (latent.max() - latent.min() + 1e-8)
        
        # Convert to uint8
        latent = (latent * 255).astype(np.uint8)
        
        # Transpose to [F, H, W, 3]
        frames = latent.transpose(1, 2, 3, 0)
        
        # Save as video
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        imageio.mimsave(output_path, frames, fps=8)
        print(f"Saved visualization to {output_path}")
    
    def visualize_codebook_usage(
        self,
        indices: torch.Tensor,
        output_path: str,
    ):
        """
        Visualize codebook usage statistics.
        
        Args:
            indices: Codebook indices [B, F, H, W]
            output_path: Path to save visualization
        """
        # Flatten indices
        indices_flat = indices.cpu().numpy().flatten()
        
        # Count usage
        unique, counts = np.unique(indices_flat, return_counts=True)
        
        # Calculate statistics
        total_codes = self.config['model']['num_embeddings']
        used_codes = len(unique)
        usage_ratio = used_codes / total_codes
        
        print(f"\nCodebook Usage Statistics:")
        print(f"  Total codes: {total_codes}")
        print(f"  Used codes: {used_codes}")
        print(f"  Usage ratio: {usage_ratio:.2%}")
        print(f"  Most common code: {unique[np.argmax(counts)]} (used {counts.max()} times)")
        print(f"  Least common code: {unique[np.argmin(counts)]} (used {counts.min()} times)")
        
        # Save statistics
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'total_codes': int(total_codes),
            'used_codes': int(used_codes),
            'usage_ratio': float(usage_ratio),
            'unique_codes': unique.tolist(),
            'counts': counts.tolist(),
        }
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Saved statistics to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Inference with Quantized Diffusion Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./outputs/inference', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames')
    parser.add_argument('--height', type=int, default=256, help='Video height')
    parser.add_argument('--width', type=int, default=256, help='Video width')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance scale')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--visualize', action='store_true', help='Visualize latents')
    parser.add_argument('--analyze_codebook', action='store_true', help='Analyze codebook usage')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize inference engine
    inference = QuantizedDiffusionInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    
    # Generate samples
    for i in range(args.num_samples):
        print(f"\n{'='*60}")
        print(f"Generating sample {i+1}/{args.num_samples}")
        print(f"{'='*60}")
        
        # Set seed for reproducibility
        seed = args.seed + i if args.seed is not None else None
        
        # Generate
        latents, indices = inference.generate(
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=seed,
            return_indices=args.analyze_codebook,
        )
        
        # Save latents
        latent_path = output_dir / f'sample_{i:04d}_latents.pt'
        torch.save(latents.cpu(), latent_path)
        print(f"Saved latents to {latent_path}")
        
        # Visualize if requested
        if args.visualize:
            video_path = output_dir / f'sample_{i:04d}_visualization.mp4'
            inference.visualize_latents(latents, video_path)
        
        # Analyze codebook usage if requested
        if args.analyze_codebook and indices is not None:
            stats_path = output_dir / f'sample_{i:04d}_codebook_stats.json'
            inference.visualize_codebook_usage(indices, stats_path)
            
            # Save indices
            indices_path = output_dir / f'sample_{i:04d}_indices.pt'
            torch.save(indices.cpu(), indices_path)
            print(f"Saved indices to {indices_path}")
    
    print(f"\n{'='*60}")
    print(f"Inference completed! Results saved to {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

